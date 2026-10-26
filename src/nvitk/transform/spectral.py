"""
Spectral CT results from a photoelectric / Compton-scatter basis pair.

Description
-----------
A dual-energy (or dual-layer) CT measures attenuation at two effective energies, and in the
diagnostic range every material's attenuation is, to a good approximation, a mix of two
processes with fixed energy dependences: photoelectric absorption (falling as ~E^-3) and
Compton scatter (the Klein-Nishina cross-section, proportional to electron density). The
Philips spectral base images (:mod:`nvitk.io.conversors.philips_sbi`) store exactly that pair:
for each voxel, how much of the attenuation at a reference energy each process contributes
(HU + 1000 units, so the two add up to the conventional image). Everything else follows from
the pair:

* **virtual monoenergetic** images at any energy (40-200 keV): a linear combination of the two;
* **iodine density** (mg/mL) and **virtual non-contrast** (VNC, HU): a water / iodine
  two-material decomposition;
* **effective atomic number** (Z_eff): from the photoelectric-to-scatter ratio per electron;
* **electron density** (% of water's): from the scatter part alone;
* the **spectral curve** of a voxel or a region: HU against keV.

The physics model
-----------------
Water's attenuation (NIST, Hubbell & Seltzer) is fitted as ``a (E_ref/E)^n + b KN(E)/KN(E_ref)``
(photoelectric-like and Klein-Nishina; within 0.2 % from 30 to 200 keV); iodine's (NIST, above
its K edge) on the same two functions. A vendor's basis pair need not be the pure physical one
(each of its two images may carry a little of the other process), so the pair is mapped onto
the physical basis by two constants of the vendor's decomposition (:class:`SpectralCalibration`):
the share of water's attenuation in the photoelectric image, and the share of iodine's. For
Philips SBI V5.1 at 120 kVp they were measured on a clinical cardiac study (water 0.100 from
the fat-soft-tissue line, iodine 0.957 from the contrast-filled blood pool against soft tissue,
the same within 1 % across three reconstructions), with the conventional image taken as the
70 keV equivalent. With them, water is flat at 0 HU at every energy, iodine follows its NIST
curve, and the 70 keV image reproduces the conventional one.

**These are nvitk's derivations, not the vendor's calibrated results** (Philips derives them
with proprietary lookup tables). Expect the same behaviour (iodine bright at low keV, bone
and iodine separated from water, Z_eff ~7.4 for soft tissue), and values within the model's
accuracy, not the workstation's numbers to the HU.

Arrays follow the current backend (``setup(globals())``): the inputs are made backend arrays,
so on the GPU every result is computed there.

Example::

    from nvitk.transform.spectral import monoenergetic, iodine_density, SpectralCalibration

    mono40 = monoenergetic(sbi.photo, sbi.scatter, 40)          # HU
    iodine = iodine_density(sbi.photo, sbi.scatter)             # mg/mL
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Callable, Sequence

from nvitk.core.array import as_backend_array
from nvitk.core.backend import get_current_backend, setup, using

setup(globals())

#: Energies (keV) of the NIST tables below.
_E_TABLE = (30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 200.0)
#: Water, total mass attenuation with coherent scattering (cm²/g; NIST, Hubbell & Seltzer).
_WATER = (0.3756, 0.2683, 0.2269, 0.2059, 0.1837, 0.1707, 0.1505, 0.1370)
#: Iodine (Z = 53), same (cm²/g); 30 keV is below its K edge (None: not used).
_IODINE = (None, 22.10, 12.32, 7.614, 3.612, 1.942, 0.6978, 0.3663)
#: Classical electron radius squared (cm²) and the electron rest energy (keV).
_RE2 = 7.9407877e-26
_ME_KEV = 510.99895

#: Energies the monoenergetic results are offered at.
KEV_RANGE = (40.0, 200.0)


def klein_nishina(kev: Any) -> Any:
    """Klein-Nishina total cross-section per electron (cm²) at *kev*: host values: the
    model's constants are a handful of scalars, computed on the CPU."""
    with using("cpu"):
        k = np.asarray(kev, dtype=float) / _ME_KEV
        a = (1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - np.log1p(2 * k) / k)
        b = np.log1p(2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k) ** 2
        return 2 * np.pi * _RE2 * (a + b)


@dataclass(frozen=True)
class SpectralCalibration:
    """The constants that tie a vendor's photoelectric / scatter pair to physics.

    ``reference_kev``: the energy the pair's contributions refer to (the conventional
    image's equivalent). ``water_photo_share`` / ``iodine_photo_share``: the share of water's
    and of iodine's attenuation that the vendor's *photoelectric* image carries.
    ``zeff_exponent`` / ``zeff_water``: the power law of Z_eff (Mayneord) and water's Z_eff.
    """

    reference_kev: float = 70.0
    water_photo_share: float = 0.100
    iodine_photo_share: float = 0.957
    zeff_exponent: float = 2.94
    zeff_water: float = 7.42

    def with_(self, **changes: Any) -> "SpectralCalibration":
        return replace(self, **changes)


#: Philips SBI V5.1, 120 kVp (see the module notes).
PHILIPS_SBI_V51 = SpectralCalibration()


@dataclass(frozen=True)
class _Model:
    """The fitted physics for one calibration (host scalars)."""

    exponent: float           # photoelectric-like power law
    water_photo: float        # water's physical photoelectric share at E_ref
    water_mu_ref: float       # water's attenuation at E_ref (model, cm²/g)
    water_a: float            # water = a (E_ref/E)^n + b KN(E)/KN(E_ref)
    water_b: float
    iodine_hu_per_mg: float   # HU per mg/mL of iodine at E_ref
    to_physical: tuple[float, float, float, float]   # (u1, u2, v1, v2): vendor (P, S) -> physical (p, c)


def _fit_two(mu: Any, energies: Any, e_ref: float, exponent: float) -> tuple[float, float]:
    """Least-squares (relative) weights of ``(E_ref/E)^n`` and ``KN(E)/KN(E_ref)`` for *mu* (host)."""
    with using("cpu"):
        mu, energies = np.asarray(mu, dtype=float), np.asarray(energies, dtype=float)
        basis = np.stack([(e_ref / energies) ** exponent, klein_nishina(energies) / klein_nishina(e_ref)], axis=1)
        a, b = np.linalg.lstsq(basis / mu[:, None], np.ones_like(mu), rcond=None)[0]
        return float(a), float(b)


@lru_cache(maxsize=16)
def _model(calibration: SpectralCalibration) -> _Model:
    e_ref = float(calibration.reference_kev)
    with using("cpu"):
        energies, water = np.asarray(_E_TABLE), np.asarray(_WATER)
        kn_ratio = klein_nishina(energies) / klein_nishina(e_ref)
        # Water: the photoelectric power law that fits best (relative error), with KN.
        best = None
        for exponent in np.arange(2.5, 3.51, 0.01):
            a, b = _fit_two(water, energies, e_ref, float(exponent))
            fit = a * (e_ref / energies) ** exponent + b * kn_ratio
            err = float(np.max(np.abs(fit / water - 1)))
            if best is None or err < best[0]:
                best = (err, float(exponent), a, b)
        _err, n, wa, wb = best  # type: ignore[misc]
        water_mu = wa + wb
        water_photo = wa / water_mu
        # Iodine above its K edge, on the same two functions.
        keep = [k for k, v in enumerate(_IODINE) if v is not None]
        ia, ib = _fit_two([_IODINE[k] for k in keep], [_E_TABLE[k] for k in keep], e_ref, n)
        iodine_photo = ia / (ia + ib)
        hu_per_mg = 1000.0 * 1e-3 * (ia + ib) / water_mu
        # Vendor photoelectric image P and scatter image S -> physical (p, c):
        #   p = u1 P + u2 S, c = v1 P + v2 S, with u + v = 1 for each image (both are
        # attenuation at E_ref), water and iodine landing on their physical shares.
        pw, pi = float(calibration.water_photo_share), float(calibration.iodine_photo_share)
        u1, u2 = (float(v) for v in np.linalg.solve(np.asarray([[pw, 1 - pw], [pi, 1 - pi]]),
                                                    np.asarray([water_photo, iodine_photo])))
    return _Model(n, water_photo, water_mu, wa, wb, hu_per_mg, (u1, u2, 1 - u1, 1 - u2))


def model_summary(calibration: SpectralCalibration = PHILIPS_SBI_V51) -> dict[str, float]:
    """The fitted constants, for reports and checks."""
    m = _model(calibration)
    return {
        "photoelectric_exponent": m.exponent,
        "water_photoelectric_share_physical": m.water_photo,
        "iodine_hu_per_mg_ml_at_reference": m.iodine_hu_per_mg,
        "u_photo": m.to_physical[0],
        "u_scatter": m.to_physical[1],
    }


def monoenergetic_weights(kev: float, calibration: SpectralCalibration = PHILIPS_SBI_V51) -> tuple[float, float, float]:
    """``(w_photo, w_scatter, offset)`` with HU(kev) = w_photo·photo + w_scatter·scatter + offset,
    *photo* / *scatter* in the SBI convention (contribution - 1000)."""
    m = _model(calibration)
    e_ref, e = float(calibration.reference_kev), float(kev)
    phi_p = (e_ref / e) ** m.exponent
    phi_c = float(klein_nishina(e) / klein_nishina(e_ref))
    water_e = m.water_a * phi_p + m.water_b * phi_c          # model water at E (cm²/g)
    ratio = m.water_mu_ref / water_e
    u1, u2, v1, v2 = m.to_physical
    w_p = (u1 * phi_p + v1 * phi_c) * ratio
    w_s = (u2 * phi_p + v2 * phi_c) * ratio
    return w_p, w_s, (w_p + w_s) * 1000.0 - 1000.0


def _on_gpu() -> bool:
    return str(get_current_backend()).lower() not in ("numpy", "cpu")


def _chunked(compute: Callable[[Any, Any], Any], photo: Any, scatter: Any, out: Any = None) -> Any:
    """``compute(photo, scatter)`` element-wise as float32: in one go on the GPU; on the CPU
    in slabs along the last axis on worker threads (NumPy releases the GIL), written into
    *out* when given (a float32 array of the same shape); the volumes are hundreds of MB."""
    if _on_gpu():
        result = compute(as_backend_array(photo).astype(np.float32), as_backend_array(scatter).astype(np.float32))
        result = result.astype(np.float32)
        if out is not None:
            out[...] = result
            return out
        return result
    photo, scatter = as_backend_array(photo), as_backend_array(scatter)
    if out is None:
        out = np.empty(photo.shape, dtype=np.float32)
    if photo.ndim == 0 or photo.size < 2_000_000:
        out[...] = compute(photo.astype(np.float32), scatter.astype(np.float32))
        return out
    n = photo.shape[-1]
    step = max(1, n // (4 * (os.cpu_count() or 4)))

    def slab(start: int) -> None:
        sl = (Ellipsis, slice(start, min(start + step, n)))
        out[sl] = compute(photo[sl].astype(np.float32), scatter[sl].astype(np.float32))

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
        list(pool.map(slab, range(0, n, step)))
    return out


def _linear(photo: Any, scatter: Any, a: float, b: float, c: float, out: Any = None) -> Any:
    """``a·photo + b·scatter + c`` (float32)."""
    a32, b32, c32 = np.float32(a), np.float32(b), np.float32(c)
    return _chunked(lambda p, s: p * a32 + s * b32 + c32, photo, scatter, out)


def monoenergetic(photo: Any, scatter: Any, kev: float,
                  calibration: SpectralCalibration = PHILIPS_SBI_V51, *, out: Any = None) -> Any:
    """Virtual monoenergetic image (HU) at *kev* (into *out* when given)."""
    w_p, w_s, offset = monoenergetic_weights(kev, calibration)
    return _linear(photo, scatter, w_p, w_s, offset, out)


def _decomposition_matrix(calibration: SpectralCalibration) -> tuple[tuple[float, float], tuple[float, float]]:
    """Inverse of [[water], [iodine]] in (P, S): (P, S) -> (water g/mL, iodine mg/mL) (host scalars)."""
    m = _model(calibration)
    pw, pi = float(calibration.water_photo_share), float(calibration.iodine_photo_share)
    k = m.iodine_hu_per_mg
    with using("cpu"):
        inv = np.linalg.inv(np.asarray([[1000 * pw, pi * k], [1000 * (1 - pw), (1 - pi) * k]]))
        return (float(inv[0, 0]), float(inv[0, 1])), (float(inv[1, 0]), float(inv[1, 1]))


def iodine_density(photo: Any, scatter: Any, calibration: SpectralCalibration = PHILIPS_SBI_V51,
                   *, out: Any = None) -> Any:
    """Iodine concentration (mg/mL) of the water / iodine decomposition. Calcium (bone)
    reads as iodine too, as on any water / iodine decomposition."""
    _water, (i_p, i_s) = _decomposition_matrix(calibration)
    # (P, S) = (photo, scatter) + 1000
    return _linear(photo, scatter, i_p, i_s, 1000.0 * (i_p + i_s), out)


def virtual_non_contrast(photo: Any, scatter: Any, calibration: SpectralCalibration = PHILIPS_SBI_V51,
                         *, out: Any = None) -> Any:
    """The image without its iodine (HU): the water part of the decomposition."""
    (w_p, w_s), _iodine = _decomposition_matrix(calibration)
    return _linear(photo, scatter, 1000.0 * w_p, 1000.0 * w_s, 1000.0 * 1000.0 * (w_p + w_s) - 1000.0, out)


def material_decomposition(photo: Any, scatter: Any,
                           calibration: SpectralCalibration = PHILIPS_SBI_V51) -> tuple[Any, Any]:
    """Water (g/mL) and iodine (mg/mL) whose mixture gives each voxel's pair."""
    (w_p, w_s), (i_p, i_s) = _decomposition_matrix(calibration)
    water = _linear(photo, scatter, w_p, w_s, 1000.0 * (w_p + w_s))
    iodine = _linear(photo, scatter, i_p, i_s, 1000.0 * (i_p + i_s))
    return water, iodine


def electron_density(photo: Any, scatter: Any, calibration: SpectralCalibration = PHILIPS_SBI_V51,
                     *, out: Any = None) -> Any:
    """Electron density relative to water's, in % (water 100; air 0)."""
    _u1, _u2, v1, v2 = _model(calibration).to_physical
    scale = 100.0 / (1000.0 * (1.0 - _model(calibration).water_photo))
    a, b = v1 * scale, v2 * scale
    zero = np.float32(0)
    a32, b32, c32 = np.float32(a), np.float32(b), np.float32(1000.0 * (a + b))
    return _chunked(lambda p, s: np.maximum(p * a32 + s * b32 + c32, zero), photo, scatter, out)


def effective_z(photo: Any, scatter: Any, calibration: SpectralCalibration = PHILIPS_SBI_V51,
                *, min_density: float = 0.2, out: Any = None) -> Any:
    """Effective atomic number (Mayneord); 0 where the electron density is under
    *min_density* of water's (air, where the ratio means nothing)."""
    m = _model(calibration)
    u1, u2, v1, v2 = m.to_physical
    water_ratio = m.water_photo / (1.0 - m.water_photo)
    floor = np.float32(min_density * 1000.0 * (1.0 - m.water_photo))
    f = [np.float32(v) for v in (u1, u2, v1, v2, 1000.0 * (u1 + u2), 1000.0 * (v1 + v2))]
    zw, inv_m = np.float32(calibration.zeff_water), np.float32(1.0 / calibration.zeff_exponent)
    wr, zero, tiny = np.float32(water_ratio), np.float32(0), np.float32(1e-3)

    def compute(p: Any, s: Any) -> Any:
        photo_part = p * f[0] + s * f[1] + f[4]
        compton_part = p * f[2] + s * f[3] + f[5]
        ratio = np.maximum(photo_part, zero) / np.maximum(compton_part, tiny)
        return np.where(compton_part > floor, zw * (ratio / wr) ** inv_m, zero)

    return _chunked(compute, photo, scatter, out)


def spectral_curve(photo: float, scatter: float, kevs: Sequence[float],
                   calibration: SpectralCalibration = PHILIPS_SBI_V51) -> list[float]:
    """HU against keV for one pair of values (a voxel, or a region's means)."""
    out = []
    for kev in kevs:
        w_p, w_s, offset = monoenergetic_weights(float(kev), calibration)
        out.append(float(w_p * float(photo) + w_s * float(scatter) + offset))
    return out


#: The results :func:`spectral_result` makes, with their units.
SPECTRAL_RESULTS: dict[str, str] = {
    "monoenergetic": "HU",
    "conventional": "HU",
    "iodine": "mg/mL",
    "vnc": "HU",
    "zeff": "",
    "electron_density": "% water",
    "photo": "HU",
    "scatter": "HU",
}


def spectral_result(name: str, photo: Any, scatter: Any, *, kev: float = 70.0,
                    conventional: Any = None,
                    calibration: SpectralCalibration = PHILIPS_SBI_V51, out: Any = None) -> Any:
    """One result by name (see :data:`SPECTRAL_RESULTS`), into *out* when given."""
    if name == "monoenergetic":
        return monoenergetic(photo, scatter, kev, calibration, out=out)
    if name == "iodine":
        return iodine_density(photo, scatter, calibration, out=out)
    if name == "vnc":
        return virtual_non_contrast(photo, scatter, calibration, out=out)
    if name == "zeff":
        return effective_z(photo, scatter, calibration, out=out)
    if name == "electron_density":
        return electron_density(photo, scatter, calibration, out=out)
    if name == "photo":
        return _linear(photo, scatter, 1.0, 0.0, 0.0, out)
    if name == "scatter":
        return _linear(photo, scatter, 0.0, 1.0, 0.0, out)
    if name == "conventional":
        if conventional is None:
            return monoenergetic(photo, scatter, calibration.reference_kev, calibration, out=out)
        return _linear(conventional, conventional, 1.0, 0.0, 0.0, out)
    raise ValueError(f"Unknown spectral result {name!r}; one of {sorted(SPECTRAL_RESULTS)}.")


__all__ = [
    "KEV_RANGE",
    "PHILIPS_SBI_V51",
    "SPECTRAL_RESULTS",
    "SpectralCalibration",
    "effective_z",
    "electron_density",
    "iodine_density",
    "klein_nishina",
    "material_decomposition",
    "model_summary",
    "monoenergetic",
    "monoenergetic_weights",
    "spectral_curve",
    "spectral_result",
    "virtual_non_contrast",
]
