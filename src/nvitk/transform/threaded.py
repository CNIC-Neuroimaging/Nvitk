"""
Thread-parallel ``affine_transform`` and ``map_coordinates`` for host arrays.

Description
-----------
Drop-in replacements for the two ``scipy.ndimage`` interpolators nvitk
resamples with. Every output voxel is computed independently of the others, so
the output is split into slabs along its first axis and each slab is
interpolated on the shared worker pool (:mod:`nvitk.core.parallel`).
``scipy.ndimage`` releases the GIL inside these kernels, so the slabs genuinely
run concurrently (~7x on 8 threads).

Exactness and reproducibility
-----------------------------
- :func:`map_coordinates` is **bit-identical** to SciPy: the coordinates are
  explicit, so a slab samples exactly the points the whole call would.
- :func:`affine_transform` folds each slab's start row into the offset, which
  rounds differently in the last bit. Values agree to float precision, except a
  voxel whose source coordinate lands within rounding of the input's edge: in
  ``mode="constant"`` that voxel can flip between its value and *cval* (measured:
  0.005-0.01 % of voxels on an oblique 160x256x256 resample, all on the field-of-
  view edge). It is the same coin-flip SciPy makes at that position, just decided
  the other way.
- The slab layout depends on the output **shape only**, never on the thread
  count, so a given input resamples to the same array on a laptop and on a
  32-core workstation; the budget only decides how many slabs run at once.

Array / axis conventions
------------------------
Unchanged from ``scipy.ndimage``: *matrix*/*offset* map **output** voxel indices
to **input** voxel indices, and *coordinates* has shape ``(input.ndim, *out)``.

Backends
--------
A CuPy input is handed straight to ``cupyx.scipy.ndimage`` — the GPU kernel is
already parallel and slabbing it would only add launches. Host inputs use the
NumPy/SciPy modules from :func:`~nvitk.core.backend.get_backend_modules`, which
is safe inside worker threads (unlike :class:`~nvitk.core.backend.using`, which
rewrites module globals process-wide).

Spline orders above 1
---------------------
``prefilter=True`` with ``order > 1`` runs a whole-volume spline filter. Done per
slab it would repeat that filter once per slab, so it is run **once** up front
and the slabs are interpolated with ``prefilter=False`` on the filtered
coefficients — exactly what SciPy does internally.
"""

from __future__ import annotations

from typing import Any

from nvitk.core.array import to_numpy
from nvitk.core.backend import get_backend_modules, is_cupy_array
from nvitk.core.parallel import MIN_PARALLEL_ELEMENTS, chunk_bounds, parallel_map

# Host modules, resolved once: these functions only ever run the host path on
# host arrays, and must not depend on which backend a caller happens to be in.
_HOST = get_backend_modules("numpy")
_hnp = _HOST.xp
_hndi = _HOST.ndi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


#: Slabs per call for a large output — fixed, so the decomposition (and with it
#: every rounding decision) is the same whatever the thread budget. 64 keeps a
#: 32-thread pool busy with headroom for uneven slab cost.
_SLABS = 64


def _slab_count(n_out: int, first_axis: int) -> int:
    """How many slabs to cut the output into (1 = hand the whole call to SciPy).

    A function of the output geometry alone — see *Exactness and
    reproducibility* in the module docstring.
    """
    if n_out < MIN_PARALLEL_ELEMENTS or first_axis < 2:
        return 1
    return max(1, min(first_axis, _SLABS))


#: Modes for which SciPy pre-pads before spline filtering and passes the pad
#: width to its C kernel — not reproducible through the public API, so these
#: run serially rather than risk a different answer at the borders.
_PREPADDED_MODES = frozenset({"nearest", "grid-constant"})


def _needs_serial(order: int, mode: str, prefilter: bool) -> bool:
    """Whether slabbing could change the result for these interpolation settings."""
    return bool(prefilter) and int(order) > 1 and str(mode) in _PREPADDED_MODES


def _output_array(output: Any, shape: tuple[int, ...], default_dtype: Any) -> Any:
    """The array slabs write into: the caller's, or a new one of the asked dtype."""
    if isinstance(output, _hnp.ndarray):
        if tuple(output.shape) != tuple(shape):
            raise ValueError(f"output has shape {tuple(output.shape)}, expected {tuple(shape)}.")
        return output
    return _hnp.empty(shape, dtype=output if output is not None else default_dtype)


def _filtered_input(data: Any, order: int, mode: str, prefilter: bool) -> tuple[Any, bool]:
    """``(coefficients, prefilter_flag)`` — spline-filter once when the order needs it."""
    if prefilter and int(order) > 1:
        coeffs = _hndi.spline_filter(data, order=int(order), output=_hnp.float64, mode=mode)
        return coeffs, False
    return data, bool(prefilter)


# ---------------------------------------------------------------------------
# affine_transform
# ---------------------------------------------------------------------------


def affine_transform(
    input: Any,  # noqa: A002 — mirrors scipy.ndimage's signature
    matrix: Any,
    offset: Any = 0.0,
    output_shape: tuple[int, ...] | None = None,
    *,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
    output: Any = None,
    workers: int | None = None,
) -> Any:
    """
    ``scipy.ndimage.affine_transform``, slab-parallel over output axis 0.

    Parameters
    ----------
    input
        Source array (host or CuPy).
    matrix
        ``(ndim, ndim)`` linear part, or ``(ndim+1, ndim+1)`` homogeneous matrix,
        mapping output indices to input indices. A 1-D diagonal is accepted too.
    offset
        Translation added after *matrix* (ignored with a homogeneous matrix).
    output_shape
        Shape of the result; defaults to *input*'s shape.
    order, mode, cval, prefilter
        As in SciPy.
    output
        Optional output dtype or preallocated array of the output shape.
    workers
        Thread budget override; default :func:`~nvitk.core.parallel.get_worker_count`.

    Returns
    -------
    ndarray
        SciPy's values, up to the edge rounding described in the module docstring.
    """
    if is_cupy_array(input):
        cu = get_backend_modules("cupy")
        return cu.ndi.affine_transform(
            input, matrix, offset=offset, output_shape=output_shape, order=order,
            mode=mode, cval=cval, prefilter=prefilter, output=output,
        )

    data = to_numpy(input)
    ndim = data.ndim
    out_shape = tuple(int(s) for s in (output_shape if output_shape is not None else data.shape))
    mat = to_numpy(matrix).astype(_hnp.float64, copy=False)

    # ---- 1. Normalise to (linear, offset) --------------------------------------
    if mat.ndim == 1:
        linear = _hnp.diag(mat)
        shift = _hnp.broadcast_to(to_numpy(offset).astype(_hnp.float64), (ndim,)).copy()
    elif mat.shape == (ndim + 1, ndim + 1):
        linear = mat[:ndim, :ndim]
        shift = mat[:ndim, ndim].copy()
    else:
        linear = mat
        shift = _hnp.broadcast_to(to_numpy(offset).astype(_hnp.float64), (ndim,)).copy()

    n_out = int(_hnp.prod(out_shape)) if out_shape else 0
    slabs = _slab_count(n_out, out_shape[0] if out_shape else 0)
    if slabs <= 1 or _needs_serial(order, mode, prefilter):
        return _hndi.affine_transform(
            data, linear, offset=shift, output_shape=out_shape, order=int(order),
            mode=mode, cval=cval, prefilter=prefilter, output=output,
        )

    # ---- 2. Spline-filter once, then interpolate slab by slab ------------------
    coeffs, slab_prefilter = _filtered_input(data, order, mode, prefilter)
    result = _output_array(output, out_shape, data.dtype)

    def _slab(bounds: tuple[int, int]) -> None:
        start, stop = bounds
        # Output row i of the slab is global row start + i: fold the start into
        # the offset so the slab maps through exactly the same transform.
        slab_shift = shift + linear[:, 0] * float(start)
        _hndi.affine_transform(
            coeffs, linear, offset=slab_shift, output_shape=(stop - start, *out_shape[1:]),
            order=int(order), mode=mode, cval=cval, prefilter=slab_prefilter,
            output=result[start:stop],
        )

    parallel_map(_slab, chunk_bounds(out_shape[0], slabs), workers=workers)
    return result


# ---------------------------------------------------------------------------
# map_coordinates
# ---------------------------------------------------------------------------


def map_coordinates(
    input: Any,  # noqa: A002 — mirrors scipy.ndimage's signature
    coordinates: Any,
    *,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
    output: Any = None,
    workers: int | None = None,
) -> Any:
    """
    ``scipy.ndimage.map_coordinates``, parallel over the first output axis.

    *coordinates* has shape ``(input.ndim, *out_shape)``; the result has
    ``out_shape``. Bit-identical to SciPy's result.
    """
    if is_cupy_array(input):
        cu = get_backend_modules("cupy")
        return cu.ndi.map_coordinates(
            input, coordinates, order=order, mode=mode, cval=cval,
            prefilter=prefilter, output=output,
        )

    data = to_numpy(input)
    coords = to_numpy(coordinates)
    out_shape = tuple(int(s) for s in coords.shape[1:])
    n_out = int(_hnp.prod(out_shape)) if out_shape else 0
    slabs = _slab_count(n_out, out_shape[0] if out_shape else 0)
    if slabs <= 1 or _needs_serial(order, mode, prefilter):
        return _hndi.map_coordinates(
            data, coords, order=int(order), mode=mode, cval=cval,
            prefilter=prefilter, output=output,
        )

    coeffs, slab_prefilter = _filtered_input(data, order, mode, prefilter)
    result = _output_array(output, out_shape, data.dtype)

    def _slab(bounds: tuple[int, int]) -> None:
        start, stop = bounds
        _hndi.map_coordinates(
            coeffs, coords[:, start:stop], order=int(order), mode=mode, cval=cval,
            prefilter=slab_prefilter, output=result[start:stop],
        )

    parallel_map(_slab, chunk_bounds(out_shape[0], slabs), workers=workers)
    return result


__all__ = ["affine_transform", "map_coordinates"]
