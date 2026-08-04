"""The :class:`Image` container: voxel array plus metadata, axes, and optional orientation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import as_backend_array, is_cupy_array, to_cupy, to_numpy
from nvitk.core.exceptions import ValidationError


# ──────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ──────────────────────────────────────────────────────────────────────────────


def _is_array_like(value: Any) -> bool:
    """True for objects that quack like an ndarray (shape/dtype/positive ndim).

    Used to decide whether an indexing/ufunc result should be re-wrapped as an
    :class:`Image` or returned as a plain scalar.
    """
    if not (hasattr(value, "shape") and hasattr(value, "dtype")):
        return False
    ndim = getattr(value, "ndim", None)
    return isinstance(ndim, int) and ndim > 0


_NON_DICOM_METADATA_KEYS = {
    "axes",
    "shape",
    "affine",
    "spacing",
    "origin",
    "direction",
    "x_res",
    "y_res",
    "z_res",
    "t_res",
    "temporal_resolution",
    "series_uid",
    "series_description",
    "series_number",
    "submodality",
    "rescale_type",
    "filename",
    "dtype",
    "mode",
    "tiff_tags",
    "orientation",
}


def _is_dicom_tag_key(key: str) -> bool:
    """Return True if *key* looks like a DICOM tag name (group/element or capitalized tag)."""
    if key in _NON_DICOM_METADATA_KEYS:
        return False

    if key.startswith("(") and key.endswith(")") and "," in key:
        body = key[1:-1]
        group, element = body.split(",", 1)
        group = group.strip()
        element = element.strip()
        hexdigits = set("0123456789abcdefABCDEF")
        return len(group) == 4 and len(element) == 4 and all(ch in hexdigits for ch in (group + element))

    if "_" in key:
        return False
    return bool(key) and key[0].isupper()


# ──────────────────────────────────────────────────────────────────────────────
# Image
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Image:
    """
    Voxel data with a metadata dict, optional axis labels, and NumPy/CuPy interoperability.

    **Construction**

    - Pass a NumPy/CuPy array (and optional ``metadata`` / ``axes`` / ``name`` / ``orientation``).
    - Pass a path string or :class:`~pathlib.Path` to load via :func:`nvitk.io.imageio.imread`.

    **Behavior**

    Indexing and ufuncs return a new :class:`Image` when the result is array-like.
    Use :meth:`with_data` or :meth:`take` to replace voxels while merging metadata safely.
    """

    data: Any
    metadata: dict[str, Any] | None = None
    axes: str | None = None
    name: str | None = None
    orientation: str | None = None
    """Voxel axes orientation codes (e.g. ``\"RAS\"``, ``\"LPS\"``) from the image affine."""

    def __post_init__(self) -> None:
        """Normalize metadata, validate ``axes`` vs ``ndim``, and merge path loads."""
        # Allow direct construction from file path:
        #   Image("/path/to/scan.nii.gz")
        if isinstance(self.data, (str, Path)):
            source_path = Path(self.data)
            from nvitk.io.imageio import imread

            loaded = imread(str(source_path), axes=self.axes)
            if isinstance(loaded, list):
                raise ValidationError(
                    f"Path '{source_path}' contains multiple series. "
                    "Use imread(...) and select one series explicitly."
                )

            # Merge user-provided metadata on top of file metadata.
            user_metadata = dict(self.metadata or {})
            merged_metadata = dict(loaded.metadata or {})
            merged_metadata.update(user_metadata)

            self.data = loaded.data
            self.metadata = merged_metadata
            if self.axes is None:
                self.axes = loaded.axes
            if self.name is None:
                self.name = loaded.name or source_path.stem
            if self.orientation is None:
                self.orientation = getattr(loaded, "orientation", None) or self.metadata.get("orientation")
        else:
            self.metadata = dict(self.metadata or {})

        if self.orientation is None:
            self.orientation = self.metadata.get("orientation")
        if self.orientation is not None:
            self.metadata["orientation"] = self.orientation

        # Prefer explicit axes, else metadata axes.
        if self.axes is None:
            self.axes = self.metadata.get("axes")

        if self.axes is not None:
            if len(self.axes) != getattr(self.data, "ndim", -1):
                raise ValidationError(
                    f"Axes '{self.axes}' do not match data ndim={getattr(self.data, 'ndim', None)}"
                )
            self.metadata["axes"] = self.axes

        series_description = (
            self.metadata.get("submodality")
            or self.metadata.get("SeriesDescription")
            or self.metadata.get("(0008,103E)")
            or self.metadata.get("series_description")
        )
        if series_description not in (None, ""):
            self.metadata["submodality"] = series_description
            self.metadata.setdefault("series_description", series_description)

        self.metadata.setdefault("rescale_type", "DV")
        self.metadata["shape"] = tuple(getattr(self.data, "shape", ()))

    @property
    def backend(self) -> str:
        """``\"cupy\"`` or ``\"numpy\"`` depending on ``self.data``."""
        return "cupy" if is_cupy_array(self.data) else "numpy"

    @property
    def modality(self) -> str | None:
        """DICOM Modality (tag ``(0008,0060)``), e.g. ``\"PT\"``, ``\"CT\"``, ``\"MR\"``."""
        return self.metadata.get("Modality") or self.metadata.get("(0008,0060)")

    @property
    def submodality(self) -> str | None:
        """Series-level descriptor (series description / submodality), if recorded."""
        return (
            self.metadata.get("submodality")
            or self.metadata.get("SeriesDescription")
            or self.metadata.get("(0008,103E)")
            or self.metadata.get("series_description")
        )

    @property
    def rescale_type(self) -> str:
        """Intensity units tag (default ``\"DV\"``), upper-cased; e.g. ``\"BQML\"`` for PET."""
        value = self.metadata.get("rescale_type", "DV")
        return str(value).upper()

    @property
    def is_pet(self) -> bool:
        """True when :attr:`modality` is a PET code (``PT`` / ``PET``)."""
        mod = self.modality
        if mod is None:
            return False
        return str(mod).upper() in {"PT", "PET"}

    @property
    def is_3d(self) -> bool:
        """True for a 3-D volume (no time axis)."""
        return self.ndim == 3

    @property
    def is_4d(self) -> bool:
        """True for a 4-D volume (e.g. a time series or 4D-flow stack)."""
        return self.ndim == 4

    @property
    def shape(self) -> tuple[int, ...]:
        """Voxel-array shape as a plain tuple of ints."""
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        """Number of array dimensions."""
        return int(self.data.ndim)

    @property
    def dtype(self) -> Any:
        """Voxel dtype of the underlying array."""
        return self.data.dtype

    @property
    def affine(self) -> np.ndarray | None:
        """4x4 world affine from ``metadata['affine']`` (float), or ``None`` if unset."""
        value = self.metadata.get("affine")
        if value is None:
            return None
        return np.asarray(value, dtype=float)

    @affine.setter
    def affine(self, value: Any) -> None:
        """Store a 4x4 world affine; rejects any other shape."""
        arr = np.asarray(value, dtype=float)
        if arr.shape != (4, 4):
            raise ValidationError(f"Affine matrix must be shape (4, 4), got {arr.shape}")
        self.metadata["affine"] = arr

    @property
    def spacing(self) -> tuple[float, ...] | None:
        """Physical voxel size in mm per axis.

        Prefers an explicit ``metadata['spacing']``; otherwise assembles it from
        the per-axis ``x_res``/``y_res``/``z_res``/``t_res`` keys. ``None`` when no
        spacing information is available.
        """
        if "spacing" in self.metadata and self.metadata["spacing"] is not None:
            spacing = self.metadata["spacing"]
            try:
                return tuple(float(v) for v in spacing)
            except Exception:
                return None

        values: list[float] = []
        for key in ("x_res", "y_res", "z_res", "t_res"):
            if key in self.metadata and self.metadata[key] is not None:
                values.append(float(self.metadata[key]))
        return tuple(values) if values else None

    @spacing.setter
    def spacing(self, values: Any) -> None:
        """Set spacing and mirror the first four entries into ``x/y/z/t_res``."""
        if values is None:
            self.metadata.pop("spacing", None)
            return

        seq = tuple(float(v) for v in values)
        self.metadata["spacing"] = seq
        if len(seq) > 0:
            self.metadata["x_res"] = seq[0]
        if len(seq) > 1:
            self.metadata["y_res"] = seq[1]
        if len(seq) > 2:
            self.metadata["z_res"] = seq[2]
        if len(seq) > 3:
            self.metadata["t_res"] = seq[3]

    @property
    def temporal_resolution(self) -> float | None:
        """Time between frames of a 4-D series (``t_res``), or ``None``."""
        val = self.metadata.get("t_res", self.metadata.get("temporal_resolution"))
        if val is None:
            return None
        return float(val)

    @property
    def dicom_tags(self) -> dict[str, Any]:
        """Metadata entries that look like DICOM tags (excludes nvitk housekeeping keys)."""
        out: dict[str, Any] = {}
        for key, value in self.metadata.items():
            if not isinstance(key, str):
                continue
            if _is_dicom_tag_key(key):
                out[key] = value
        return out

    def __repr__(self) -> str:
        """Compact one-line summary (shape, dtype, backend, axes, modality, …)."""
        return (
            f"Image(shape={self.shape}, dtype={self.dtype}, backend={self.backend}, "
            f"axes={self.axes!r}, orientation={self.orientation!r}, name={self.name!r}, "
            f"modality={self.modality!r}, submodality={self.submodality!r}, rescale_type={self.rescale_type!r})"
        )

    def __len__(self) -> int:
        """Length of the leading axis (number of slices/frames)."""
        return len(self.data)

    def __iter__(self):
        """Iterate the leading axis, yielding an :class:`Image` per slice."""
        for i in range(len(self)):
            yield self[i]

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        """NumPy view for ``np.asarray(image)`` — always copies GPU data to host."""
        arr = to_numpy(self.data)
        if dtype is not None:
            arr = np.asarray(arr, dtype=dtype)
        return np.asarray(arr)

    @property
    def __cuda_array_interface__(self):
        """Expose the CUDA array interface so CuPy-backed images interop zero-copy."""
        if is_cupy_array(self.data):
            return self.data.__cuda_array_interface__
        raise AttributeError("__cuda_array_interface__ is only available for CuPy-backed images.")

    def _clone(self, data: Any, *, axes: str | None = None) -> "Image":
        """Wrap *data* in a new Image, carrying metadata/name/orientation forward."""
        md = dict(self.metadata or {})
        if axes is None:
            axes = self.axes
        if axes is not None:
            md["axes"] = axes
        md["shape"] = tuple(getattr(data, "shape", ()))
        ori = md.get("orientation", self.orientation)
        return Image(data=data, metadata=md, axes=axes, name=self.name, orientation=ori)

    def _axes_after_indexing(self, key: Any) -> str | None:
        """Recompute the ``axes`` label string after ``__getitem__``.

        Integer indices drop their axis; slices keep it. Returns ``None`` (i.e.
        forget axis labels) whenever ``newaxis``/``None`` makes the mapping
        ambiguous, so we never carry a wrong axis convention forward.
        """
        if self.axes is None:
            return None

        if not isinstance(key, tuple):
            key = (key,)

        # newaxis/None complicates axis naming. Keep metadata conservative.
        if any(k is None for k in key):
            return None

        expanded: list[Any] = []
        used_items = 0
        for item in key:
            if item is Ellipsis:
                remaining = self.ndim - (len(key) - 1)
                expanded.extend([slice(None)] * remaining)
                used_items += remaining
            else:
                expanded.append(item)
                used_items += 1

        if used_items < self.ndim:
            expanded.extend([slice(None)] * (self.ndim - used_items))

        out_axes: list[str] = []
        for axis_char, idx in zip(self.axes, expanded):
            if isinstance(idx, (int, np.integer)):
                continue
            out_axes.append(axis_char)

        return "".join(out_axes) if out_axes else None

    def _axes_after_take(self, new_data: Any, *, axis: int | None) -> str | None:
        """Infer ``axes`` string after :meth:`numpy.ndarray.take` when possible."""
        if self.axes is None:
            return None
        nd_old = self.ndim
        nd_new = int(getattr(new_data, "ndim", 0))
        if len(self.axes) != nd_old:
            return None
        if nd_new == nd_old:
            return self.axes
        if nd_new == nd_old - 1 and axis is not None:
            ax = axis if axis >= 0 else nd_old + axis
            if 0 <= ax < len(self.axes):
                return self.axes[:ax] + self.axes[ax + 1 :]
        return None

    def __getitem__(self, key: Any) -> Any:
        """Index the voxel array; array results stay :class:`Image`, scalars unwrap."""
        out = self.data[key]
        if _is_array_like(out):
            return self._clone(out, axes=self._axes_after_indexing(key))
        if hasattr(out, "item"):
            try:
                return out.item()
            except Exception:
                return out
        return out

    def __setitem__(self, key: Any, value: Any) -> None:
        """Assign into the voxel array, unwrapping an :class:`Image` right-hand side."""
        self.data[key] = value.data if isinstance(value, Image) else value

    def copy(self, deep_data: bool = True) -> "Image":
        """Duplicate voxel data when *deep_data* is True (default); shallow copy shares the array reference."""
        copied = self.data.copy() if deep_data and hasattr(self.data, "copy") else self.data
        return self._clone(copied)

    def with_data(self, data: Any, *, axes: str | None = None) -> "Image":
        """Return a new :class:`Image` with replaced voxel ``data`` (and merged metadata when ``data`` is an :class:`Image`)."""
        if isinstance(data, Image):
            md = dict(self.metadata or {})
            md.update(dict(data.metadata or {}))
            eff_axes = axes if axes is not None else data.axes
            md["shape"] = tuple(getattr(data.data, "shape", ()))
            if eff_axes is not None:
                md["axes"] = eff_axes
            else:
                md.pop("axes", None)
            out_name = self.name if self.name is not None else data.name
            return Image(
                data=data.data,
                metadata=md,
                axes=eff_axes,
                name=out_name,
                orientation=md.get("orientation"),
            )

        nd = int(getattr(data, "ndim", -1))
        eff_axes = axes
        if eff_axes is None and self.axes is not None and len(self.axes) != nd:
            md = dict(self.metadata or {})
            md.pop("axes", None)
            md["shape"] = tuple(getattr(data, "shape", ()))
            md.pop("orientation", None)
            return Image(data=data, metadata=md, axes=None, name=self.name, orientation=None)
        return self._clone(data, axes=eff_axes)

    def take(self, indices: Any, axis: int | None = None, **kwargs: Any) -> "Image":
        """Apply :meth:`numpy.ndarray.take` to ``data`` and wrap the result as an :class:`Image`."""
        new_data = self.data.take(indices, axis=axis, **kwargs)
        new_axes = self._axes_after_take(new_data, axis=axis)
        if new_axes is None and self.axes is not None:
            md = dict(self.metadata or {})
            md.pop("axes", None)
            md["shape"] = tuple(getattr(new_data, "shape", ()))
            md.pop("orientation", None)
            return Image(data=new_data, metadata=md, axes=None, name=self.name, orientation=None)
        return self._clone(new_data, axes=new_axes)

    def astype(self, dtype: Any, copy: bool = True) -> "Image":
        """Return a new :class:`Image` whose array is ``self.data.astype(...)``."""
        return self._clone(self.data.astype(dtype, copy=copy))

    def to_numpy(self, copy: bool = False) -> "Image":
        """Host-memory :class:`Image` (delegates to :func:`nvitk.core.array.to_numpy`)."""
        return self._clone(to_numpy(self.data, copy=copy))

    def to_cupy(self, copy: bool = False, strict: bool = False) -> "Image":
        """CuPy-backed :class:`Image` when available."""
        return self._clone(to_cupy(self.data, copy=copy, strict=strict))

    def to_backend(self, backend: str, copy: bool = False, strict: bool = False) -> "Image":
        """Coerce voxels with :func:`nvitk.core.array.as_backend_array`."""
        return self._clone(as_backend_array(self.data, backend=backend, copy=copy, strict=strict))

    def orient_to(self, target: str) -> "Image":
        """
        Return a new image reoriented to *target* axis codes (e.g. ``\"RAS\"``).

        Uses nibabel orientation transforms on the first 3 spatial axes and updates
        affine/orientation metadata accordingly.
        """
        target_codes = str(target).strip().upper()
        if len(target_codes) != 3:
            raise ValidationError(f"target orientation must have length 3, got {target!r}")
        if any(c not in {"L", "R", "A", "P", "S", "I"} for c in target_codes):
            raise ValidationError(f"Invalid orientation codes: {target!r}")
        affine = self.affine
        if affine is None:
            raise ValidationError("orient_to requires metadata['affine'] to be set.")
        if self.ndim < 3:
            raise ValidationError("orient_to requires at least 3 dimensions.")

        try:
            import nibabel as nib
        except Exception as exc:
            raise ValidationError(
                "nibabel is required for Image.orient_to. Install nibabel to use this method."
            ) from exc

        data_np = to_numpy(self.data, copy=False)
        old_ornt = nib.orientations.io_orientation(affine)
        target_ornt = nib.orientations.axcodes2ornt(tuple(target_codes))
        xfm = nib.orientations.ornt_transform(old_ornt, target_ornt)

        if np.allclose(xfm, np.array([[0, 1], [1, 1], [2, 1]], dtype=float)):
            out = self.copy(deep_data=True)
            out.orientation = target_codes
            out.metadata["orientation"] = target_codes
            return out

        reoriented_np = nib.orientations.apply_orientation(data_np, xfm)
        new_affine = affine @ nib.orientations.inv_ornt_aff(xfm, data_np.shape)

        if is_cupy_array(self.data):
            new_data = to_cupy(reoriented_np, copy=False)
        else:
            new_data = reoriented_np

        md = dict(self.metadata or {})
        md["affine"] = np.asarray(new_affine, dtype=float)
        md["orientation"] = target_codes
        md["shape"] = tuple(getattr(new_data, "shape", ()))

        # Keep spacing consistent with the reoriented affine.
        for i, key in enumerate(("x_res", "y_res", "z_res")):
            md[key] = float(np.linalg.norm(md["affine"][:3, i]))
        spacing_existing = self.spacing
        spacing_xyz = (md["x_res"], md["y_res"], md["z_res"])
        if spacing_existing is not None and len(spacing_existing) > 3:
            md["spacing"] = spacing_xyz + tuple(spacing_existing[3:])
        else:
            md["spacing"] = spacing_xyz

        new_axes: str | None = self.axes
        if self.axes is not None and len(self.axes) >= 3:
            perm = [int(i) for i in xfm[:, 0]]
            spatial = [self.axes[p] for p in perm]
            tail = self.axes[3:]
            candidate = "".join(spatial) + tail
            if len(candidate) == int(getattr(new_data, "ndim", 0)):
                new_axes = candidate
            else:
                new_axes = None
                md.pop("axes", None)

        return Image(data=new_data, metadata=md, axes=new_axes, name=self.name, orientation=target_codes)

    def as_tuple(self) -> tuple[Any, dict[str, Any]]:
        """``(data, metadata)`` for interop with APIs expecting raw arrays."""
        return self.data, dict(self.metadata or {})

    def get_meta(self, key: str, default: Any = None, *aliases: str) -> Any:
        """First matching key among *key* and *aliases* in ``metadata``, else *default*."""
        if key in self.metadata:
            return self.metadata[key]
        for alias in aliases:
            if alias in self.metadata:
                return self.metadata[alias]
        return default

    def set_meta(self, key: str, value: Any) -> None:
        """Assign ``metadata[key] = value``."""
        self.metadata[key] = value

    def has_meta(self, key: str, *aliases: str) -> bool:
        """True if *key* or any *aliases* is present in ``metadata``."""
        if key in self.metadata:
            return True
        return any(alias in self.metadata for alias in aliases)

    def select_meta(self, keys: list[str] | tuple[str, ...]) -> dict[str, Any]:
        """Subset of ``metadata`` containing only keys that exist."""
        return {k: self.metadata[k] for k in keys if k in self.metadata}

    # ──────────────────────────────────────────────────────────────────────────────
    # Arithmetic
    # ──────────────────────────────────────────────────────────────────────────────

    # Every operator below routes through ``_binary`` / ``_clone`` so the result
    # keeps this image's metadata and axis labels (another Image is unwrapped to
    # its array first). Array results stay Image; reduced scalars pass through.

    def _binary(self, other: Any, op) -> Any:
        """Apply elementwise *op* against *other* (Image or array), re-wrapping arrays."""
        rhs = other.data if isinstance(other, Image) else other
        out = op(self.data, rhs)
        return self._clone(out) if _is_array_like(out) else out

    def __add__(self, other: Any) -> Any:
        """Elementwise ``self + other``."""
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other: Any) -> Any:
        """Elementwise ``other + self`` (reflected)."""
        return self._binary(other, lambda a, b: b + a)

    def __sub__(self, other: Any) -> Any:
        """Elementwise ``self - other``."""
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other: Any) -> Any:
        """Elementwise ``other - self`` (reflected)."""
        return self._binary(other, lambda a, b: b - a)

    def __mul__(self, other: Any) -> Any:
        """Elementwise ``self * other``."""
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other: Any) -> Any:
        """Elementwise ``other * self`` (reflected)."""
        return self._binary(other, lambda a, b: b * a)

    def __truediv__(self, other: Any) -> Any:
        """Elementwise ``self / other``."""
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other: Any) -> Any:
        """Elementwise ``other / self`` (reflected)."""
        return self._binary(other, lambda a, b: b / a)

    def __pow__(self, other: Any) -> Any:
        """Elementwise ``self ** other``."""
        return self._binary(other, lambda a, b: a**b)

    def __neg__(self) -> "Image":
        """Elementwise negation."""
        return self._clone(-self.data)

    def __abs__(self) -> "Image":
        """Elementwise absolute value."""
        return self._clone(abs(self.data))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """NumPy/CuPy ufunc hook so ``np.*`` calls on an Image return an Image.

        Image inputs are unwrapped to their arrays before dispatch; array outputs
        (including each element of a multi-output ufunc) are re-wrapped.
        """
        prepared = [x.data if isinstance(x, Image) else x for x in inputs]
        result = getattr(ufunc, method)(*prepared, **kwargs)
        if _is_array_like(result):
            return self._clone(result)
        if isinstance(result, tuple):
            return tuple(self._clone(x) if _is_array_like(x) else x for x in result)
        return result

    # ──────────────────────────────────────────────────────────────────────────────
    # I/O
    # ──────────────────────────────────────────────────────────────────────────────

    def save(self, path: str, **kwargs: Any) -> None:
        """Write via :func:`nvitk.io.imageio.imsave` (format from extension or ``force_type``)."""
        from nvitk.io.imageio import imsave

        imsave(path, self, **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs: Any):
        """Shorthand for :func:`nvitk.io.imageio.imread` returning an :class:`Image`."""
        from nvitk.io.imageio import imread

        return imread(path, **kwargs)
