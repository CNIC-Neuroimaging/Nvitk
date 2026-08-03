"""3D Slicer ScriptedLoadableModule: Mouse TOF Circle-of-Willis lab workflow.

Mirrors the Napari Lab tool recipe without importing nvitk:
  Stage 1: Slicer N4ITKBiasFieldCorrection → blood flood from-scratch → label CCs
  Stage 2: click CCs → assign to Left ICA / Right ICA / Basilar → multilabel expand

Optional pip deps (Slicer already has numpy / scipy + built-in N4ITK CLI):
  scikit-image, scikit-learn
"""

from __future__ import annotations

import logging
import os
from typing import Any

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

import sys

_module_dir = os.path.dirname(os.path.abspath(__file__))
if _module_dir not in sys.path:
    sys.path.insert(0, _module_dir)

from MouseTOFCoWLib import TREE_SPECS, ensure_optional_deps, expand_cow_trees, run_stage1

#
# MouseTOFCoW
#


class MouseTOFCoW(ScriptedLoadableModule):
    """Module metadata for Additional Module Paths discovery."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Mouse TOF CoW"
        self.parent.categories = ["nvitk"]
        self.parent.dependencies = []
        self.parent.contributors = ["nvitk"]
        self.parent.helpText = (
            "Stage 1: Slicer N4ITKBiasFieldCorrection → blood flood → CC labels on a 3D TOF. "
            "Stage 2: click CCs and assign them to Left ICA / Right ICA / Basilar, "
            "then finalize with a multilabel blood-flood expand. "
            "Self-contained module (no full nvitk / ANTsPy). "
            "Requires scikit-image and scikit-learn in Slicer's Python."
        )
        self.parent.acknowledgementText = "nvitk Mouse TOF CoW lab workflow (Slicer port)."
        iconPath = os.path.join(
            os.path.dirname(__file__),
            "Resources",
            "Icons",
            "MouseTOFCoW.png",
        )
        if os.path.isfile(iconPath):
            self.parent.icon = qt.QIcon(iconPath)


#
# Colors (match Napari lab tints)
#

_TREE_COLORS_RGB = {
    1: (51, 153, 255),  # Left ICA — blue
    2: (255, 89, 89),  # Right ICA — red
    3: (89, 230, 102),  # Basilar — green
}
_HIGHLIGHT_RGB = (255, 255, 80)


#
# Stage-2 session
#


class MouseTofCowSlicerSession:
    """Interactive Stage-2 assignment of CCs to CoW vessel trees."""

    def __init__(
        self,
        *,
        source_name: str,
        intensity: np.ndarray,
        cc_source: np.ndarray,
        cc_volume_node,
        status_callback=None,
        on_finished=None,
    ):
        self.TREE_SPECS = TREE_SPECS
        self.source_name = source_name
        self.intensity = np.asarray(intensity)
        self.cc_source = np.asarray(cc_source, dtype=np.int32)
        self.cc_volume_node = cc_volume_node
        self.status_callback = status_callback
        self.on_finished = on_finished

        self.tree_index = 0
        self.highlight_id: int | None = None
        self.assigned: dict[int, int] = {}
        self._finished = False
        self._pick_observer_tags: list[tuple[Any, int]] = []
        self._color_node = None
        self._vr_display_node = None
        self._three_d_views: dict[Any, Any] = {}  # interactor -> threeDView
        self.cut_mode: bool = False
        self.cut_radius_mm: float = 0.5

    @property
    def tree_name(self) -> str:
        return self.TREE_SPECS[self.tree_index][0]

    @property
    def tree_label(self) -> int:
        return self.TREE_SPECS[self.tree_index][1]

    def status_text(self) -> str:
        if self._finished:
            return "Mouse TOF CoW: done."
        hl = f"CC {self.highlight_id}" if self.highlight_id else "none"
        n_tree = sum(1 for v in self.assigned.values() if v == self.tree_label)
        return (
            f"Selecting: {self.tree_name} (label {self.tree_label}) — "
            f"highlight={hl} — in tree: {n_tree} CC(s) — "
            f"assigned total: {len(self.assigned)}"
        )

    def _push_status(self) -> None:
        if self.status_callback is not None:
            try:
                self.status_callback(self.status_text())
            except Exception:
                pass

    def _rgba_for_label(self, lid: int) -> tuple[float, float, float, float]:
        """Return (r,g,b,a) for a CC id (opaque 3D-friendly alphas)."""
        if lid in self.assigned:
            r, g, b = _TREE_COLORS_RGB[int(self.assigned[lid])]
            return (r / 255.0, g / 255.0, b / 255.0, 1.0)
        if self.highlight_id is not None and lid == int(self.highlight_id):
            r, g, b = _HIGHLIGHT_RGB
            return (r / 255.0, g / 255.0, b / 255.0, 1.0)
        h = (lid * 37) % 180
        return (
            0.55 + 0.35 * ((h % 60) / 60.0),
            0.55 + 0.35 * (((h // 60) % 3) / 3.0),
            0.35,
            1.0,
        )

    def install_pick_observers(self) -> None:
        self.uninstall_pick_observers()
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        # 2D slice views
        for name in layoutManager.sliceViewNames():
            sliceWidget = layoutManager.sliceWidget(name)
            if sliceWidget is None:
                continue
            interactor = sliceWidget.sliceView().interactorStyle().GetInteractor()
            if interactor is None:
                continue
            tag = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent,
                self._on_slice_click,
                1.0,
            )
            self._pick_observer_tags.append((interactor, tag))
        # 3D views
        try:
            n3d = layoutManager.threeDViewCount
        except Exception:
            n3d = 1
        for i in range(int(n3d)):
            try:
                threeDWidget = layoutManager.threeDWidget(i)
            except Exception:
                continue
            if threeDWidget is None:
                continue
            threeDView = threeDWidget.threeDView()
            interactor = threeDView.interactor()
            if interactor is None:
                continue
            tag = interactor.AddObserver(
                vtk.vtkCommand.LeftButtonPressEvent,
                self._on_three_d_click,
                1.0,
            )
            self._pick_observer_tags.append((interactor, tag))
            self._three_d_views[interactor] = threeDView

        # Fill colors first, then create VR (avoids GetColor on empty table).
        self.refresh_display()
        self._push_status()

    def uninstall_pick_observers(self) -> None:
        for interactor, tag in self._pick_observer_tags:
            try:
                interactor.RemoveObserver(tag)
            except Exception:
                pass
        self._pick_observer_tags.clear()
        self._three_d_views.clear()

    def _on_slice_click(self, caller, event) -> None:
        if self._finished:
            return
        interactor = caller
        try:
            x, y = interactor.GetEventPosition()
        except Exception:
            return

        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return
        ras = None
        for name in layoutManager.sliceViewNames():
            sliceWidget = layoutManager.sliceWidget(name)
            if sliceWidget is None:
                continue
            view_interactor = sliceWidget.sliceView().interactorStyle().GetInteractor()
            if view_interactor is not interactor:
                continue
            sliceNode = sliceWidget.sliceLogic().GetSliceNode()
            point = sliceNode.GetXYToRAS().MultiplyPoint(
                [float(x), float(y), 0.0, 1.0]
            )
            ras = [point[0], point[1], point[2]]
            break
        if ras is None:
            return
        self._handle_pick_ras(ras)

    def _on_three_d_click(self, caller, event) -> None:
        """Pick first non-zero CC along the camera ray through the click."""
        if self._finished:
            return
        interactor = caller
        try:
            x, y = interactor.GetEventPosition()
        except Exception:
            return

        threeDView = self._three_d_views.get(interactor)
        if threeDView is None:
            layoutManager = slicer.app.layoutManager()
            if layoutManager is None:
                return
            try:
                for i in range(int(layoutManager.threeDViewCount)):
                    w = layoutManager.threeDWidget(i)
                    if w is not None and w.threeDView().interactor() is interactor:
                        threeDView = w.threeDView()
                        break
            except Exception:
                return
        if threeDView is None:
            return

        ras = self._raycast_cc_ras(threeDView, float(x), float(y))
        if ras is None:
            return
        self._handle_pick_ras(ras)

    def _raycast_cc_ras(self, threeDView, x: float, y: float):
        """March from near→far clip through the labelmap; return RAS of first CC hit."""
        renderer = threeDView.renderWindow().GetRenderers().GetFirstRenderer()
        if renderer is None or self.cc_volume_node is None:
            return None

        near = [0.0, 0.0, 0.0, 1.0]
        far = [0.0, 0.0, 0.0, 1.0]
        vtk.vtkInteractorObserver.ComputeDisplayToWorld(renderer, x, y, 0.0, near)
        vtk.vtkInteractorObserver.ComputeDisplayToWorld(renderer, x, y, 1.0, far)
        p0 = np.array(near[:3], dtype=np.float64)
        p1 = np.array(far[:3], dtype=np.float64)
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            return None
        direction /= length

        spacing = self.cc_volume_node.GetSpacing()
        step = 0.5 * float(min(abs(spacing[0]), abs(spacing[1]), abs(spacing[2]), 1.0))
        step = max(step, 1e-3)
        n_steps = int(length / step) + 1

        for i in range(n_steps):
            ras = (p0 + direction * (i * step)).tolist()
            lid = self._sample_cc_at_ras(ras)
            if lid is not None and lid > 0:
                return ras
        return None

    def _handle_pick_ras(self, ras) -> None:
        if self.cut_mode:
            # In cut mode, clicks carve the highlighted CC (must already be selected).
            self.cut_highlighted_cc_at_ras(ras, radius_mm=self.cut_radius_mm)
            return
        lid = self._sample_cc_at_ras(ras)
        if lid is None or lid <= 0:
            return
        if lid in self.assigned:
            slicer.util.warningDisplay(
                f"CC {lid} already assigned to tree label {self.assigned[lid]}.",
                windowTitle="Mouse TOF CoW",
            )
            return
        self.highlight_id = int(lid)
        self.refresh_display()
        self._push_status()
        logging.info("Mouse TOF CoW: highlighted CC %s", lid)

    def _sample_cc_at_ras(self, ras) -> int | None:
        volumeNode = self.cc_volume_node
        if volumeNode is None:
            return None
        rasToIJK = vtk.vtkMatrix4x4()
        volumeNode.GetRASToIJKMatrix(rasToIJK)
        point_ijk = rasToIJK.MultiplyPoint([ras[0], ras[1], ras[2], 1.0])
        ijk = [int(round(point_ijk[0])), int(round(point_ijk[1])), int(round(point_ijk[2]))]

        arr = self.cc_source
        k, j, i = ijk[2], ijk[1], ijk[0]
        if not (0 <= k < arr.shape[0] and 0 <= j < arr.shape[1] and 0 <= i < arr.shape[2]):
            return None
        return int(arr[k, j, i])

    def clear_highlight(self) -> None:
        self.highlight_id = None
        self.refresh_display()
        self._push_status()

    def add_highlighted_cc(self) -> None:
        if self._finished:
            return
        if self.highlight_id is None:
            slicer.util.warningDisplay(
                "Click a connected component on the CC labelmap (slice or 3D) first.",
                windowTitle="Mouse TOF CoW",
            )
            return
        lid = int(self.highlight_id)
        if lid in self.assigned:
            slicer.util.warningDisplay(
                f"CC {lid} is already assigned.",
                windowTitle="Mouse TOF CoW",
            )
            return
        self.assigned[lid] = int(self.tree_label)
        self.highlight_id = None
        self.refresh_display()
        self._push_status()
        logging.info("Mouse TOF CoW: added CC %s to %s", lid, self.tree_name)

    def split_highlighted_cc_by_active_slice(self) -> None:
        """Bipartition the highlighted CC with the active slice plane (for L/R bridges)."""
        if self._finished:
            return
        if self.highlight_id is None:
            slicer.util.warningDisplay(
                "Highlight a CC first, orient a slice through the bridge, then Split.",
                windowTitle="Mouse TOF CoW",
            )
            return
        lid = int(self.highlight_id)
        if lid in self.assigned:
            slicer.util.warningDisplay(
                "Unassign / only split free CCs (not yet added to a tree).",
                windowTitle="Mouse TOF CoW",
            )
            return

        sliceNode = self._active_slice_node()
        if sliceNode is None:
            slicer.util.warningDisplay(
                "No slice view available to define the cutting plane.",
                windowTitle="Mouse TOF CoW",
            )
            return

        # Plane: point + normal in RAS from the slice-to-RAS matrix.
        # Slice XY plane maps to columns 0,1 of XYToRAS; normal ≈ column 2.
        xyToRAS = sliceNode.GetXYToRAS()
        origin = np.array(
            [xyToRAS.GetElement(0, 3), xyToRAS.GetElement(1, 3), xyToRAS.GetElement(2, 3)],
            dtype=np.float64,
        )
        normal = np.array(
            [xyToRAS.GetElement(0, 2), xyToRAS.GetElement(1, 2), xyToRAS.GetElement(2, 2)],
            dtype=np.float64,
        )
        nrm = float(np.linalg.norm(normal))
        if nrm < 1e-9:
            slicer.util.warningDisplay(
                "Could not read a valid slice-plane normal.",
                windowTitle="Mouse TOF CoW",
            )
            return
        normal /= nrm

        mask = self.cc_source == lid
        if not np.any(mask):
            return

        # Compute side of plane for each voxel of this CC (vectorized via IJK grid).
        kk, jj, ii = np.where(mask)
        ijkToRAS = vtk.vtkMatrix4x4()
        self.cc_volume_node.GetIJKToRASMatrix(ijkToRAS)
        # RAS = M * (i,j,k,1); Slicer array is (k,j,i)
        ones = np.ones(ii.shape[0], dtype=np.float64)
        # Homogeneous multiply
        i_f = ii.astype(np.float64)
        j_f = jj.astype(np.float64)
        k_f = kk.astype(np.float64)
        ras_x = (
            ijkToRAS.GetElement(0, 0) * i_f
            + ijkToRAS.GetElement(0, 1) * j_f
            + ijkToRAS.GetElement(0, 2) * k_f
            + ijkToRAS.GetElement(0, 3)
        )
        ras_y = (
            ijkToRAS.GetElement(1, 0) * i_f
            + ijkToRAS.GetElement(1, 1) * j_f
            + ijkToRAS.GetElement(1, 2) * k_f
            + ijkToRAS.GetElement(1, 3)
        )
        ras_z = (
            ijkToRAS.GetElement(2, 0) * i_f
            + ijkToRAS.GetElement(2, 1) * j_f
            + ijkToRAS.GetElement(2, 2) * k_f
            + ijkToRAS.GetElement(2, 3)
        )
        side = (
            (ras_x - origin[0]) * normal[0]
            + (ras_y - origin[1]) * normal[1]
            + (ras_z - origin[2]) * normal[2]
        )
        side_a = side <= 0.0
        side_b = ~side_a
        if not np.any(side_a) or not np.any(side_b):
            slicer.util.warningDisplay(
                "Slice plane does not cut through the highlighted CC. "
                "Move the slice so it intersects the bridging region.",
                windowTitle="Mouse TOF CoW",
            )
            return

        new_a = int(self.cc_source.max()) + 1
        new_b = new_a + 1
        self.cc_source[mask] = 0
        self.cc_source[kk[side_a], jj[side_a], ii[side_a]] = new_a
        self.cc_source[kk[side_b], jj[side_b], ii[side_b]] = new_b
        # Re-label each side in case the plane created multiple fragments.
        self._relabel_mask_components(self.cc_source == new_a)
        self._relabel_mask_components(self.cc_source == new_b)

        self._push_cc_to_volume()
        self.highlight_id = None
        self.refresh_display()
        self._push_status()
        logging.info(
            "Mouse TOF CoW: split CC %s by slice → new labels (see max=%s)",
            lid,
            int(self.cc_source.max()),
        )
        slicer.util.infoDisplay(
            f"Split CC {lid} by the active slice into separate components.\n"
            f"Click each fragment to assign to Left/Right ICA (or Basilar).",
            windowTitle="Mouse TOF CoW",
        )

    def _relabel_mask_components(self, mask: np.ndarray) -> list[int]:
        """Replace True voxels in mask with fresh CC ids (connected components)."""
        from scipy import ndimage as ndi

        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labeled, n = ndi.label(mask.astype(bool), structure=structure)
        if n == 0:
            return []
        new_ids = []
        base = int(self.cc_source.max())
        self.cc_source[mask] = 0
        for comp in range(1, n + 1):
            new_id = base + comp
            self.cc_source[labeled == comp] = new_id
            new_ids.append(new_id)
        return new_ids

    def cut_highlighted_cc_at_ras(self, ras, *, radius_mm: float = 0.5) -> None:
        """Erase a sphere from the highlighted CC and re-label remaining fragments."""
        if self._finished:
            return
        if self.highlight_id is None:
            slicer.util.warningDisplay(
                "Highlight a CC first, then click the bridge to cut.",
                windowTitle="Mouse TOF CoW",
            )
            return
        lid = int(self.highlight_id)
        if lid in self.assigned:
            slicer.util.warningDisplay(
                "Only free (unassigned) CCs can be cut.",
                windowTitle="Mouse TOF CoW",
            )
            return

        spacing = self.cc_volume_node.GetSpacing()
        rasToIJK = vtk.vtkMatrix4x4()
        self.cc_volume_node.GetRASToIJKMatrix(rasToIJK)
        c_ijk = rasToIJK.MultiplyPoint([ras[0], ras[1], ras[2], 1.0])
        ci, cj, ck = float(c_ijk[0]), float(c_ijk[1]), float(c_ijk[2])
        ri = max(radius_mm / max(abs(spacing[0]), 1e-6), 1.0)
        rj = max(radius_mm / max(abs(spacing[1]), 1e-6), 1.0)
        rk = max(radius_mm / max(abs(spacing[2]), 1e-6), 1.0)

        shape = self.cc_source.shape
        i0, i1 = int(max(0, np.floor(ci - ri))), int(min(shape[2] - 1, np.ceil(ci + ri)))
        j0, j1 = int(max(0, np.floor(cj - rj))), int(min(shape[1] - 1, np.ceil(cj + rj)))
        k0, k1 = int(max(0, np.floor(ck - rk))), int(min(shape[0] - 1, np.ceil(ck + rk)))

        before = self.cc_source == lid
        if not np.any(before):
            return

        # Build spherical erase mask in the local window.
        kk, jj, ii = np.ogrid[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]
        sphere = ((ii - ci) / ri) ** 2 + ((jj - cj) / rj) ** 2 + ((kk - ck) / rk) ** 2 <= 1.0
        window = self.cc_source[k0 : k1 + 1, j0 : j1 + 1, i0 : i1 + 1]
        erase = sphere & (window == lid)
        if not np.any(erase):
            return
        window[erase] = 0

        remaining = before & (self.cc_source == lid)
        self.cc_source[remaining] = 0
        new_ids = self._relabel_mask_components(remaining)
        self._push_cc_to_volume()
        self.highlight_id = new_ids[0] if len(new_ids) == 1 else None
        self.refresh_display()
        self._push_status()
        logging.info(
            "Mouse TOF CoW: cut CC %s → %s fragment(s) %s",
            lid,
            len(new_ids),
            new_ids,
        )

    def _push_cc_to_volume(self) -> None:
        if self.cc_volume_node is None:
            return
        max_id = int(self.cc_source.max()) if self.cc_source.size else 0
        if max_id > np.iinfo(np.int16).max:
            raise ValueError(f"Label id {max_id} exceeds int16 labelmap range.")
        slicer.util.updateVolumeFromArray(
            self.cc_volume_node, self.cc_source.astype(np.int16, copy=False)
        )

    def _active_slice_node(self):
        layoutManager = slicer.app.layoutManager()
        if layoutManager is None:
            return None
        # Prefer the slice view that currently has focus; else Red.
        try:
            for name in ("Red", "Yellow", "Green"):
                w = layoutManager.sliceWidget(name)
                if w is not None and w.sliceView().hasFocus():
                    return w.sliceLogic().GetSliceNode()
        except Exception:
            pass
        for name in layoutManager.sliceViewNames():
            w = layoutManager.sliceWidget(name)
            if w is not None:
                return w.sliceLogic().GetSliceNode()
        return None

    def cleanup_intermediates(self, trees_node=None) -> None:
        """Remove Stage-1 CC labelmap, its VR, and color node; keep final trees."""
        # Tear down VR on the CC volume
        if self.cc_volume_node is not None:
            try:
                vrLogic = slicer.modules.volumerendering.logic()
                vr = vrLogic.GetFirstVolumeRenderingDisplayNode(self.cc_volume_node)
                if vr is not None:
                    vr.SetVisibility(False)
                    slicer.mrmlScene.RemoveNode(vr)
            except Exception:
                pass
            try:
                slicer.mrmlScene.RemoveNode(self.cc_volume_node)
            except Exception:
                pass
            self.cc_volume_node = None
        self._vr_display_node = None

        if self._color_node is not None:
            try:
                slicer.mrmlScene.RemoveNode(self._color_node)
            except Exception:
                pass
            self._color_node = None

        # Also remove any leftover N4 temp nodes from failed runs
        try:
            for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
                name = node.GetName() or ""
                if name.endswith("_tof_cow_n4_tmp") or name.endswith("_tof_cow_cc_colors"):
                    slicer.mrmlScene.RemoveNode(node)
        except Exception:
            pass

        if trees_node is not None:
            try:
                slicer.util.setSliceViewerLayers(label=trees_node, fit=False)
            except Exception:
                pass

    def finish_current_tree(self) -> None:
        if self._finished:
            return
        n_tree = sum(1 for v in self.assigned.values() if v == self.tree_label)
        logging.info("Mouse TOF CoW: %s done (%s CC(s))", self.tree_name, n_tree)
        if self.tree_index + 1 < len(self.TREE_SPECS):
            self.tree_index += 1
            self.highlight_id = None
            self.refresh_display()
            self._push_status()
            return
        self.finalize()

    def finalize(self) -> None:
        if self._finished:
            return
        seeds = np.zeros(self.cc_source.shape, dtype=np.int32)
        for cc_id, tree_lab in self.assigned.items():
            seeds[self.cc_source == int(cc_id)] = int(tree_lab)

        logging.info("Mouse TOF CoW: final multilabel blood-flood expand…")
        slicer.app.processEvents()
        out = expand_cow_trees(self.intensity, seeds)

        # Keep a reference before cleanup removes the CC node geometry source.
        ref_node = self.cc_volume_node
        logic = MouseTOFCoWLogic()
        trees_node = logic.write_labelmap(
            out,
            reference_volume=ref_node,
            name=f"{self.source_name}_tof_cow_trees",
        )
        logic.apply_tree_colors(trees_node)

        self._finished = True
        self.uninstall_pick_observers()
        self.cleanup_intermediates(trees_node)
        self._push_status()
        if self.on_finished is not None:
            try:
                self.on_finished(trees_node)
            except Exception:
                pass
        slicer.util.infoDisplay(
            f"Mouse TOF CoW complete: {trees_node.GetName()}\n"
            f"(1=Left ICA, 2=Right ICA, 3=Basilar; {len(self.assigned)} CC(s)).\n"
            f"Intermediate CC labelmap / 3D overlay removed.",
            windowTitle="Mouse TOF CoW",
        )

    def _ensure_volume_rendering(self) -> None:
        """Show the CC labelmap in the 3D view via volume rendering."""
        volumeNode = self.cc_volume_node
        if volumeNode is None:
            return
        logic = MouseTOFCoWLogic()
        self._vr_display_node = logic.show_labelmap_in_3d(volumeNode)

    def refresh_display(self) -> None:
        volumeNode = self.cc_volume_node
        if volumeNode is None:
            return
        displayNode = volumeNode.GetDisplayNode()
        if displayNode is None:
            volumeNode.CreateDefaultDisplayNodes()
            displayNode = volumeNode.GetDisplayNode()
        if displayNode is None:
            return

        colorNode = self._ensure_color_node()
        max_label = int(self.cc_source.max()) if self.cc_source.size else 0
        n_colors = max(max_label + 1, 4)
        MouseTOFCoWLogic.resize_color_table(colorNode, n_colors)

        rgba_by_label: dict[int, tuple[float, float, float, float]] = {}
        colorNode.SetColor(0, 0.0, 0.0, 0.0, 0.0)
        for lid in range(1, n_colors):
            rgba = self._rgba_for_label(lid)
            rgba_by_label[lid] = rgba
            r, g, b, a = rgba
            colorNode.SetColor(lid, r, g, b, a)

        displayNode.SetAndObserveColorNodeID(colorNode.GetID())
        displayNode.SetVisibility(True)

        if self._vr_display_node is None:
            self._ensure_volume_rendering()
        MouseTOFCoWLogic.apply_label_transfer_functions(
            self._vr_display_node, rgba_by_label, max_label=max_label
        )
        try:
            layoutManager = slicer.app.layoutManager()
            for i in range(int(layoutManager.threeDViewCount)):
                layoutManager.threeDWidget(i).threeDView().scheduleRender()
        except Exception:
            pass

    def _ensure_color_node(self):
        if self._color_node is not None:
            return self._color_node
        colorNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLColorTableNode",
            f"{self.source_name}_tof_cow_cc_colors",
        )
        colorNode.SetTypeToUser()
        colorNode.SetHideFromEditors(True)
        self._color_node = colorNode
        return colorNode

    def cancel(self) -> None:
        self.uninstall_pick_observers()
        self._finished = True
        self._push_status()


#
# Logic
#


class MouseTOFCoWLogic(ScriptedLoadableModuleLogic):
    """MRML ↔ numpy and local Stage-1 / expand (no nvitk)."""

    def ensure_deps(self) -> None:
        ensure_optional_deps()

    def array_from_volume(self, volumeNode) -> np.ndarray:
        return np.asarray(slicer.util.arrayFromVolume(volumeNode))

    @staticmethod
    def resize_color_table(color_node, n_colors: int) -> None:
        """Ensure a User color table has at least n_colors entries."""
        n = max(int(n_colors), 2)
        color_node.SetTypeToUser()
        color_node.SetNumberOfColors(n)
        lut = color_node.GetLookupTable()
        if lut is not None:
            try:
                lut.SetNumberOfTableValues(n)
                lut.SetTableRange(0, max(n - 1, 1))
                lut.Build()
            except Exception:
                pass

    @staticmethod
    def show_labelmap_in_3d(volumeNode, rgba_by_label=None):
        """Create/show volume rendering for a labelmap in the 3D view."""
        if volumeNode is None:
            return None
        try:
            vrLogic = slicer.modules.volumerendering.logic()
        except Exception as exc:
            logging.warning("Volume rendering module unavailable: %s", exc)
            return None

        displayNode = vrLogic.GetFirstVolumeRenderingDisplayNode(volumeNode)
        if displayNode is None:
            displayNode = vrLogic.CreateDefaultVolumeRenderingNodes(volumeNode)
        if displayNode is None:
            return None

        displayNode.SetVisibility(True)
        try:
            displayNode.SetCroppingEnabled(False)
        except Exception:
            pass

        max_label = 0
        try:
            arr = slicer.util.arrayFromVolume(volumeNode)
            max_label = int(np.max(arr)) if arr is not None and arr.size else 0
        except Exception:
            pass

        if rgba_by_label is not None:
            MouseTOFCoWLogic.apply_label_transfer_functions(
                displayNode, rgba_by_label, max_label=max_label
            )

        try:
            layoutManager = slicer.app.layoutManager()
            if layoutManager is not None and layoutManager.threeDViewCount > 0:
                layoutManager.threeDWidget(0).threeDView().resetFocalPoint()
        except Exception:
            pass
        return displayNode

    @staticmethod
    def apply_label_transfer_functions(
        vr_display_node,
        rgba_by_label: dict,
        *,
        max_label: int,
    ) -> None:
        """Nearest-neighbor RGB/opacity TFs so discrete labels render clearly in 3D."""
        if vr_display_node is None:
            return
        vp_node = vr_display_node.GetVolumePropertyNode()
        if vp_node is None:
            return
        volume_property = vp_node.GetVolumeProperty()
        if volume_property is None:
            return

        color_tf = vtk.vtkColorTransferFunction()
        opacity_tf = vtk.vtkPiecewiseFunction()
        color_tf.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
        opacity_tf.AddPoint(0.0, 0.0)

        n = max(int(max_label), 0)
        for lid in range(1, n + 1):
            rgba = rgba_by_label.get(lid) if rgba_by_label else None
            if rgba is None:
                h = (lid * 37) % 180
                r = 0.55 + 0.35 * ((h % 60) / 60.0)
                g = 0.55 + 0.35 * (((h // 60) % 3) / 3.0)
                b = 0.35
            else:
                r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
            # High opacity so thin vessel CCs are visible in 3D.
            a3d = 0.92
            color_tf.AddRGBPoint(float(lid), r, g, b)
            opacity_tf.AddPoint(float(lid) - 0.45, 0.0)
            opacity_tf.AddPoint(float(lid), a3d)
            opacity_tf.AddPoint(float(lid) + 0.45, 0.0)

        volume_property.SetColor(color_tf)
        volume_property.SetScalarOpacity(opacity_tf)
        volume_property.SetInterpolationTypeToNearest()
        volume_property.ShadeOff()
        try:
            volume_property.SetScalarOpacityUnitDistance(0.5)
        except Exception:
            pass
        try:
            # Disable gradient opacity so labels stay solid.
            volume_property.SetDisableGradientOpacity(1)
        except Exception:
            pass
        vr_display_node.SetVisibility(True)

    def write_labelmap(
        self,
        labels: np.ndarray,
        *,
        reference_volume,
        name: str,
    ):
        labels = np.asarray(labels)
        existing = slicer.util.getFirstNodeByClassByName("vtkMRMLLabelMapVolumeNode", name)
        if existing is not None:
            labelNode = existing
        else:
            volumesLogic = slicer.modules.volumes.logic()
            labelNode = volumesLogic.CreateAndAddLabelVolume(reference_volume, name)

        imageData = labelNode.GetImageData()
        if imageData is None or tuple(imageData.GetDimensions()) != (
            labels.shape[2],
            labels.shape[1],
            labels.shape[0],
        ):
            volumesLogic = slicer.modules.volumes.logic()
            slicer.mrmlScene.RemoveNode(labelNode)
            labelNode = volumesLogic.CreateAndAddLabelVolume(reference_volume, name)

        max_id = int(labels.max()) if labels.size else 0
        if max_id > np.iinfo(np.int16).max:
            raise ValueError(
                f"Label id {max_id} exceeds int16 labelmap range; too many CCs."
            )
        slicer.util.updateVolumeFromArray(labelNode, labels.astype(np.int16, copy=False))
        labelNode.SetName(name)
        slicer.util.setSliceViewerLayers(label=labelNode, fit=False)
        return labelNode

    def run_stage1(self, inputVolumeNode):
        """Run Stage 1; return (cc_node, labeled, raw_intensity)."""
        self.ensure_deps()

        arr = self.array_from_volume(inputVolumeNode)
        if arr.ndim != 3:
            raise ValueError(f"Mouse TOF CoW expects a 3D volume, got shape {arr.shape}")

        logging.info("Mouse TOF CoW Stage 1: starting on %s", inputVolumeNode.GetName())
        slicer.app.processEvents()
        labeled, raw = run_stage1(inputVolumeNode)
        labeled = np.asarray(labeled, dtype=np.int32)
        raw = np.asarray(raw, dtype=np.float32)

        source_name = inputVolumeNode.GetName()
        cc_name = f"{source_name}_tof_cow_cc"
        cc_node = self.write_labelmap(labeled, reference_volume=inputVolumeNode, name=cc_name)
        logging.info(
            "Mouse TOF CoW Stage 1 done: %s CCs → %s",
            int(labeled.max()) if labeled.size else 0,
            cc_name,
        )
        return cc_node, labeled, raw

    def apply_tree_colors(self, labelNode) -> None:
        colorNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLColorTableNode",
            f"{labelNode.GetName()}_colors",
        )
        self.resize_color_table(colorNode, 4)
        colorNode.SetColor(0, 0.0, 0.0, 0.0, 0.0)
        names = {1: "Left ICA", 2: "Right ICA", 3: "Basilar"}
        rgba_by_label: dict[int, tuple[float, float, float, float]] = {}
        for lab, rgb in _TREE_COLORS_RGB.items():
            r, g, b = rgb
            colorNode.SetColor(lab, r / 255.0, g / 255.0, b / 255.0, 1.0)
            colorNode.SetColorName(lab, names.get(lab, str(lab)))
            rgba_by_label[lab] = (r / 255.0, g / 255.0, b / 255.0, 1.0)
        displayNode = labelNode.GetDisplayNode()
        if displayNode is None:
            labelNode.CreateDefaultDisplayNodes()
            displayNode = labelNode.GetDisplayNode()
        if displayNode is not None:
            displayNode.SetAndObserveColorNodeID(colorNode.GetID())
        self.show_labelmap_in_3d(labelNode, rgba_by_label=rgba_by_label)
        # Apply solid TFs for the three trees
        vr = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(
            labelNode
        )
        if vr is not None:
            self.apply_label_transfer_functions(vr, rgba_by_label, max_label=3)


#
# Widget
#


class MouseTOFCoWWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """UI: Stage 1 run + Stage 2 Add / Deselect / Tree done / Cancel."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._session: MouseTofCowSlicerSession | None = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = MouseTOFCoWLogic()

        parametersCollapsible = ctk.ctkCollapsibleButton()
        parametersCollapsible.text = "Input"
        self.layout.addWidget(parametersCollapsible)
        formLayout = qt.QFormLayout(parametersCollapsible)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = True
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("3D TOF intensity volume")
        formLayout.addRow("TOF volume:", self.inputSelector)

        self.runStage1Button = qt.QPushButton("Run Stage 1 (N4 → blood flood → CCs)")
        self.runStage1Button.toolTip = (
            "Slicer N4ITKBiasFieldCorrection + Frangi/hysteresis CCs (Lab recipe; no ANTsPy)."
        )
        formLayout.addRow(self.runStage1Button)
        self.runStage1Button.connect("clicked(bool)", self.onRunStage1)

        self.installDepsButton = qt.QPushButton("Install optional deps (skimage, sklearn)")
        self.installDepsButton.toolTip = (
            "pip-install scikit-image and scikit-learn into Slicer's Python (once)."
        )
        formLayout.addRow(self.installDepsButton)
        self.installDepsButton.connect("clicked(bool)", self.onInstallDeps)

        stage2Collapsible = ctk.ctkCollapsibleButton()
        stage2Collapsible.text = "Stage 2 — assign CCs to trees"
        self.layout.addWidget(stage2Collapsible)
        stage2Layout = qt.QVBoxLayout(stage2Collapsible)

        self.statusLabel = qt.QLabel("Idle — run Stage 1 to begin.")
        self.statusLabel.setWordWrap(True)
        stage2Layout.addWidget(self.statusLabel)

        btnRow = qt.QHBoxLayout()
        self.addCcButton = qt.QPushButton("Add CC to tree")
        self.deselectButton = qt.QPushButton("Deselect")
        self.treeDoneButton = qt.QPushButton("Tree done")
        self.cancelButton = qt.QPushButton("Cancel")
        for b in (
            self.addCcButton,
            self.deselectButton,
            self.treeDoneButton,
            self.cancelButton,
        ):
            btnRow.addWidget(b)
        stage2Layout.addLayout(btnRow)

        cutRow = qt.QHBoxLayout()
        self.splitSliceButton = qt.QPushButton("Split CC by active slice")
        self.splitSliceButton.toolTip = (
            "Highlight a CC that bridges L/R (or other trees). Orient a slice through the "
            "bridge, then split into two (or more) components you can assign separately."
        )
        self.cutModeCheck = qt.QCheckBox("Cut mode (click bridge to erase)")
        self.cutModeCheck.toolTip = (
            "When enabled, clicks carve a small sphere out of the highlighted CC and "
            "re-label fragments — use to sever thin connections."
        )
        cutRow.addWidget(self.splitSliceButton)
        cutRow.addWidget(self.cutModeCheck)
        stage2Layout.addLayout(cutRow)

        self.addCcButton.connect("clicked(bool)", self.onAddCc)
        self.deselectButton.connect("clicked(bool)", self.onDeselect)
        self.treeDoneButton.connect("clicked(bool)", self.onTreeDone)
        self.cancelButton.connect("clicked(bool)", self.onCancel)
        self.splitSliceButton.connect("clicked(bool)", self.onSplitSlice)
        self.cutModeCheck.connect("toggled(bool)", self.onCutModeToggled)

        helpLabel = qt.QLabel(
            "After Stage 1: left-click a CC (slice or 3D) to highlight, then Add CC to tree. "
            "If one CC spans Left and Right ICA: highlight it, put a slice through the bridge, "
            "Split CC by active slice — or enable Cut mode and click the bridge. "
            "Finish Left ICA → Right ICA → Basilar with Tree done. "
            "On finish, intermediate CC overlays are removed; only final trees remain."
        )
        helpLabel.setWordWrap(True)
        helpLabel.setStyleSheet("color: gray;")
        stage2Layout.addWidget(helpLabel)

        self.layout.addStretch(1)
        self._syncStage2Enabled()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)

    def cleanup(self):
        self._teardownSession(notify=False)
        self.removeObservers()

    def onSceneStartClose(self, caller, event):
        self._teardownSession(notify=False)

    def _sessionActive(self) -> bool:
        return self._session is not None and not self._session._finished

    def _syncStage2Enabled(self) -> None:
        active = self._sessionActive()
        for b in (
            self.addCcButton,
            self.deselectButton,
            self.treeDoneButton,
            self.cancelButton,
            self.splitSliceButton,
            self.cutModeCheck,
        ):
            b.setEnabled(active)
        self.runStage1Button.setEnabled(not active)
        if not active:
            self.cutModeCheck.setChecked(False)

    def _setStatus(self, text: str) -> None:
        self.statusLabel.setText(text)

    def _teardownSession(self, *, notify: bool = True) -> None:
        if self._session is not None:
            self._session.cancel()
            if notify:
                self._setStatus("Cancelled — CC labelmap left as-is.")
            self._session = None
        self._syncStage2Enabled()

    def onInstallDeps(self):
        with slicer.util.tryWithErrorDisplay("Dependency install failed.", waitCursor=True):
            slicer.util.pip_install("scikit-image scikit-learn")
        slicer.util.infoDisplay(
            "Installed scikit-image and scikit-learn into Slicer's Python.\n"
            "Restart Slicer if imports still fail.",
            windowTitle="Mouse TOF CoW",
        )

    def onRunStage1(self):
        if self._sessionActive():
            slicer.util.warningDisplay(
                "A Stage-2 session is active. Cancel it first.",
                windowTitle="Mouse TOF CoW",
            )
            return
        inputVolume = self.inputSelector.currentNode()
        if inputVolume is None:
            slicer.util.warningDisplay(
                "Select a 3D TOF scalar volume.",
                windowTitle="Mouse TOF CoW",
            )
            return

        try:
            with slicer.util.tryWithErrorDisplay("Stage 1 failed.", waitCursor=True):
                cc_node, labeled, raw = self.logic.run_stage1(inputVolume)
        except Exception:
            return

        self._session = MouseTofCowSlicerSession(
            source_name=inputVolume.GetName(),
            intensity=raw,
            cc_source=labeled,
            cc_volume_node=cc_node,
            status_callback=self._setStatus,
            on_finished=self._onSessionFinished,
        )
        self._session.install_pick_observers()
        self._session.refresh_display()
        self._syncStage2Enabled()
        self._setStatus(self._session.status_text())

    def _onSessionFinished(self, trees_node) -> None:
        self._session = None
        self._syncStage2Enabled()
        self._setStatus(
            f"Done — output: {trees_node.GetName()} (1=Left ICA, 2=Right ICA, 3=Basilar)."
        )

    def onAddCc(self):
        if self._sessionActive():
            self._session.add_highlighted_cc()

    def onDeselect(self):
        if self._sessionActive():
            self._session.clear_highlight()

    def onSplitSlice(self):
        if self._sessionActive():
            with slicer.util.tryWithErrorDisplay("Split failed.", waitCursor=True):
                self._session.split_highlighted_cc_by_active_slice()

    def onCutModeToggled(self, checked):
        if self._session is not None:
            self._session.cut_mode = bool(checked)
            if checked:
                self._setStatus(
                    self._session.status_text()
                    + " — CUT MODE: click the bridge on the highlighted CC"
                )
            else:
                self._setStatus(self._session.status_text())

    def onTreeDone(self):
        if self._sessionActive():
            with slicer.util.tryWithErrorDisplay("Tree finalize failed.", waitCursor=True):
                self._session.finish_current_tree()
            self._syncStage2Enabled()

    def onCancel(self):
        self._teardownSession(notify=True)
