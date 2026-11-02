"""Manim animation of the ICA siphon correction (minimum cut).

Walks through :mod:`nvitk.morphology.siphon_cut` on a real looped right ICA (PESA13727025):
the loop and the centerline shortcut it causes, pretess, the bridge / curl split of the skeleton
loop, the contact, the voxel cost field, the ring opening with its source and sink, the minimum
vertex cut, and the corrected centerline.

Every state comes from ``siphon_cut_stages.npz``, which ``siphon_cut_precompute.py`` produces by
running the library and recording what it hands its own helpers. This file only draws; it needs
numpy and manim, not nvitk.

Render (manim environment; ``--no-text`` drops every caption, title, legend and badge)::

    SC_FAST=1 python scripts/anim/siphon_cut_manim.py -ql --fps 15
    python scripts/anim/siphon_cut_manim.py -qh --fps 30              # -> siphon_mincut.mp4
    python scripts/anim/siphon_cut_manim.py -qh --fps 30 --no-text    # -> siphon_mincut_notext.mp4
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from manim import (
    BOLD,
    DEGREES,
    DL,
    DOWN,
    LEFT,
    RIGHT,
    UR,
    Create,
    Cube,
    DashedVMobject,
    Dot3D,
    FadeIn,
    FadeOut,
    Line,
    Rectangle,
    Text,
    VGroup,
    VMobject,
    config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anim_voxels import FONT, INK, INK_FAINT, INK_SOFT, CaptionScene, VoxelField, ramp3, rgb, run_scene  # noqa: E402

DATA_PATH = Path(os.environ.get("SC_STAGES", str(Path(__file__).with_name("siphon_cut_stages.npz"))))
FAST = os.environ.get("SC_FAST", "0") == "1"

config.background_color = "#FFFFFF"

# palette
TOF_LO, TOF_MID, TOF_HI = "#DCE2EC", "#88A3C6", "#22456E"   # vessel coloured by TOF brightness
LOOP = "#EFA43A"
BRIDGE = "#8E44AD"
CURL = "#158C81"
CONTACT = "#E8443A"
PRETESS = "#D6336C"
CHEAP, MID, DEAR = "#F6C445", "#A9BCD4", "#1F3F66"            # cost field
SLAB = "#C7CCD4"
SOURCE = "#2E9E4F"
SINK = "#2F7FD6"
CUT = "#D6336C"
CL_BEFORE = "#DE4133"
CL_AFTER = "#2E9E4F"
GOOD = "#2E9E4F"

OBJECT_HEIGHT = 7.0  # scene units across the crop's longest extent


def _occ_size(occ: np.ndarray) -> np.ndarray:
    """Cube size from a block's occupancy: volume-true, never vanishing below 0.3."""
    return np.where(occ > 0, np.clip(np.cbrt(occ), 0.3, 1.0), 0.3)


class SiphonMinCut(CaptionScene):
    """The siphon correction, step by step."""

    def construct(self) -> None:
        data = np.load(DATA_PATH)
        meta = json.loads(str(data["meta"]))
        T = (lambda x: max(0.4, x * 0.3)) if FAST else (lambda x: x)

        centres = np.asarray(data["blk_centres"], float)
        keep = np.ones(len(centres), bool)
        if FAST:
            keep = np.random.default_rng(0).random(len(centres)) < 0.45
        centres = centres[keep]

        def sel(key: str) -> np.ndarray:
            return np.asarray(data[key])[keep]

        occ_raw, occ_pre, occ_rep = sel("occ_raw"), sel("occ_pretess"), sel("occ_repaired")
        tof_rel = sel("tof_rel")
        weight = sel("weight")
        slab, source, sink = sel("slab_frac") > 0, sel("source_frac") > 0, sel("sink_frac") > 0

        # geometry: voxel (crop) units -> scene
        all_pts = np.asarray(data["blk_centres"], float)
        centre = (all_pts.max(0) + all_pts.min(0)) / 2.0
        pitch = OBJECT_HEIGHT / float(np.ptp(all_pts, axis=0).max())  # scene units per 0.5 mm voxel

        def scene(p: np.ndarray) -> np.ndarray:
            return (np.asarray(p, float) - centre) * pitch

        block = int(meta["block"])
        field = VoxelField(scene(centres), side=pitch * block)

        loop = np.asarray(data["loop_nodes"], float)
        bridge = np.asarray(data["loop_bridge"], bool)
        contact = np.asarray(data["loop_contact"], bool)
        anchors = np.asarray(data["loop_anchors"], float)
        loop_c = scene(loop.mean(0))

        # camera: face-on to the loop's plane, from the ICA's own side, S up
        u, s, vt = np.linalg.svd(loop - loop.mean(0))
        normal = vt[-1]
        side = 1.0 if meta["label_name"].startswith("R") else -1.0  # RAS: +i is the patient's right
        if normal[0] * side < 0:
            normal = -normal
        theta0 = float(np.degrees(np.arctan2(normal[1], normal[0])))
        phi0 = float(np.clip(np.degrees(np.arccos(np.clip(normal[2], -1, 1))), 62.0, 84.0))

        # styles
        def tof_style(occ: np.ndarray):
            t = np.clip(tof_rel / max(np.percentile(tof_rel[occ > 0], 97), 1e-9), 0.0, 1.0)
            col = ramp3(TOF_LO, TOF_MID, TOF_HI, t)
            op = np.where(occ > 0, 0.30 + 0.45 * t, 0.0)
            sop = np.where(occ > 0, 0.35 + 0.35 * t, 0.0)
            return col, op, sop, _occ_size(occ)

        tof_col, tof_op, tof_sop, size_raw = tof_style(occ_raw)

        # overlays
        def small_cubes(points: np.ndarray, color: str) -> VGroup:
            g = VGroup()
            for p in points:
                c = Cube(side_length=pitch * 0.92, fill_color=color, fill_opacity=0.95)
                c.set_stroke(color=color, width=0.6, opacity=1.0)
                c.move_to(scene(p))
                g.add(c)
            return g

        def polyline(points: np.ndarray, color: str, width: float, closed: bool = False, smooth: bool = True) -> VMobject:
            pts = [scene(p) for p in points]
            if closed:
                pts.append(pts[0])
            m = VMobject(stroke_color=color, stroke_width=width)
            m.set_points_smoothly(pts) if smooth else m.set_points_as_corners(pts)
            return m

        def loop_segments(color_of) -> VGroup:
            g = VGroup()
            k = len(loop)
            for i in range(k):
                j = (i + 1) % k
                g.add(Line(scene(loop[i]), scene(loop[j]), stroke_width=7, color=color_of(i, j)))
            return g

        beta = VGroup(Text("β₁", font=FONT, color=INK_SOFT, font_size=26),
                      Text(str(meta["betti_raw"]["beta1"]), font=FONT, weight=BOLD, color=CONTACT, font_size=34))
        beta.arrange(RIGHT, buff=0.18).to_corner(UR, buff=0.55)
        self.fixed(beta)

        def set_beta(value: int, color: str) -> None:
            new = Text(str(value), font=FONT, weight=BOLD, color=color, font_size=34).move_to(beta[1])
            self.fixed(new)
            self.play(FadeOut(beta[1]), FadeIn(new), run_time=T(0.6))
            beta.remove(beta[1])
            beta.add(new)

        # 0 · title
        self.set_camera_orientation(phi=phi0 * DEGREES, theta=(theta0 - 18) * DEGREES, zoom=1.0)
        self.add(field.group)
        title = Text("ICA siphon correction", font=FONT, weight=BOLD, color=INK, font_size=34)
        sub = Text("minimum cut  ·  nvitk.morphology.siphon_cut", font=FONT, color=INK_SOFT, font_size=21)
        who = Text(f"{meta['subject']}  ·  {meta['label_name']} (eICAB label {meta['label']})",
                   font=FONT, color=INK_SOFT, font_size=19)
        card = VGroup(title, sub, who).arrange(DOWN, buff=0.18, aligned_edge=LEFT).to_corner(DL, buff=0.55)
        self.fixed(card)
        self.play(FadeIn(card), run_time=T(1.0))
        field.transition(self, tof_col, tof_op, tof_sop, size_raw, run_time=T(2.6),
                         delay=-centres[:, 2], lag=0.7)
        self.begin_ambient_camera_rotation(rate=0.011)
        self.wait(T(1.2))

        # 1 · the loop
        cap = self.caption("1 · A loop in the mask", [
            ("the siphon's two limbs touch by partial volume on TOF", INK, 20),
            ("the mask closes into a torus: β₁ = 1", CONTACT, 19),
            (f"a skeleton centerline shortcuts across: {meta['cl_before_mm']} mm", CL_BEFORE, 19),
        ])
        self.swap(card, cap)
        self.play(FadeIn(beta), run_time=T(0.5))
        self.stop_ambient_camera_rotation()
        self.move_camera(frame_center=loop_c + np.array([0.0, 0.0, -0.35]), zoom=1.75, run_time=T(1.6))
        self.begin_ambient_camera_rotation(rate=0.011)
        ring = polyline(loop, LOOP, 7, closed=True)
        before = DashedVMobject(polyline(np.asarray(data["cl_before"], float), CL_BEFORE, 4), num_dashes=70)
        self.play(Create(ring), run_time=T(1.6))
        self.play(Create(before), run_time=T(1.8))
        self.wait(T(1.4))

        # 2 · pretess
        cap2 = self.caption("2 · Pretess", [
            ("edge- and corner-only contacts: 26-connected, not 6-connected", INK, 20),
            ("the darker voxel of each goes (as FreeSurfer's mri_pretess)", INK_SOFT, 19),
            (f"{meta['pretess_removed']} voxels removed  ·  β₁ still {meta['betti_pretess']['beta1']}",
             PRETESS, 19),
        ])
        self.swap(cap, cap2)
        pre = small_cubes(np.asarray(data["pretess_removed"], float), PRETESS)
        field.transition(self, tof_col, tof_op * 0.28, tof_sop * 0.35, run_time=T(0.8), extra=[FadeOut(before)])
        self.play(FadeIn(pre, scale=1.6), run_time=T(0.9))
        self.wait(T(0.8))
        pre_col, pre_op, pre_sop, size_pre = tof_style(occ_pre)
        field.transition(self, pre_col, pre_op, pre_sop, size_pre, run_time=T(1.0),
                         extra=[FadeOut(pre, scale=0.3)])
        self.wait(T(1.0))

        # 3 · bridge and curl
        cap3 = self.caption("3 · Bridge and curl", [
            (f"skeleton cycle of {meta['loop_length_mm']} mm, split at its anchors", INK, 20),
            (f"bridge {meta['bridge_mm']} mm  (the short arc, across the contact)", BRIDGE, 19),
            (f"curl {meta['curl_mm']} mm  (the vessel's real path)", CURL, 19),
        ])
        self.swap(cap2, cap3)
        arcs = loop_segments(lambda i, j: BRIDGE if (bridge[i] or bridge[j]) else CURL)
        anchor_dots = VGroup(*[Dot3D(scene(a), radius=pitch * 1.1, color=INK) for a in anchors])
        self.play(FadeOut(ring), FadeIn(arcs), FadeIn(anchor_dots), run_time=T(1.0))
        self.wait(T(2.0))

        # 4 · the contact
        cap4 = self.caption("4 · Find the contact", [
            ("score = (I / I_curl) · (r / r_curl) along the loop", INK, 20),
            ("the limbs pinch and darken where they touch", INK_SOFT, 19),
            (f"dip {meta['contact_min_score']} over {meta['contact_nodes']} nodes, on the bridge", CONTACT, 19),
        ])
        self.swap(cap3, cap4)
        contact_dots = VGroup(*[Dot3D(scene(p), radius=pitch * 1.6, color=CONTACT) for p in loop[contact]])
        self.play(FadeIn(contact_dots, scale=1.8), run_time=T(0.9))
        self.wait(T(2.0))

        # 5 · cost field
        cap5 = self.caption("5 · The cost of removing each voxel", [
            ("w = clip(I / I_ref) · (0.25 + core)", INK, 20),
            ("core = 1 on a limb's centreline, 0 at the wall", INK_SOFT, 19),
            ("and at a contact, however deep inside the mask", INK_SOFT, 19),
            ("yellow: cheap to cut  ·  dark blue: lumen", INK, 19),
        ])
        self.swap(cap4, cap5)
        wn = np.clip(weight / max(np.percentile(weight[occ_pre > 0], 95), 1e-9), 0.0, 1.0)
        w_col = ramp3(CHEAP, MID, DEAR, wn)
        w_op = np.where(occ_pre > 0, 0.30 + 0.45 * wn, 0.0)
        w_op = np.where((occ_pre > 0) & (wn < 0.25), 0.85, w_op)
        w_sop = np.where(occ_pre > 0, 0.4, 0.0)
        field.transition(self, w_col, w_op, w_sop, run_time=T(1.8), extra=[FadeOut(arcs), FadeOut(anchor_dots)])
        self.wait(T(2.2))

        # 6 · open the ring
        cap6 = self.caption("6 · Open the ring opposite the contact", [
            ("a slab of the loop is taken out of the graph", SLAB, 20),
            ("source and sink sit just beside it:", INK_SOFT, 19),
            ("every source → sink path crosses the contact", INK_SOFT, 19),
        ])
        self.swap(cap5, cap6)
        o_col = np.where(slab[:, None], rgb(SLAB), w_col)
        o_col = np.where(source[:, None], rgb(SOURCE), o_col)
        o_col = np.where(sink[:, None], rgb(SINK), o_col)
        o_op = np.where(slab, 0.08, w_op)
        o_op = np.where(source | sink, 0.9, o_op)
        o_sop = np.where(slab, 0.1, np.where(source | sink, 0.9, w_sop))
        field.transition(self, o_col, o_op, o_sop, run_time=T(1.6))
        legend = VGroup(*[
            VGroup(Rectangle(width=0.22, height=0.22, fill_color=c, fill_opacity=1, stroke_width=0),
                   Text(t, font=FONT, color=INK, font_size=18)).arrange(RIGHT, buff=0.12)
            for c, t in ((SOURCE, "source"), (SINK, "sink"), (SLAB, "opened slab"))
        ]).arrange(DOWN, buff=0.12, aligned_edge=LEFT).next_to(beta, DOWN, buff=0.4).align_to(beta, RIGHT)
        self.fixed(legend)
        self.play(FadeIn(legend), run_time=T(0.5))
        self.wait(T(2.0))

        # 7 · the minimum cut
        ratio = meta["cut_cost_ratio"]
        cap7 = self.caption("7 · Minimum s–t vertex cut", [
            ("max flow on a node-split 26-adjacency graph (SciPy, Dinic)", INK, 20),
            (f"cut cost {meta['cut_cost']} = {ratio} × a lumen slice  (≤ 0.6 accepted)", INK_SOFT, 19),
            (f"{meta['cut_voxels']} voxels = {meta['cut_mm3']} mm³ at the contact", CUT, 19),
        ])
        self.swap(cap6, cap7)
        cut = small_cubes(np.asarray(data["cut_voxels"], float), CUT)
        field.transition(self, o_col, o_op * 0.3, o_sop * 0.35, run_time=T(0.8))
        self.play(FadeIn(cut, scale=2.0), run_time=T(1.2))
        self.wait(T(2.2))

        # 8 · accept
        cap8 = self.caption("8 · Accept the cut", [
            (f"β₁ {meta['betti_raw']['beta1']} → {meta['betti_repaired']['beta1']}, still one component",
             GOOD, 20),
            (f"the rest of the ICA keeps eICAB's boundary: {meta['volume_change_pct']} % volume", INK_SOFT, 19),
            ("both ICAs get the same rule: no left / right bias", INK_SOFT, 19),
        ])
        self.swap(cap7, cap8)
        rep_col, rep_op, rep_sop, size_rep = tof_style(occ_rep)
        field.transition(self, rep_col, rep_op, rep_sop, size_rep, run_time=T(1.4),
                         extra=[FadeOut(cut, scale=0.2), FadeOut(legend), FadeOut(contact_dots)])
        set_beta(int(meta["betti_repaired"]["beta1"]), GOOD)
        self.wait(T(1.4))

        # 9 · centerline
        cap9 = self.caption("9 · Centerline along the curl", [
            (f"before: {meta['cl_before_mm']} mm, across the contact", CL_BEFORE, 20),
            (f"after: {meta['cl_after_mm']} mm, round the siphon", CL_AFTER, 20),
        ])
        self.swap(cap8, cap9)
        before2 = DashedVMobject(polyline(np.asarray(data["cl_before"], float), CL_BEFORE, 3.5), num_dashes=70)
        after = polyline(np.asarray(data["cl_after"], float), CL_AFTER, 6)
        self.play(FadeIn(before2), run_time=T(0.6))
        self.play(Create(after), run_time=T(2.4))
        self.stop_ambient_camera_rotation()
        self.move_camera(frame_center=np.zeros(3), zoom=1.0, run_time=T(1.8))
        self.begin_ambient_camera_rotation(rate=0.011)
        self.wait(T(2.0))

        # end card
        end = self.caption("Siphon loop opened", [
            (f"pretess {meta['pretess_removed']} voxels + one cut of {meta['cut_mm3']} mm³", INK, 20),
            (f"ICA volume {meta['volume_change_pct']} %  ·  centerline {meta['cl_before_mm']} → "
             f"{meta['cl_after_mm']} mm", INK_SOFT, 19),
            ("in the eICAB post-process and the GUI's siphon tool", INK_FAINT, 18),
        ])
        self.swap(cap9, end)
        self.wait(T(3.0))
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(end), FadeOut(beta), FadeOut(before2), FadeOut(after), run_time=T(1.0))
        self.wait(T(0.5))


if __name__ == "__main__":
    run_scene(__file__, "SiphonMinCut", "siphon_mincut")
