"""Manim animation of the eICAB post-process (qvtpy stage 1).

Walks through :func:`nvitk.pipes.qvtpy.util.eicab.eicab_postprocess.postprocess_eicab_labels`
on a real subject (PESA10061584 by default): eICAB's Circle-of-Willis labels, the anterior
choroidals dropped, the local half-maximum re-segmentation, the ICAs taken to the image floor,
the competitive distal growth along the Frangi tree with the ACA midline
guard, and the ICA siphon correction.

Every state comes from ``eicab_pp_stages.npz``, which ``eicab_pp_precompute.py`` produces by
running the post-process and recording each step's output. This file only draws; it needs numpy
and manim, not nvitk.

Render (manim environment; ``--no-text`` drops every caption, title, legend and inset)::

    EP_FAST=1 python scripts/anim/eicab_pp_manim.py -ql --fps 15
    python scripts/anim/eicab_pp_manim.py -qh --fps 30              # -> eicab_postprocess.mp4
    python scripts/anim/eicab_pp_manim.py -qh --fps 30 --no-text    # -> eicab_postprocess_notext.mp4
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
    OUT,
    RIGHT,
    UL,
    UP,
    UR,
    Create,
    Cube,
    FadeIn,
    FadeOut,
    ImageMobject,
    Rectangle,
    Text,
    VGroup,
    VMobject,
    config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anim_voxels import FONT, INK, INK_FAINT, INK_SOFT, CaptionScene, VoxelField, no_text, rgb, run_scene  # noqa: E402

DATA_PATH = Path(os.environ.get("EP_STAGES", str(Path(__file__).with_name("eicab_pp_stages.npz"))))
FAST = os.environ.get("EP_FAST", "0") == "1"

config.background_color = "#FFFFFF"

#: One colour per eICAB label: left darker, right lighter.
LABEL_COLORS = {
    1: "#C99A06", 2: "#E5C34A",              # ICAs
    3: "#D9558F",                            # basilar
    4: "#6C3F8F",                            # AComm
    5: "#D63A2F", 6: "#EF867B",              # ACAs
    7: "#2A72C8", 8: "#7DAEE8",              # MCAs
    9: "#8E44AD", 10: "#BC8BD6",             # PComms
    11: "#127A70", 12: "#5DB5AA",            # PCA P1
    13: "#2E8C47", 14: "#7FC491",            # PCA P2
    15: "#B4581F", 16: "#DE9A6A",            # SCAs
    17: "#6F7A7B", 18: "#A6AFB0",            # AChAs
}
REMOVED = "#E8443A"
CUT = "#D6336C"
PLANE = "#6C3F8F"
PATH = "#1B1D23"

OBJECT_WIDTH = 7.4  # scene units across the vessels' widest extent (final mask)
#: The view looks a little below the vessels' centre, lifting them clear of the captions. A
#: vertical offset stays put while the camera turns about the vertical axis.
LIFT = 0.9
ROTATION = 0.007  # rad / s


def _occ_size(occ: np.ndarray) -> np.ndarray:
    """Cube size from a block's occupancy: volume-true, never vanishing below 0.32."""
    return np.where(occ > 0, np.clip(np.cbrt(occ), 0.32, 1.0), 0.32)


class EicabPostProcess(CaptionScene):
    """The eICAB post-process, step by step."""

    def construct(self) -> None:
        data = np.load(DATA_PATH)
        meta = json.loads(str(data["meta"]))
        names = {int(k): v for k, v in meta["names"].items()}
        T = (lambda x: max(0.4, x * 0.3)) if FAST else (lambda x: x)

        world_all = np.asarray(data["blk_world"], float)
        stages = ("raw", "noacha", "reseg", "ext", "distal", "final")
        # Blocks some stage labels (the precompute also keeps the unclaimed Frangi tree).
        shown = np.zeros(len(world_all), bool)
        for k in stages:
            shown |= np.asarray(data[f"occ_{k}"]) > 0
        keep = shown.copy()
        if FAST:
            keep &= np.random.default_rng(0).random(len(world_all)) < 0.45
        world = world_all[keep]

        def sel(key: str) -> np.ndarray:
            return np.asarray(data[key])[keep]

        lab = {k: sel(f"label_{k}").astype(int) for k in stages}
        occ = {k: sel(f"occ_{k}").astype(float) for k in lab}
        arrival = sel("arrival").astype(float)

        # geometry: world mm (RAS) -> scene
        final_w = world_all[np.asarray(data["occ_final"]) > 0]
        centre = (final_w.max(0) + final_w.min(0)) / 2.0
        scale = OBJECT_WIDTH / float(np.ptp(final_w[:, :2], axis=0).max())
        lift = np.array([0.0, 0.0, -LIFT])

        def view(zoom: float) -> np.ndarray:
            """Frame centre for *zoom*: the lift stays the same on screen."""
            return lift / zoom

        def scene(p: np.ndarray) -> np.ndarray:
            return (np.asarray(p, float) - centre) * scale

        aff = np.asarray(meta["affine"], float)

        def vox_to_world(p: np.ndarray) -> np.ndarray:
            p = np.atleast_2d(np.asarray(p, float))
            return (aff[:3, :3] @ p[:, :3].T).T + aff[:3, 3]

        block_mm = float(meta["block"]) * float(meta["spacing_mm"][0])
        field = VoxelField(scene(world), side=block_mm * scale)

        def style(stage: str, opacity: float = 0.82):
            lb, oc = lab[stage], occ[stage]
            col = np.array([rgb(LABEL_COLORS.get(int(v), "#93A2BC")) for v in lb])
            op = np.where(oc > 0, opacity, 0.0)
            sop = np.where(oc > 0, 0.9, 0.0)
            return col, op, sop, _occ_size(oc)

        # legend (labels present at the end)
        present = sorted({int(v) for v in np.unique(np.asarray(data["label_final"])) if v > 0}
                         | {int(v) for v in np.unique(np.asarray(data["label_raw"])) if v > 0})

        def legend_for(ids: list[int]) -> VGroup:
            items = [VGroup(Rectangle(width=0.16, height=0.16, fill_color=LABEL_COLORS[i], fill_opacity=1, stroke_width=0),
                            Text(names[i], font=FONT, color=INK, font_size=13)).arrange(RIGHT, buff=0.08)
                     for i in ids]
            half = (len(items) + 1) // 2
            cols = VGroup(VGroup(*items[:half]).arrange(DOWN, buff=0.06, aligned_edge=LEFT),
                          VGroup(*items[half:]).arrange(DOWN, buff=0.06, aligned_edge=LEFT))
            g = cols.arrange(RIGHT, buff=0.45, aligned_edge=UP).to_corner(UL, buff=0.35)
            g.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.1)
            self.fixed(g)
            return g

        def inset(key: str, width: float = 6.0):
            if no_text():  # the insets are annotated plots: an invisible stand-in keeps the timing
                img = Rectangle(width=width, height=1.0, stroke_width=0, fill_opacity=0).to_corner(UR, buff=0.35)
                self.fixed(img)
                return img
            img = ImageMobject(np.asarray(data[key], np.uint8))
            img.width = width
            img.to_corner(UR, buff=0.35)
            self.add_fixed_in_frame_mobjects(img)
            self.remove(img)
            return img

        # 0 · title
        self.set_camera_orientation(phi=62 * DEGREES, theta=96 * DEGREES, zoom=1.45, frame_center=view(1.45))
        self.add(field.group)
        title = Text("eICAB post-process", font=FONT, weight=BOLD, color=INK, font_size=34)
        sub = Text("qvtpy stage 1  ·  nvitk.pipes.qvtpy.util.eicab.eicab_postprocess", font=FONT, color=INK_SOFT,
                   font_size=19)
        who = Text(f"{meta['subject']}  ·  TOF MRA, 0.5 mm", font=FONT, color=INK_SOFT, font_size=18)
        card = VGroup(title, sub, who).arrange(DOWN, buff=0.18, aligned_edge=LEFT).to_corner(DL, buff=0.55)
        self.fixed(card)
        self.play(FadeIn(card), run_time=T(1.0))
        raw_col, raw_op, raw_sop, raw_size = style("raw")
        field.transition(self, raw_col, raw_op, raw_sop, raw_size, run_time=T(2.6), delay=world[:, 2], lag=0.7)
        self.begin_ambient_camera_rotation(rate=ROTATION)
        self.wait(T(1.0))

        # 1 · eICAB's labels
        v = meta["volume_mm3"]
        cap = self.caption("eICAB's Circle-of-Willis labels", [
            (f"{meta['n_raw_labels']} vessels labelled by eICAB's network, inside its CoW box", INK, 20),
            ("masks thicker than the lumen  ·  trunks cut at the box's floor", INK_SOFT, 18),
            ("distal arteries stop short", INK_SOFT, 18),
        ])
        legend = legend_for(present)
        self.swap(card, cap)
        self.play(FadeIn(legend), run_time=T(0.6))
        self.wait(T(2.4))

        # 2 · drop the anterior choroidals
        cap1 = self.caption("1 · Drop the anterior choroidals", [
            (f"LAChA / RAChA: {meta['acha_mm3']} mm³", INK, 20),
            ("small, and eICAB's least reliable labels", INK_SOFT, 18),
        ])
        self.swap(cap, cap1)
        gone = (occ["raw"] > 0) & (occ["noacha"] == 0)
        flash = np.where(gone[:, None], rgb(REMOVED), raw_col)
        field.transition(self, flash, np.where(gone, 0.95, raw_op), raw_sop, run_time=T(0.7))
        na_col, na_op, na_sop, na_size = style("noacha")
        field.transition(self, na_col, na_op, na_sop, na_size, run_time=T(1.0))
        self.wait(T(1.2))

        # 3 · local re-segmentation
        drop = (1.0 - v["reseg"] / max(v["noacha"], 1e-9)) * 100.0
        cap2 = self.caption("2 · Local re-segmentation at half maximum", [
            ("T = bg + ½(core − bg), per vessel, from its own intensities", INK, 20),
            ("core: its centreline  ·  bg: a 1–3 mm ring  ·  other labels are barriers", INK_SOFT, 18),
            (f"volume {v['noacha']:.0f} → {v['reseg']:.0f} mm³  (−{drop:.0f} %)", REMOVED, 18),
        ])
        self.swap(cap1, cap2)
        img = inset("inset_reseg")
        self.play(FadeIn(img), run_time=T(0.6))
        rs_col, rs_op, rs_sop, rs_size = style("reseg")
        field.transition(self, rs_col, rs_op, rs_sop, rs_size, run_time=T(2.6))
        self.wait(T(2.4))
        self.play(FadeOut(img), run_time=T(0.5))

        # 4 · trunks to the floor
        e = meta["extension"]

        def ext_text(lid: str, label: str) -> str:
            x = e.get(lid) or {}
            more = f", {x['branches']} branches" if x.get("branches", 1) > 1 else ""
            return f"{label} {x.get('length_mm', 0)} mm{more}"

        cap3 = self.caption("3 · ICAs to the image floor", [
            ("minimum-cost path through the Frangi tree (+ bright lumen)", INK, 20),
            ("from the open end, to a vessel cross-section at the floor", INK_SOFT, 18),
            (f"{ext_text('1', 'LICA')}  ·  {ext_text('2', 'RICA')}", INK, 18),
        ])
        self.swap(cap2, cap3)
        self.stop_ambient_camera_rotation()
        self.move_camera(zoom=1.15, frame_center=view(1.15), run_time=T(1.4))
        self.begin_ambient_camera_rotation(rate=ROTATION)
        img = inset("inset_ext")
        self.play(FadeIn(img), run_time=T(0.6))
        paths = np.asarray(data["ext_paths"], float)
        lines = VGroup()
        for k in np.unique(paths[:, 3]) if len(paths) else []:
            pts = scene(vox_to_world(paths[paths[:, 3] == k][:, :3]))
            if len(pts) > 2:
                m = VMobject(stroke_color=PATH, stroke_width=3.5)
                m.set_points_smoothly(list(pts[:: max(1, len(pts) // 40)]) + [pts[-1]])
                lines.add(m)
        self.play(*[Create(m) for m in lines], run_time=T(1.6))
        ex_col, ex_op, ex_sop, ex_size = style("ext")
        new = (occ["ext"] > 0) & (occ["reseg"] == 0)
        field.transition(self, ex_col, ex_op, ex_sop, ex_size, run_time=T(3.0),
                         delay=np.where(new, -world[:, 2], world[:, 2].min() - 1.0), lag=0.8,
                         glow=LABEL_COLORS[3], glow_strength=0.4)
        self.wait(T(1.6))
        self.play(FadeOut(img), FadeOut(lines), run_time=T(0.5))

        prev_cap = cap3

        # 6 · distal growth
        dg = meta["distal"]
        added = sum(float(x) for x in dg.values())
        cap5 = self.caption("5 · Distal growth along the Frangi tree", [
            ("ACAs, MCAs, PCAs (P2) and SCAs flood the tree at once", INK, 20),
            ("each tree voxel goes to the front that reaches it first", INK_SOFT, 18),
            ("cost 1–5 per mm by vesselness: the fronts keep to the lumen", INK_SOFT, 18),
            (f"+{added / 1000:.1f} cm³  (volume {v['ext']:.0f} → {v['distal']:.0f} mm³)", INK, 18),
        ])
        self.swap(prev_cap, cap5)
        self.stop_ambient_camera_rotation()
        self.move_camera(zoom=1.0, frame_center=view(1.0), run_time=T(1.4))
        self.begin_ambient_camera_rotation(rate=ROTATION)
        mid = meta.get("midplane")
        plane = None
        if mid:
            p0 = vox_to_world(np.asarray(mid["point_vox"], float))[0]
            sp = np.asarray(meta["spacing_mm"], float)
            nw = aff[:3, :3] @ (np.asarray(mid["normal"], float) / sp)
            nw = nw / np.linalg.norm(nw)
            # 40 mm tall × 60 mm front to back once turned onto the sagittal midline.
            plane = Rectangle(width=40 * scale, height=60 * scale, fill_color=PLANE, fill_opacity=0.14,
                              stroke_color=PLANE, stroke_width=1.5)
            # Rectangle lies in the XY plane (normal OUT): turn OUT onto the midline normal.
            axis = np.cross(OUT, nw)
            ang = float(np.arccos(np.clip(np.dot(OUT, nw), -1.0, 1.0)))
            if np.linalg.norm(axis) > 1e-9:
                plane.rotate(ang, axis=axis / np.linalg.norm(axis))
            plane.move_to(scene(p0))
        di_col, di_op, di_sop, di_size = style("distal")
        grow = (occ["distal"] > 0) & (occ["ext"] == 0)
        order = np.where(grow & (arrival >= 0), arrival, np.where(grow, np.nanmax(arrival) + 1.0, -1.0))
        field.transition(self, di_col, di_op, di_sop, di_size, run_time=T(8.0), delay=order, lag=0.9,
                         glow="#EFA43A", glow_strength=0.6)
        if plane is not None:
            cap5b = self.caption("ACA guard", [
                ("each side of the AComm seeds its own ACA", INK, 20),
                ("a midline plane (head outline, anchored at the AComm):", INK_SOFT, 18),
                (f"no ACA grows more than {mid['slack_mm']:g} mm into the other hemisphere", PLANE, 18),
            ])
            self.swap(cap5, cap5b)
            self.play(FadeIn(plane), run_time=T(0.8))
            self.wait(T(2.4))
            self.play(FadeOut(plane), run_time=T(0.6))
            prev_cap = cap5b
        else:
            prev_cap = cap5
        self.wait(T(1.0))

        # 7 · siphon correction
        sip = meta.get("siphon") or {}
        cut_entries = {k: x for k, x in sip.items() if (x.get("loops_cut") or 0) > 0}
        removed = np.asarray(data["siphon_removed"], float)
        if cut_entries and len(removed):
            lid, x = next(iter(cut_entries.items()))
            cap6 = self.caption("6 · ICA siphon correction", [
                (f"{names[int(lid)]}: β₁ {x['genus_before']['beta1']} → {x['genus_after']['beta1']}, "
                 f"{x['loops_cut']} loop cut", INK, 20),
                (f"pretess {x['pretess']['n_removed']} voxel(s) + a minimum cut of {x['cut_mm3']} mm³", CUT, 18),
                ("only the contact between the siphon's limbs goes", INK_SOFT, 18),
            ])
            self.swap(prev_cap, cap6)
            wpos = vox_to_world(removed)
            focus = scene(wpos.mean(0))
            cubes = VGroup()
            for p in wpos:
                c = Cube(side_length=0.5 * scale * 1.6, fill_color=CUT, fill_opacity=0.95)
                c.set_stroke(color=CUT, width=0.6, opacity=1.0)
                c.move_to(scene(p))
                cubes.add(c)
            self.stop_ambient_camera_rotation()
            ghost_col, ghost_op, ghost_sop, ghost_size = style("distal", opacity=0.22)
            field.transition(self, ghost_col, ghost_op, ghost_sop * 0.4, run_time=T(0.8))
            self.move_camera(frame_center=focus, zoom=2.4, run_time=T(1.6))
            self.play(FadeIn(cubes, scale=2.0), run_time=T(0.8))
            self.wait(T(1.8))
            fi_col, fi_op, fi_sop, fi_size = style("final")
            field.transition(self, fi_col, fi_op, fi_sop, fi_size, run_time=T(1.0), extra=[FadeOut(cubes, scale=0.2)])
            self.move_camera(frame_center=view(1.0), zoom=1.0, run_time=T(1.6))
            self.begin_ambient_camera_rotation(rate=ROTATION)
            prev_cap = cap6
        else:
            fi_col, fi_op, fi_sop, fi_size = style("final")
            field.transition(self, fi_col, fi_op, fi_sop, fi_size, run_time=T(0.8))

        # end card
        end = self.caption("Post-processed eICAB mask", [
            (f"volume {v['raw']:.0f} → {v['final']:.0f} mm³: thinner lumen, full trunks, distal trees", INK, 20),
            ("writes *_pp, centerlines_mask_pp and eicab_postprocess.json", INK_SOFT, 18),
            ("stage 7 measures it", INK_FAINT, 18),
        ])
        self.swap(prev_cap, end)
        self.wait(T(4.0))
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(end), FadeOut(legend), run_time=T(1.0))
        self.wait(T(0.5))


if __name__ == "__main__":
    run_scene(__file__, "EicabPostProcess", "eicab_postprocess")
