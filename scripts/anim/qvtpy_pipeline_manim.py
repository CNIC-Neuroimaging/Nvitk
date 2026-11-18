"""Manim animation of the qvtpy 4D-flow pipeline, stages 2-5, on the example subject.

From the TOF segmentation (the post-processed eICAB labels) to the 4D-flow labels and their
measuring stations:

- stage 2: the rigid FLIRT registration of the TOF onto the 4D-flow angiogram;
- stage 3: the labels carried into the 4D-flow grid and the centerlines (the backbone: arteries
  from the labels, sinuses from the CD);
- stage 4: the 4D-flow segmentation: a local CD threshold per vessel around its centerline, region
  growing of the ICAs and basilar, then the blood flood (Frangi + hysteresis tree, watershed from
  the A1 / M1 / P1, written as their A2 / M2 / P2) and the siphon correction;
- stage 5: the LOCs, one measuring plane per vessel, on the segmentation's centerlines.

Every state comes from ``qvtpy_pipeline_stages.npz`` (``qvtpy_pipeline_precompute.py``, which
re-runs stage 4 and reproduces the shipped ``seg_4dflow`` voxel for voxel); this file needs numpy,
scipy, moderngl and manim only. Every voxel is drawn at its own grid on the GPU (:mod:`gl_scene`).

Render (manim environment)::

    QP_FAST=1 python scripts/anim/qvtpy_pipeline_manim.py -ql --fps 15
    python scripts/anim/qvtpy_pipeline_manim.py -qh --fps 30              # -> qvtpy_pipeline.mp4
    python scripts/anim/qvtpy_pipeline_manim.py -qh --fps 30 --no-text
"""

from __future__ import annotations

import json
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
from pathlib import Path

import numpy as np
from manim import DOWN, LEFT, RIGHT, UL, UP, FadeIn, FadeOut, Rectangle, Text, VGroup
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anim_voxels import FONT, INK, INK_SOFT, rgb, run_scene  # noqa: E402
from gl_render import Quad, Spheres, Tubes, VolumeMIP, Voxels  # noqa: E402
from gl_scene import GLScene  # noqa: E402
from vessel_palette import ADDED, CUT, GHOST, QVTPY_COLORS, QVTPY_NAMES, colors_of  # noqa: E402

DATA_PATH = Path(os.environ.get("QP_STAGES", str(Path(__file__).with_name("qvtpy_pipeline_stages.npz"))))
FAST = os.environ.get("QP_FAST", "0") == "1"
ORBIT = 3.0
SHIFT = (0.0, 0.14)
TOF_TINT = (0.16, 0.17, 0.2)
FLOW_TINT = (0.12, 0.38, 0.72)
CL_INK = "#1B1D23"
LOC_C = "#E07B22"


def rigid_path(W: np.ndarray, centre: np.ndarray):
    """``f -> 4 × 4``: the rigid transform *W* applied gradually (rotation about *centre* by an
    increasing angle about W's axis, translation linear)."""
    R, t = W[:3, :3], W[:3, 3]
    ang = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if ang > 1e-9:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    else:
        axis = np.array([0.0, 0.0, 1.0])
    tc = R @ centre + t - centre  # W x = c + R (x - c) + tc

    def at(f: float) -> np.ndarray:
        a = ang * f
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        Rf = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        M = np.eye(4)
        M[:3, :3] = Rf
        M[:3, 3] = centre - Rf @ centre + tc * f
        return M

    return at


def disc(n: int = 128, color: str = LOC_C, alpha: float = 0.42) -> np.ndarray:
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r = np.sqrt(x * x + y * y)
    a = np.clip((1.0 - r) / 0.06, 0, 1) * alpha
    a = np.maximum(a, np.where(np.abs(r - 0.97) < 0.03, 0.95, 0.0))
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = (rgb(color) * 255).astype(np.uint8)
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


class QvtpyPipeline(GLScene):
    """qvtpy stages 2-5 on the example subject."""

    def construct(self) -> None:
        data = np.load(DATA_PATH)
        meta = json.loads(str(data["meta"]))
        T = (lambda x: max(0.3, x * 0.35)) if FAST else (lambda x: x)

        # ---- data
        ta = np.asarray(data["tof_affine"], float)
        tw = (ta[:3, :3] @ np.asarray(data["tof_vox"], float).T).T + ta[:3, 3]
        tlab = np.asarray(data["tof_lab"], int)
        W = np.asarray(data["W"], float)
        fa = np.asarray(data["f_affine"], float)
        fw = (fa[:3, :3] @ np.asarray(data["f_vox"], float).T).T + fa[:3, 3]
        e4d = np.asarray(data["e4d"], int)
        venous = np.asarray(data["venous"], bool)
        clm = np.asarray(data["clm"], int)
        tree = np.asarray(data["tree"], bool)
        order = np.asarray(data["order"], float)
        seg = {k: np.asarray(data[f"seg_{k}"], int) for k in ("thr", "rg", "vsplit", "preflood", "flood", "final")}
        sp4 = np.asarray(meta["spacing_4d"], float)

        # ---- layers
        tvox = Voxels(tw, pitch=0.5, fill=0.9)
        fvox = Voxels(fw, pitch=float(sp4[0]), fill=0.9)
        tof_mip = VolumeMIP(np.asarray(data["tof8"]), np.asarray(data["tof8_affine"], float), color=TOF_TINT,
                            opacity=0.0, lo=0.2, hi=0.85)
        angio_mip = VolumeMIP(np.asarray(data["angio8"]), np.asarray(data["f8_affine"], float), color=FLOW_TINT,
                              opacity=0.0, lo=0.32, hi=0.95, gamma=1.6)
        tof_base = tof_mip.model.copy()
        self.gl.add(angio_mip, tof_mip, fvox, tvox)

        centre = 0.5 * (fw[seg["final"] > 0].max(0) + fw[seg["final"] > 0].min(0))
        extent = float(np.ptp(fw[seg["final"] > 0], axis=0)[:2].max())
        dist = extent / (2 * np.tan(np.radians(12))) * 1.05
        self.set_camera(target=centre, distance=dist * 1.35, phi=74, theta=96, fov=24, shift=SHIFT)
        self.orbit(ORBIT)

        def fstyle(lab: np.ndarray, op: float = 1.0, table=QVTPY_COLORS):
            return colors_of(lab, table), np.where(lab > 0, op, 0.0).astype("f4"), np.where(lab > 0, 1.0, 0.4)

        def legend(ids: list[int]) -> VGroup:
            items = [VGroup(Rectangle(width=0.16, height=0.16, fill_color=QVTPY_COLORS[i], fill_opacity=1,
                                      stroke_width=0),
                            Text(QVTPY_NAMES[i], font=FONT, color=INK, font_size=13)).arrange(RIGHT, buff=0.08)
                     for i in ids if i in QVTPY_COLORS]
            half = (len(items) + 1) // 2
            cols = VGroup(VGroup(*items[:half]).arrange(DOWN, buff=0.06, aligned_edge=LEFT),
                          VGroup(*items[half:]).arrange(DOWN, buff=0.06, aligned_edge=LEFT))
            g = cols.arrange(RIGHT, buff=0.45, aligned_edge=UP).to_corner(UL, buff=0.35)
            g.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.1)
            self.fixed(g, keep=True)  # the colour key stays in the text-free version
            return g

        # ------------------------------------------------------------ 0 · title, two acquisitions
        card = self.title_card("From TOF to 4D flow: the qvtpy pipeline",
                               "stages 2-5  ·  registration · centerlines · 4D-flow segmentation · LOCs",
                               "the same head, two acquisitions")
        self.play(FadeIn(card), run_time=T(1.0))
        cap0 = self.caption("Two acquisitions of the same arteries", [
            ("TOF MRA: 0.5 mm, the labelled arterial tree (stage 1)", INK, 20),
            ("4D flow: 0.78 mm, its own angiogram and a velocity per voxel and frame", "#1F61B8", 20),
            ("acquired in different positions: they do not overlap", INK_SOFT, 18),
        ])

        def show_both(f: float) -> None:
            tof_mip.opacity = 0.75 * f
            angio_mip.opacity = 0.55 * f

        self.animate_gl(show_both, run_time=T(2.0))
        self.swap(card, cap0)
        self.voxel_transition(tvox, colors_of(tlab), np.full(len(tvox), 1.0), run_time=T(2.0), delay=tw[:, 2],
                              lag=0.6)
        self.hold(T(2.0))

        # ------------------------------------------------------------ 1 · stage 2 registration
        reg = meta["registration"]
        tmm = float(np.linalg.norm(meta["translation_mm"]))
        cap1 = self.caption("Stage 2 · Rigid registration", [
            (f"FSL FLIRT, {reg.get('dof', 6)} degrees of freedom, {reg.get('cost', 'normmi')} "
             "(normalised mutual information)", INK, 20),
            (f"TOF → the 4D-flow angiogram: {meta['rotation_deg']:.1f}° and {tmm:.0f} mm", INK_SOFT, 18),
            ("one matrix carries the labels across", INK_SOFT, 18),
        ])
        self.swap(cap0, cap1)
        at = rigid_path(W, tw.mean(0))

        def register(f: float) -> None:
            M = at(f)
            tvox.centers = ((M[:3, :3] @ tw.T).T + M[:3, 3]).astype("f4")
            tvox.dirty = True
            tof_mip.model = M @ tof_base

        self.animate_gl(register, run_time=T(4.0), rate=lambda x: float(x * x * (3 - 2 * x)))
        self.hold(T(1.5))

        # ------------------------------------------------------------ 2 · stage 3: labels + centerlines
        # (straight to the point: the labels land in the 4D-flow grid and the sinuses join them, uncommented)
        ec, eo, es = fstyle(e4d)
        vsel = venous & (e4d == 0)
        cap4 = self.caption("Stage 3 · Centerlines: the backbone", [
            ("the labels, carried into the 4D-flow grid (0.78 mm): one centerline per artery", INK, 20),
            ("the venous sinuses from the 4D-flow CD, named SSS · StrS · LTS · RTS", "#4B3F72", 18),
            ("every later step starts from these lines", INK_SOFT, 18),
        ])
        self.swap(cap1, cap4)
        self.move_camera(run_time=T(1.4), distance=dist * 0.95)

        def to_grid(f: float) -> None:
            tof_mip.opacity = 0.75 * (1 - f)
            angio_mip.opacity = 0.55 * (1 - f)

        v_rgb = np.where(vsel[:, None], rgb("#4B3F72"), ec)
        v_op = np.where(vsel, 0.9, eo)
        self.voxel_transition(tvox, None, np.zeros(len(tvox)), run_time=T(1.4), hook=to_grid)
        self.voxel_transition(fvox, v_rgb, v_op, np.where(vsel, 1.0, es), run_time=T(1.4))
        tvox.visible = tof_mip.visible = angio_mip.visible = False
        leg = legend(sorted(set(np.unique(e4d)) - {0}))
        beads_w = fw[clm > 0]
        beads = Spheres(beads_w, radius=0.62, color=rgb(CL_INK), subdiv=2)
        beads.set(op=np.zeros(len(beads_w), "f4"), rgb=colors_of(clm[clm > 0], QVTPY_COLORS) * 0.55)
        self.gl.add(beads)
        ghost_op = np.where((e4d > 0) | vsel, 0.16, 0.0)
        self.voxel_transition(fvox, v_rgb, ghost_op, run_time=T(1.2), extra=[FadeIn(leg)],
                              hook=lambda f: beads.set(op=np.full(len(beads_w), f, "f4")))
        tags = []
        for vid in (31, 32, 33, 34):
            pts = fw[clm == vid]
            if len(pts):
                tags.append(self.label_3d(QVTPY_NAMES[vid], pts[len(pts) // 2], color="#4B3F72", font_size=14))
        self.play(*[FadeIn(t) for t in tags], run_time=T(0.6))
        self.hold(T(2.2))
        self.play(*[FadeOut(t) for t in tags], run_time=T(0.5))

        # ------------------------------------------------------------ 5 · stage 4 threshold
        cap5 = self.caption("Stage 4 · 4D-flow segmentation: a local threshold", [
            (f"per vessel, a CD crop around its centerline, {meta.get('thr_algorithm') or 'otsu'} threshold", INK,
             20),
            ("the other vessels' centerlines are walls; only pieces touching its own line stay", INK_SOFT, 18),
            ("sinuses kept inside the venous mask", "#4B3F72", 18),
        ])
        self.swap(cap4, cap5)
        # grow outward from each label's own centerline voxels
        dist_cl = np.full(len(fw), 99.0)
        for lid in np.unique(seg["thr"][seg["thr"] > 0]):
            own = fw[(clm == lid)]
            m = seg["thr"] == lid
            if len(own) and m.any():
                dist_cl[m] = cKDTree(own).query(fw[m])[0]
        tc, to, ts = fstyle(seg["thr"])
        self.voxel_transition(fvox, tc, to, ts, run_time=T(3.0), delay=dist_cl, lag=0.7,
                              hook=lambda f: beads.set(op=np.full(len(beads_w), 1.0 - 0.7 * f, "f4")))
        self.hold(T(1.4))

        # ------------------------------------------------------------ 6 · region growing
        rg_new = (seg["rg"] > 0) & (seg["thr"] == 0)
        cap6 = self.caption("Stage 4 · Region growing of the ICAs and the basilar", [
            ("6-connected, into CD above 0.45 × their seed's mean and their own threshold", INK, 20),
            ("walled off from the other vessels; rolled back past a volume cap", INK_SOFT, 18),
            (f"{meta['volumes_mm3']['thr']:.0f} → {meta['volumes_mm3']['rg']:.0f} mm³", ADDED, 18),
        ])
        self.swap(cap5, cap6)
        rc, ro, rs = fstyle(seg["rg"])
        self.voxel_transition(fvox, np.where(rg_new[:, None], rgb(ADDED), rc), ro, rs, run_time=T(1.6),
                              delay=fw[:, 2], lag=0.5)
        self.voxel_transition(fvox, rc, ro, rs, run_time=T(1.0))
        prev = cap6
        if (seg["vsplit"] != seg["rg"]).any():
            vs = meta.get("vertebral_split") or {}
            capv = self.caption("Stage 4 · Vertebral split", [
                ("the basilar's lower branches become the left and right vertebrals", INK, 20),
                (json.dumps(vs)[:80], INK_SOFT, 16),
            ])
            self.swap(prev, capv)
            prev = capv
            vc, vo, vs_ = fstyle(seg["vsplit"])
            self.voxel_transition(fvox, vc, vo, vs_, run_time=T(1.4))
        self.hold(T(1.0))

        # ------------------------------------------------------------ 7 · blood flood
        cap7 = self.caption("Stage 4 · The blood flood: distal expansion", [
            ("Frangi vesselness of the CD → a GMM + hysteresis tree of tubes", INK, 20),
            ("watershed from the A1 · M1 · P1 into the tree, all at once; ICAs and basilar are walls", INK_SOFT,
             18),
            ("what they flood beyond their own lumen becomes their A2 · M2 · P2", ADDED, 18),
        ])
        self.swap(prev, cap7)
        pc, po, ps = fstyle(seg["preflood"])
        t_rgb = np.where(tree[:, None], rgb(GHOST), pc)
        t_op = np.where(tree, 0.25, po)
        self.voxel_transition(fvox, t_rgb, t_op, np.where(tree, 0.85, ps), run_time=T(1.6), delay=-fw[:, 2],
                              lag=0.5)
        self.hold(T(0.6))
        flc, flo, fls = fstyle(seg["flood"])
        claimed = (seg["flood"] > 0) & (seg["preflood"] == 0)
        w_rgb = np.where(claimed[:, None], flc, np.where(tree[:, None], rgb(GHOST), pc))
        w_op = np.where(claimed, 1.0, np.where(tree, 0.0, po))
        ordr = np.where(claimed & (order >= 0), order, np.where(claimed, order.max() + 1, -1.0))
        self.voxel_transition(fvox, w_rgb, w_op, np.where(claimed, 1.0, ps), run_time=T(6.0), delay=ordr, lag=0.9,
                              glow="#FFD25A", glow_strength=0.7)
        leg2 = legend(sorted(set(np.unique(seg["final"])) - {0}))
        self.play(FadeOut(leg), FadeIn(leg2), run_time=T(0.6))
        self.hold(T(1.6))

        # ------------------------------------------------------------ 8 · siphon
        cut = (seg["flood"] > 0) & (seg["final"] == 0)
        capS = self.caption("Stage 4 · Siphon correction", [
            ("as on TOF: where an ICA siphon's limbs touch and close a loop,", INK, 20),
            (f"only the contact is cut ({int(cut.sum())} voxels here)", CUT, 18),
        ])
        self.swap(cap7, capS)
        fc, fo, fs = fstyle(seg["final"])
        if cut.any():
            self.voxel_transition(fvox, np.where(cut[:, None], rgb(CUT), flc), flo, np.where(cut, 1.5, fls),
                                  run_time=T(1.0))
            self.hold(T(1.0))
        self.voxel_transition(fvox, fc, fo, fs, run_time=T(1.0),
                              hook=lambda f: beads.set(op=np.full(len(beads_w), 0.3 * (1 - f), "f4")))
        self.gl.remove(beads)
        self.hold(T(1.2))

        # ------------------------------------------------------------ 9 · stage 5 LOCs
        br = np.asarray(data["branches"], float)
        bl = [(fa[:3, :3] @ br[br[:, 3] == k][:, :3].T).T + fa[:3, 3] for k in np.unique(br[:, 3])]
        blab = [meta["branches"][int(k)]["label"] for k in np.unique(br[:, 3])]
        lines = Tubes(bl, radius=0.32, colors=[colors_of(np.array([b]), QVTPY_COLORS)[0] * 0.5 for b in blab])
        lines.progress = 0.0
        self.gl.add(lines)
        cap9 = self.caption("Stage 5 · LOCs: where the flow is measured", [
            ("centerlines of the 4D-flow segmentation, one station per vessel:", INK, 20),
            ("ICAs + basilar on a shared axial plane · A1 / M1 / P1 at their middle · A2 / M2 / P2 on the trunk",
             INK_SOFT, 16),
            ("a plane normal to the centerline through each station", LOC_C, 18),
        ])
        self.swap(capS, cap9)

        def glass(f: float) -> None:
            fvox.set(op=(fo * (1 - 0.7 * f)).astype("f4"))
            lines.progress = f

        self.animate_gl(glass, run_time=T(2.4))
        locs = np.asarray(data["locs"], float)
        tex = disc()
        quads, ltags = [], []
        for k, row in enumerate(locs):
            p = fa[:3, :3] @ row[1:4] + fa[:3, 3]
            t = fa[:3, :3] @ row[4:7]
            t = t / max(np.linalg.norm(t), 1e-9)
            a = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(t, a)
            u /= np.linalg.norm(u)
            v = np.cross(t, u)
            R = 10.0 * float(sp4[0]) * 0.6
            q = Quad(np.array([p - R * u - R * v, p + R * u - R * v, p + R * u + R * v, p - R * u + R * v]), tex,
                     opacity=0.0)
            quads.append(q)
            name = meta["loc_names"][k].replace("_", " ")
            ltags.append(self.label_3d(name, p, color="#9A4A12", font_size=12, offset=(0.0, 0.22)))
        self.gl.add(*quads)
        delays = np.linspace(0, 0.6, len(quads))

        def planes_in(f: float) -> None:
            for q, d0 in zip(quads, delays):
                q.opacity = float(np.clip((f - d0) / 0.4, 0, 1))

        self.animate_gl(planes_in, run_time=T(2.4))
        art = locs[:, 0] < 30
        c_art = fa[:3, :3] @ locs[art, 1:4].mean(0) + fa[:3, 3]
        self.orbit(ORBIT * 0.4)
        self.move_camera(run_time=T(2.0), target=c_art, distance=dist * 0.62, phi=58, shift=(0.0, 0.1),
                         extra=[FadeOut(leg2)])
        self.play(*[FadeIn(t) for t in ltags], run_time=T(0.8))
        self.hold(T(4.0))

        end = self.caption(f"{len(quads)} stations, ready to measure", [
            ("stage 6 reads the velocity through each plane, frame by frame:", INK, 20),
            ("flow, area, pulsatility, transmission and wave speed", INK_SOFT, 18),
        ])
        self.swap(cap9, end)
        self.play(*[FadeOut(t) for t in ltags], run_time=T(0.6))
        self.orbit(ORBIT)
        self.move_camera(run_time=T(3.0), target=centre, distance=dist * 1.05, phi=62, shift=SHIFT)
        self.hold(T(2.0))
        self.play(FadeOut(end), run_time=T(0.8))
        self.hold(T(0.4))


if __name__ == "__main__":
    run_scene(__file__, "QvtpyPipeline", "qvtpy_pipeline")
