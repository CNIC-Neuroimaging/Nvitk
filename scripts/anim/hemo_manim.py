"""Manim animation of the 4D-flow hemodynamics (qvtpy stage 6), on the example subject.

What stage 6 measures, in general: the velocity vector of every voxel of the 4D-flow
segmentation through the cardiac cycle; the blood's paths (particles advected through that field,
a visualisation aid); the flow through a measuring plane (LOC) frame by frame: area, mean velocity,
flow Q(t), pulsatility and resistivity indices; every vessel's flow; and along each arterial tree,
from its dense stations: how pulsatility is transmitted (PITC) and how fast the pulse wave travels
(PWV).

Every number comes from ``hemo_stages.npz`` (``hemo_precompute.py``, which re-runs stage 6's LOC
measurements and dense profile and reproduces the shipped tables); this file needs numpy, moderngl
and manim only. Everything 3D is drawn on the GPU (:mod:`gl_scene`): ~16,000 velocity arrows, 5,000
particles with their trails, 18 live cross-section maps.

Render (manim environment)::

    HE_FAST=1 python scripts/anim/hemo_manim.py -ql --fps 15
    python scripts/anim/hemo_manim.py -qh --fps 30              # -> flow_hemodynamics.mp4
    python scripts/anim/hemo_manim.py -qh --fps 30 --no-text
"""

from __future__ import annotations

import json
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
from pathlib import Path

import numpy as np
from manim import DOWN, LEFT, RIGHT, UL, UP, UR, Create, Dot, FadeIn, FadeOut, Line, Rectangle, Text, VGroup, VMobject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anim_voxels import FONT, INK, INK_FAINT, INK_SOFT, hex_of, rgb, run_scene  # noqa: E402
from gl_render import Arrows, Quad, Spheres, Voxels  # noqa: E402
from gl_scene import GLScene, make_axes  # noqa: E402
from vessel_palette import GHOST, QVTPY_COLORS, colors_of, sequential, speed_ramp  # noqa: E402

DATA_PATH = Path(os.environ.get("HE_STAGES", str(Path(__file__).with_name("hemo_stages.npz"))))
FAST = os.environ.get("HE_FAST", "0") == "1"
ORBIT = 2.6
SHIFT = (0.0, 0.14)
CYCLE_S = 4.0  # seconds of video per cardiac cycle (slow motion ~4x)
V_MAX = 500.0  # mm/s at the top of the speed colour ramp
PI_STOPS = ("#F3E9D2", "#E8B04B", "#D0612A", "#7A1631")
SPEED_STOPS = ("#1D3A8A", "#2E86C1", "#3FC1C9", "#F2D43D", "#F08A24", "#C8222C")


class FlowHemodynamics(GLScene):
    """Stage-6 hemodynamics, in general."""

    def construct(self) -> None:
        data = np.load(DATA_PATH)
        meta = json.loads(str(data["meta"]))
        T = (lambda x: max(0.3, x * 0.35)) if FAST else (lambda x: x)
        aff = np.asarray(data["affine"], float)
        sp = np.asarray(meta["spacing"], float)
        D = aff[:3, :3] / sp  # index-axis direction -> world direction
        nt = int(meta["nt"])
        period = float(meta["period_s"])

        idx = np.asarray(data["seg_vox"], float)
        world = (aff[:3, :3] @ idx.T).T + aff[:3, 3]
        lab = np.asarray(data["seg_lab"], int)
        vel = np.asarray(data["vel"], np.float32) @ D.T.astype(np.float32)  # (n, nt, 3) world mm/s
        art = lab < 30

        def to_world(p: np.ndarray) -> np.ndarray:
            return (aff[:3, :3] @ np.atleast_2d(p).T).T + aff[:3, 3]

        # ---- layers
        vox = Voxels(world, pitch=float(sp[0]), fill=0.92, edge_dark=0.1)
        arrows = Arrows(world, np.zeros_like(world), width=0.2)
        arrows.set(op=np.zeros(len(world), "f4"))
        self.gl.add(vox, arrows)

        centre = 0.5 * (world[art].max(0) + world[art].min(0))
        extent = float(np.ptp(world, axis=0)[:2].max())
        dist = extent / (2 * np.tan(np.radians(12))) * 1.0
        self.set_camera(target=centre, distance=dist * 1.05, phi=72, theta=96, fov=24, shift=SHIFT)
        self.orbit(ORBIT)

        # ---- the cardiac clock (bottom right): the left ICA's flow through the cycle
        names = meta["loc_names"]
        flows = np.asarray(data["loc_flow"], float) * 60.0  # mL/min
        k_ica = names.index("LICA") if "LICA" in names else 0
        q_ica = flows[k_ica]
        clock_ax = make_axes(1.0, float(np.ceil(q_ica.max() / 100) * 100), xstep=0.25,
                             ystep=float(np.ceil(q_ica.max() / 100) * 50), x_length=3.2, y_length=1.3)
        tt = np.linspace(0, 1, 200)
        qq = np.interp(tt * nt, np.arange(nt + 1), np.r_[q_ica, q_ica[0]])
        wave = VMobject(stroke_color="#C8222C", stroke_width=2.4)
        wave.set_points_as_corners([clock_ax.c2p(a, b) for a, b in zip(tt, qq)])
        clock_t = Text("cardiac cycle · LICA flow (mL/min)", font=FONT, color=INK, font_size=13)
        clock_t.next_to(clock_ax, UP, buff=0.1).align_to(clock_ax, LEFT)
        clock_dot = Dot(clock_ax.c2p(0, qq[0]), radius=0.06, color="#C8222C")
        clock = VGroup(clock_ax, wave, clock_t)
        clock.to_corner(UL, buff=0.45)
        clock_dot.move_to(clock_ax.c2p(0, qq[0]))
        clock_bg = Rectangle(width=clock.width + 0.35, height=clock.height + 0.3, fill_color="#FFFFFF",
                             fill_opacity=0.88, stroke_width=0).move_to(clock)
        clock_all = VGroup(clock_bg, clock, clock_dot)
        self.fixed(clock_all, keep=True)

        state = {"phase": 0.0, "arrows": False, "particles": None, "planes": [], "clock": False}

        def frame_of(phase: float) -> tuple[int, int, float]:
            x = (phase % 1.0) * nt
            f0 = int(np.floor(x)) % nt
            return f0, (f0 + 1) % nt, float(x - np.floor(x))

        def tick(dt: float) -> None:
            state["phase"] += dt / CYCLE_S
            f0, f1, w = frame_of(state["phase"])
            if state["arrows"]:
                v = vel[:, f0] * (1 - w) + vel[:, f1] * w
                s = np.linalg.norm(v, axis=1)
                L = np.clip(s / V_MAX, 0.0, 1.0) * 3.4 + 0.25
                vec = v / np.maximum(s, 1e-6)[:, None] * L[:, None]
                arrows.set(vectors=vec, rgb=speed_ramp(s / V_MAX))
            for pl in state["planes"]:
                pl.update(f0, f1, w)
            if state["particles"] is not None:
                state["particles"](state["phase"])
            if state["clock"]:
                ph = state["phase"] % 1.0
                clock_dot.move_to(clock_ax.c2p(ph, float(np.interp(ph * nt, np.arange(nt + 1), np.r_[q_ica, q_ica[0]]))))

        self.add_frame_hook(tick)

        # ------------------------------------------------------------ 0 · title
        card = self.title_card("Blood flow in the brain: 4D-flow hemodynamics",
                               "qvtpy stage 6  ·  velocity in 3D, through the cardiac cycle",
                               "the 4D-flow segmentation of stage 4, 0.78 mm voxels")
        seg_rgb = colors_of(lab, QVTPY_COLORS)
        self.voxel_transition(vox, seg_rgb, np.full(len(vox), 1.0), run_time=T(2.4), delay=world[:, 2], lag=0.6,
                              extra=[FadeIn(card)])
        self.hold(T(1.6))

        # ------------------------------------------------------------ 1 · velocity field
        cap1 = self.caption("4D flow: a velocity vector in every voxel", [
            ("three phase-contrast encodings (RL · AP · FH) → (vx, vy, vz)", INK, 20),
            (f"{nt} frames across one heartbeat ({period * 1000:.0f} ms), here ~{CYCLE_S / period:.0f}× slower",
             INK_SOFT, 18),
            ("arrow length and colour: speed", INK_SOFT, 18),
        ])
        self.swap(card, cap1)
        bar = self._colorbar(SPEED_STOPS, "speed (mm/s)", 0, int(V_MAX))
        state["arrows"] = True
        state["clock"] = True

        def show_arrows(f: float) -> None:
            vox.set(op=np.full(len(vox), 1.0 - 0.88 * f, "f4"))
            arrows.set(op=np.full(len(world), f, "f4"))

        self.animate_gl(show_arrows, run_time=T(1.6), extra=[FadeIn(bar), FadeIn(clock_all)])
        self.hold(T(4.0))
        self.move_camera(run_time=T(3.0), distance=dist * 0.55, target=centre + np.array([0.0, 0.0, -6.0]))
        self.hold(T(3.0))

        # ------------------------------------------------------------ 2 · particles
        cap2 = self.caption("Following the blood", [
            ("particles carried by the measured velocity, frame after frame", INK, 20),
            ("born at the carotids and basilar, and throughout the vessels", INK_SOFT, 18),
            ("a picture of the field (stage 6 measures planes, not particles)", INK_FAINT, 16),
        ])
        self.swap(cap1, cap2)
        P = np.asarray(data["p_pos"], np.float32)
        PS = np.asarray(data["p_speed"], np.float32)
        PA = np.asarray(data["p_age"], np.int32)
        nfr, npart = P.shape[:2]
        fpc = int(meta["frames_per_cycle"])
        tail = (0, 2, 4, 7, 10, 14)
        sizes = np.array([0.34, 0.3, 0.26, 0.22, 0.18, 0.14], "f4")
        alphas = np.array([1.0, 0.85, 0.7, 0.5, 0.35, 0.2], "f4")
        parts = Spheres(np.zeros((npart * len(tail), 3)), radius=0.3, subdiv=1)
        parts.set(op=np.zeros(npart * len(tail), "f4"))
        self.gl.add(parts)
        p0 = {"phase": None, "fade": 0.0}

        def draw_particles(phase: float) -> None:
            if p0["phase"] is None:
                p0["phase"] = phase
            k = int((phase - p0["phase"]) * fpc) % nfr
            cs, rs, os_ = [], [], []
            for j, off in enumerate(tail):
                kk = max(k - off, 0)
                cs.append(P[kk])
                alive = (PA[k] >= off) & (k - off >= 0)
                fade_in = np.clip(PA[k] / 6.0, 0, 1)
                rs.append(np.where(alive, sizes[j], 0.0))
                os_.append(np.where(alive, alphas[j] * fade_in * p0["fade"], 0.0))
            col = speed_ramp(np.tile(PS[k], len(tail)) / V_MAX)
            parts.set(centers=np.concatenate(cs), radius=np.concatenate(rs).astype("f4"), rgb=col,
                      op=np.concatenate(os_).astype("f4"))

        state["particles"] = draw_particles

        def to_particles(f: float) -> None:
            p0["fade"] = f
            arrows.set(op=np.full(len(world), 1.0 - f, "f4"))
            vox.set(op=np.full(len(vox), 0.12 + 0.03 * f, "f4"))

        self.animate_gl(to_particles, run_time=T(1.6))
        state["arrows"] = False
        self.hold(T(4.5))
        self.move_camera(run_time=T(3.0), distance=dist * 0.9, target=centre, phi=66)
        self.hold(T(3.5))

        # ------------------------------------------------------------ 3 · one LOC, measured
        locs = np.asarray(data["loc_rows"], float)
        planes_v = np.asarray(data["loc_planes"], np.float32)
        masks = np.asarray(data["loc_masks"], bool)
        scal = np.asarray(data["loc_scalar"], float)
        R = float(meta["plane_radius_vox"])

        n_pl = planes_v.shape[-1]

        class Plane:
            """A LOC's measuring plane: the through-plane speed as a texture (the lumen opaque, the
            rest of the plane faint) and a bundle of arrows through the lumen, both following the
            cardiac frame. The texture spans exactly the sampled square (±R voxels)."""

            def __init__(self, k: int):
                c, t = locs[k, 1:4], locs[k, 4:7]
                t = t / np.linalg.norm(t)
                ref = np.array([0.0, 0.0, 1.0]) if abs(t[2]) <= 0.95 else np.array([0.0, 1.0, 0.0])
                u = np.cross(t, ref)
                u /= np.linalg.norm(u)
                v = np.cross(t, u)
                corners = np.array([c - R * u - R * v, c + R * u - R * v, c + R * u + R * v, c - R * u + R * v])
                self.k = k
                self.vmax = max(float(np.abs(planes_v[k][:, masks[k]]).max(initial=1.0)), 50.0)
                self.quad = Quad(to_world(corners), self.texture(0, 0, 0.0), opacity=0.0)
                g = np.linspace(-R, R, n_pl)
                ii, jj = np.nonzero(masks[k])
                keep = (ii % 3 == 1) & (jj % 3 == 1)
                self.ij = (ii[keep], jj[keep])
                pts = c + g[jj[keep], None] * u + g[ii[keep], None] * v
                self.tw = D @ t
                self.tw /= np.linalg.norm(self.tw)
                self.arrows = Arrows(to_world(pts) if len(pts) else np.zeros((0, 3)), np.zeros((len(pts), 3)),
                                     width=0.14)
                self.arrows.set(op=np.zeros(len(pts), "f4"))
                self.arrow_op = 0.0

            def texture(self, f0: int, f1: int, w: float) -> np.ndarray:
                v = np.abs(planes_v[self.k, f0] * (1 - w) + planes_v[self.k, f1] * w)
                m = masks[self.k]
                rgb_ = speed_ramp((v / self.vmax).ravel()).reshape(v.shape + (3,))
                a = np.where(m, 1.0, 0.4)
                rgb_ = np.where(m[..., None], rgb_, rgb("#B9C0CA"))
                edge = m ^ np.pad(m, 1)[1:-1, 2:] | m ^ np.pad(m, 1)[2:, 1:-1]
                rgb_ = np.where(edge[..., None], rgb(INK), rgb_)
                out = np.concatenate([rgb_, a[..., None]], -1)
                return (np.clip(out, 0, 1) * 255).astype(np.uint8)

            def update(self, f0: int, f1: int, w: float) -> None:
                if self.quad.opacity > 0.004:
                    self.quad.set_image(self.texture(f0, f1, w))
                if len(self.ij[0]) and self.arrow_op > 0.004:
                    vt = planes_v[self.k, f0][self.ij] * (1 - w) + planes_v[self.k, f1][self.ij] * w
                    s_ = np.abs(vt)
                    L = 1.0 + 5.0 * np.clip(s_ / V_MAX, 0, 1.2)
                    self.arrows.set(vectors=np.sign(np.median(vt)) * self.tw[None, :] * L[:, None],
                                    rgb=speed_ramp(s_ / V_MAX), op=np.full(len(s_), self.arrow_op, "f4"))

        k0 = k_ica
        main = Plane(k0)
        self.gl.add(main.quad, main.arrows)
        state["planes"].append(main)
        a_mm2, v_mean, q_mean, pi, ri = scal[k0]
        cap3 = self.caption("Measuring at a LOC", [
            ("the plane normal to the centreline; the vessel's own label inside it", INK, 20),
            ("v(t): mean through-plane velocity  ·  Q(t) = v(t) · area", INK_SOFT, 18),
            (f"LICA: {a_mm2:.1f} mm² · mean {v_mean:.0f} mm/s · Q {q_mean * 60:.0f} mL/min · PI {pi:.2f} · "
             f"RI {ri:.2f}", "#C8222C", 18),
        ])
        self.swap(cap2, cap3)
        c_loc = to_world(locs[k0, 1:4])[0]
        t_loc = D @ locs[k0, 4:7]
        t_loc /= np.linalg.norm(t_loc)
        if t_loc[2] < 0:
            t_loc = -t_loc  # look down the vessel from above, onto the plane's face
        side = np.cross(t_loc, [0.0, 0.0, 1.0])
        side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-6 else np.array([0.0, 1.0, 0.0])
        look = t_loc + 0.75 * side + np.array([0.0, 0.5, 0.0])
        look /= np.linalg.norm(look)
        phi_l = float(np.degrees(np.arccos(np.clip(look[2], -1, 1))))
        th_l = float(np.degrees(np.arctan2(look[1], look[0])))
        th_l = self.cam.theta + ((th_l - self.cam.theta + 180.0) % 360.0 - 180.0)
        self.orbit(0.0)

        own = lab == lab[np.argmin(np.linalg.norm(world - c_loc, axis=1))]

        def plane_in(f: float) -> None:
            main.quad.opacity = f
            main.arrow_op = f
            p0["fade"] = 1.0 - f
            vox.set(op=np.where(own, 0.15 - 0.05 * f, 0.15 - 0.11 * f).astype("f4"))

        self.move_camera(run_time=T(2.4), target=c_loc, distance=dist * 0.2, phi=phi_l, theta=th_l,
                         shift=(-0.22, 0.02))
        self.animate_gl(plane_in, run_time=T(1.0))
        # Q(t) plot (top right)
        qmax = float(np.ceil(q_ica.max() / 100) * 100)
        ax = make_axes(1.0, qmax, xstep=0.25, ystep=qmax / 2, x_length=4.6, y_length=2.0)
        qt = Text("Q(t) through the plane (mL/min)", font=FONT, weight="BOLD", color=INK, font_size=15)
        qt.next_to(ax, UP, buff=0.12).align_to(ax, LEFT)
        xl = Text("fraction of the cardiac cycle", font=FONT, color=INK_SOFT, font_size=12).next_to(ax, DOWN,
                                                                                                    buff=0.32)
        pan = VGroup(ax, qt, xl).to_corner(UR, buff=0.45)
        bgp = Rectangle(width=pan.width + 0.4, height=pan.height + 0.3, fill_color="#FFFFFF", fill_opacity=0.9,
                        stroke_width=0).move_to(pan)
        frames_x = np.arange(nt) / nt
        pts = VGroup(*[Dot(ax.c2p(x, y), radius=0.045, color="#C8222C") for x, y in zip(frames_x, q_ica)])
        curve = VMobject(stroke_color="#C8222C", stroke_width=2.4)
        curve.set_points_as_corners([ax.c2p(a, b) for a, b in zip(tt, qq)])
        mean_l = Line(ax.c2p(0, q_mean * 60), ax.c2p(1, q_mean * 60), stroke_color=INK_SOFT, stroke_width=1.5)
        mean_t = Text(f"mean {q_mean * 60:.0f}", font=FONT, color=INK_SOFT, font_size=12).next_to(mean_l, RIGHT,
                                                                                                    buff=0.05)
        board = VGroup(bgp, pan)
        self.fixed(board, pts, curve, mean_l, mean_t, keep=True, drop=(qt,))
        self.play(FadeIn(board), FadeOut(bar), run_time=T(0.6))
        self.play(*[FadeIn(p, scale=2) for p in pts], run_time=T(1.4))
        self.play(Create(curve), FadeIn(mean_l), FadeIn(mean_t), run_time=T(1.2))
        pi_txt = VGroup(Text(f"PI = (Qmax − Qmin) / Qmean = {pi:.2f}", font=FONT, color=INK, font_size=13),
                        Text(f"RI = (Qmax − Qmin) / Qmax = {ri:.2f}", font=FONT, color=INK, font_size=13)).arrange(
            DOWN, buff=0.08, aligned_edge=LEFT)
        pi_txt.next_to(pan, DOWN, buff=0.15).align_to(pan, LEFT)
        pi_txt.add_background_rectangle(color="#FFFFFF", opacity=0.9, buff=0.08)
        self.fixed(pi_txt)
        self.play(FadeIn(pi_txt), run_time=T(0.6))
        self.hold(T(4.0))

        # ------------------------------------------------------------ 4 · every LOC
        cap4 = self.caption("Every vessel, its own station", [
            ("18 LOCs: carotids, basilar, A1 / M1 / P1, A2 / M2 / P2, the sinuses", INK, 20),
            ("mean flow per vessel (mL/min)", INK_SOFT, 18),
        ])
        self.swap(cap3, cap4)
        others = [Plane(k) for k in range(len(locs)) if k != k0]
        self.gl.add(*[p.quad for p in others], *[p.arrows for p in others])
        state["planes"].extend(others)
        self.play(FadeOut(board), FadeOut(pts), FadeOut(curve), FadeOut(mean_l), FadeOut(mean_t), FadeOut(pi_txt),
                  run_time=T(0.5))
        self.move_camera(run_time=T(2.6), target=centre, distance=dist * 0.95, phi=64, theta=self.cam.theta - 60,
                         shift=(-0.2, 0.12))
        self.orbit(ORBIT * 0.6)

        def planes_in(f: float) -> None:
            for p in others:
                p.quad.opacity = f
                p.arrow_op = f
            vox.set(op=np.full(len(vox), 0.1 + 0.05 * f, "f4"))

        self.animate_gl(planes_in, run_time=T(1.4))
        state["particles"] = None
        self.gl.remove(parts)
        # bar chart of mean flow
        qm = scal[:, 2] * 60.0
        order = np.argsort(-qm)
        rows = VGroup()
        qtop = float(qm.max())
        for k in order:
            nm = names[k].replace("_", " ")
            vid = int(locs[k, 0])
            col = QVTPY_COLORS.get(vid, INK_SOFT)
            label = Text(nm, font=FONT, color=INK, font_size=12)
            barr = Rectangle(width=max(0.03, 2.6 * qm[k] / qtop), height=0.13, fill_color=col, fill_opacity=1,
                             stroke_width=0)
            val = Text(f"{qm[k]:.0f}", font=FONT, color=INK_SOFT, font_size=11)
            rows.add(VGroup(label, barr, val))
        lw = max(r[0].width for r in rows)
        for i, r in enumerate(rows):
            y = -i * 0.2
            r[0].move_to([lw - r[0].width / 2, y, 0])  # names right-aligned
            r[1].move_to([lw + 0.12 + r[1].width / 2, y, 0])
            r[2].next_to(r[1], RIGHT, buff=0.08)
        tot = sum(qm[names.index(n)] for n in ("LICA", "RICA", "BASILAR") if n in names)
        head = Text(f"mean flow (mL/min) · ICAs + basilar = {tot:.0f}", font=FONT, weight="BOLD", color=INK,
                    font_size=13).next_to(rows, UP, buff=0.15).align_to(rows, LEFT)
        chart = VGroup(head, rows).to_corner(UR, buff=0.4)
        chart.add_background_rectangle(color="#FFFFFF", opacity=0.9, buff=0.14)
        self.fixed(chart, keep=True)
        self.play(FadeIn(chart, shift=LEFT * 0.2), run_time=T(1.0))
        self.hold(T(4.5))

        # ------------------------------------------------------------ 5 · PITC
        st = np.asarray(data["stations"], float)
        regions = meta["regions"]
        rid = "L_ICA"
        reg = regions[rid]
        thr = float(reg["quality_thresh"])
        sw = to_world(st[:, 2:5])
        mine = st[:, 0] == reg["index"]
        good = st[:, 7] > thr
        pis = st[:, 6]
        cap5 = self.caption("Along the tree: pulsatility transmission (PITC)", [
            (f"{len(st)} stations: a cross-section every centreline point of every named branch", INK, 20),
            (f"quality 0-4 from the neighbours' flow, area, roundness, agreement; ≤ {thr:g}: left out (grey)",
             INK_SOFT, 16),
            ("PI(d) = slope · d + intercept, weighted by quality", "#7A1631", 18),
        ])
        self.swap(cap4, cap5)
        self.play(FadeOut(chart), run_time=T(0.4))

        def planes_out(f: float) -> None:
            for p in state["planes"]:
                p.quad.opacity = 1.0 - f
                p.arrow_op = 1.0 - f
                p.arrows.set(op=np.full(len(p.ij[0]), 1.0 - f, "f4"))
            vox.set(op=np.where(art, 0.15 - 0.05 * f, 0.15 - 0.11 * f).astype("f4"))

        self.animate_gl(planes_out, run_time=T(1.0))
        state["planes"] = []
        pimax = 1.6
        dots = Spheres(sw, radius=np.where(good, 0.55, 0.32), subdiv=2)
        dots.set(rgb=np.where(good[:, None], sequential(pis / pimax, PI_STOPS), rgb(GHOST)),
                 op=np.zeros(len(sw), "f4"))
        self.gl.add(dots)
        pbar = self._colorbar(PI_STOPS, "pulsatility index", 0, pimax)
        dist_all = st[:, 5]
        self.animate_gl(lambda f: dots.set(op=np.clip((f * 1.3 - dist_all / max(dist_all.max(), 1)) * 4, 0, 1)
                                           .astype("f4")), run_time=T(2.4), extra=[FadeIn(pbar)])
        # PITC plot (top right) for the left ICA tree
        dmax = float(np.ceil(st[mine, 5].max() / 50) * 50)
        ax2 = make_axes(dmax, pimax, xstep=50, ystep=0.4, x_length=4.6, y_length=2.0)
        t2 = Text(f"{rid.replace('_', ' ')} tree: PI vs distance from the root", font=FONT, weight="BOLD", color=INK,
                  font_size=14).next_to(ax2, UP, buff=0.12).align_to(ax2, LEFT)
        x2 = Text("distance (mm)", font=FONT, color=INK_SOFT, font_size=12).next_to(ax2, DOWN, buff=0.32)
        pan2 = VGroup(ax2, t2, x2).to_corner(UR, buff=0.45)
        bg2 = Rectangle(width=pan2.width + 0.4, height=pan2.height + 0.3, fill_color="#FFFFFF", fill_opacity=0.9,
                        stroke_width=0).move_to(pan2)
        sc = VGroup(*[Dot(ax2.c2p(d_, min(p_, pimax)), radius=0.025,
                          color=hex_of(sequential(np.array([p_ / pimax]), PI_STOPS)[0]) if g_ else INK_FAINT)
                      for d_, p_, g_ in zip(st[mine, 5], pis[mine], good[mine])])
        fit = Line(ax2.c2p(0, reg["pitc_intercept"]), ax2.c2p(dmax, reg["pitc_intercept"] + reg["pitc_slope"] * dmax),
                   stroke_color="#7A1631", stroke_width=2.6)
        ft = Text(f"slope {reg['pitc_slope']:+.4f} /mm  ·  intercept {reg['pitc_intercept']:.2f}  ·  "
                  f"n {reg['pitc_n']}", font=FONT, color="#7A1631", font_size=12).next_to(pan2, DOWN, buff=0.12)
        ft.align_to(pan2, LEFT)
        board2 = VGroup(bg2, pan2)
        self.fixed(board2, sc, fit, keep=True, drop=(t2,))
        self.fixed(ft)
        self.play(FadeOut(pbar), FadeIn(board2), run_time=T(0.6))
        self.play(FadeIn(sc, lag_ratio=0.002), run_time=T(1.6))
        self.play(Create(fit), FadeIn(ft), run_time=T(1.0))
        self.hold(T(3.6))

        # ------------------------------------------------------------ 6 · PWV
        pwv_f = float(reg["pwv_fielding_m_s"]) if reg["pwv_fielding_m_s"] not in ("", None) else float("nan")
        slope_x, icpt_x = float(reg["pwv_xcor_slope_s_per_mm"]), float(reg["pwv_xcor_intercept_s"])
        pp = np.asarray(data[f"pwv_pts_{rid}"], float)  # distance mm, delay s, area weight, 1 fit / 0 outlier / -1 Q
        cap6 = self.caption("The pulse wave (PWV)", [
            ("each heartbeat sends a pressure wave down the tree: the waveform arrives later downstream", INK, 18),
            (f"every station's delay vs the root by cross-correlation · grey: quality ≤ {thr:g} · open: outliers",
             INK_SOFT, 16),
            (f"delay = d / PWV, area-weighted (dot size)  ·  {rid.replace('_', ' ')}: PWV {pwv_f:.1f} m/s",
             "#1D3A8A", 18),
        ])
        self.swap(cap5, cap6)
        self.play(FadeOut(board2), FadeOut(sc), FadeOut(fit), FadeOut(ft), run_time=T(0.5))

        # PWV plot (top right): delay (ms) vs distance, the fit over the filtered points
        ylo, yhi = -50.0, 150.0
        ax3 = make_axes(dmax, yhi, xstep=50, ystep=50, x_length=4.6, y_length=2.0, ymin=ylo)
        t3 = Text(f"{rid.replace('_', ' ')} tree: pulse delay vs distance from the root", font=FONT, weight="BOLD",
                  color=INK, font_size=14).next_to(ax3, UP, buff=0.12).align_to(ax3, LEFT)
        x3 = Text("distance (mm)", font=FONT, color=INK_SOFT, font_size=12).next_to(ax3, DOWN, buff=0.32)
        y3 = Text("delay (ms)", font=FONT, color=INK_SOFT, font_size=12).rotate(np.pi / 2)
        y3.next_to(ax3, LEFT, buff=0.42)
        pan3 = VGroup(ax3, t3, x3, y3).to_corner(UR, buff=0.45)
        bg3 = Rectangle(width=pan3.width + 0.4, height=pan3.height + 0.3, fill_color="#FFFFFF", fill_opacity=0.9,
                        stroke_width=0).move_to(pan3)

        def at(d_: float, t_s: float):
            return ax3.c2p(min(max(d_, 0.0), dmax), min(max(t_s * 1000.0, ylo), yhi))

        ok = np.isfinite(pp[:, 0]) & np.isfinite(pp[:, 1])
        kept, outl, qex = (ok & (pp[:, 3] == c) for c in (1, 0, -1))
        qex &= (pp[:, 1] * 1000.0 >= ylo) & (pp[:, 1] * 1000.0 <= yhi)  # off-scale greys would pile on the edges
        wmax = max(float(pp[kept, 2].max()), 1e-9) if kept.any() else 1.0
        grey = VGroup(*[Dot(at(d_, t_), radius=0.02, color=INK_FAINT, fill_opacity=0.7) for d_, t_ in pp[qex, :2]])
        rings = VGroup(*[Dot(at(d_, t_), radius=0.032, fill_opacity=0.0, stroke_width=1.4, stroke_color="#C0392B")
                         for d_, t_ in pp[outl, :2]])
        blue = VGroup(*[Dot(at(d_, t_), radius=0.014 + 0.03 * np.sqrt(w_ / wmax), color="#2E86C1", fill_opacity=0.8)
                        for d_, t_, w_ in pp[kept, :3]])
        fit3 = Line(at(0, icpt_x), at(dmax, icpt_x + slope_x * dmax), stroke_color="#1D3A8A", stroke_width=2.6)
        ft3 = Text(f"slope {slope_x * 1000:.3f} ms/mm  ·  PWV {pwv_f:.2f} m/s  ·  n {int(kept.sum())}/{int(ok.sum())}",
                   font=FONT, color="#1D3A8A", font_size=12)
        ft3.next_to(pan3, DOWN, buff=0.12).align_to(pan3, LEFT)
        board3 = VGroup(bg3, pan3)
        cursor = Line(ax3.c2p(0, 0), ax3.c2p(dmax, 0), stroke_color="#2E86C1", stroke_width=2.0, stroke_opacity=0.0)
        self.fixed(board3, grey, rings, blue, fit3, cursor, keep=True, drop=(t3,))
        self.fixed(ft3)

        # the wave: stations light up as it passes (delay on the fitted line), slowed 200x; the plot's
        # cursor climbs the delay axis in step with it
        use = mine & good
        d_use = st[:, 5]
        delay = np.where(use, icpt_x + slope_x * d_use, np.inf)  # s
        t0 = min(icpt_x, 0.0) - 0.01
        t1 = float(np.nanmax(np.where(np.isfinite(delay), delay, 0))) * 1.15
        slow = 200.0
        base_rgb = np.where(good[:, None], sequential(pis / pimax, PI_STOPS), rgb(GHOST))
        base_r = np.where(good, 0.55, 0.32)

        def wave_pass(f: float) -> None:
            tnow = t0 + f * (t1 - t0)
            g = np.exp(-((tnow - delay) * slow / 0.6) ** 2)
            g = np.where(np.isfinite(delay), g, 0.0)
            dots.set(rgb=base_rgb + (rgb("#2E86C1") - base_rgb) * g[:, None], radius=(base_r * (1 + 0.9 * g)).astype("f4"),
                     op=np.where(mine, 1.0, 0.25).astype("f4"))
            y = min(max(tnow * 1000.0, ylo), yhi)
            cursor.put_start_and_end_on(ax3.c2p(0, y), ax3.c2p(dmax, y))
            cursor.set_stroke(opacity=0.0 if f >= 0.999 else 0.85)

        c_tree = 0.5 * (sw[mine].max(0) + sw[mine].min(0))
        self.orbit(ORBIT * 0.4)
        self.move_camera(run_time=T(1.8), target=c_tree, distance=dist * 0.62, shift=(0.0, 0.1),
                         extra=[FadeIn(board3)])
        self.play(FadeIn(grey, lag_ratio=0.002), FadeIn(rings, lag_ratio=0.01), run_time=T(1.0))
        self.play(FadeIn(blue, lag_ratio=0.002), run_time=T(1.4))
        self.play(Create(fit3), FadeIn(ft3), run_time=T(1.0))
        self.add(cursor)
        for _ in range(3):
            self.animate_gl(wave_pass, run_time=T(3.2))
        self.hold(T(1.0))
        self.play(FadeOut(board3), FadeOut(grey), FadeOut(rings), FadeOut(blue), FadeOut(fit3), FadeOut(ft3),
                  run_time=T(0.6))

        # ------------------------------------------------------------ end
        dmp = {k: v for k, v in meta["damping"].items() if v is not None and np.isfinite(v)}
        end = self.caption("What stage 6 writes", [
            ("per LOC: area · mean velocity · flow Q(t) · PI · RI", INK, 20),
            ("per tree: PITC slope, global PI, PWV  ·  per branch: damping index", INK_SOFT, 18),
            (f"loc_measurements · pitc_profile · vessel_hemodynamics  ({len(dmp)} branches)", INK_FAINT, 16),
        ])
        self.swap(cap6, end)
        self.orbit(ORBIT)
        self.move_camera(run_time=T(3.0), target=centre, distance=dist * 1.0, phi=70, shift=SHIFT,
                         extra=[FadeOut(clock_all)])
        self.hold(T(2.0))
        self.play(FadeOut(end), run_time=T(0.8))
        self.hold(T(0.4))

    def _colorbar(self, stops, label: str, lo: float, hi: float) -> VGroup:
        n = 40
        cs = sequential(np.linspace(0, 1, n), stops)
        cells = VGroup(*[Rectangle(width=0.06, height=0.18, fill_color=hex_of(c), fill_opacity=1, stroke_width=0)
                         for c in cs]).arrange(RIGHT, buff=0)
        lo_t = Text(f"{lo:g}", font=FONT, color=INK_SOFT, font_size=13).next_to(cells, LEFT, buff=0.1)
        hi_t = Text(f"{hi:g}", font=FONT, color=INK_SOFT, font_size=13).next_to(cells, RIGHT, buff=0.1)
        lab = Text(label, font=FONT, color=INK, font_size=14).next_to(cells, UP, buff=0.1)
        g = VGroup(cells, lo_t, hi_t, lab).to_corner(UR, buff=0.5)
        g.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.12)
        self.fixed(g, keep=True)
        return g


if __name__ == "__main__":
    run_scene(__file__, "FlowHemodynamics", "flow_hemodynamics")
