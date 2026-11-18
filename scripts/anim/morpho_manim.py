"""Manim animation of the TOF morphometrics (qvtpy stage 7), on the example subject.

What :func:`nvitk.measure.morphometrics.run_morphometrics_case` measures, in general: the
post-processed eICAB mask voxel by voxel, one surface per vessel, the VMTK centerlines from each
root to every terminal, the two calibre signals (the maximum inscribed sphere and the cross-section
cut by the normal plane, 36 rays, ``ray_median``), the calibre map, length / tortuosity /
curvature and the siphon mask, the healthy reference that stenoses and enlargements are measured
against, the circle of Willis variants, and the per-vessel summary.

Every number and shape comes from ``morpho_stages.npz`` (``morpho_precompute.py``, which runs the
morphometrics); this file needs numpy, moderngl and manim only. The 3D content is drawn on the GPU
(:mod:`gl_scene`).

Render (manim environment)::

    MO_FAST=1 python scripts/anim/morpho_manim.py -ql --fps 15
    python scripts/anim/morpho_manim.py -qh --fps 30              # -> tof_morphometrics.mp4
    python scripts/anim/morpho_manim.py -qh --fps 30 --no-text
"""

from __future__ import annotations

import json
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # numpy's per-frame work is small: no 32-thread spin
import sys
from pathlib import Path

import numpy as np
from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UP,
    UR,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    Line,
    Polygon,
    Rectangle,
    Text,
    VGroup,
    VMobject,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anim_voxels import FONT, INK, INK_FAINT, INK_SOFT, hex_of, rgb, run_scene  # noqa: E402
from gl_render import Lines, Mesh, Points, Quad, Spheres, Tubes, Voxels  # noqa: E402
from gl_scene import GLScene, make_axes  # noqa: E402
from vessel_palette import EICAB_COLORS, colors_of, sequential  # noqa: E402

DATA_PATH = Path(os.environ.get("MO_STAGES", str(Path(__file__).with_name("morpho_stages.npz"))))
FAST = os.environ.get("MO_FAST", "0") == "1"
ORBIT = 3.0
SHIFT = (0.0, 0.14)
MIS_C = "#E07B22"
CS_C = "#22456E"
REF_C = "#2E9E4F"
LESION_C = "#C0392B"
SIPHON_C = "#B03A9A"
RAY_C = "#E07B22"


def disc_texture(n: int = 128, color: str = "#22456E", alpha: float = 0.28) -> np.ndarray:
    """A soft-edged disc (the cut plane), RGBA."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r = np.sqrt(x * x + y * y)
    a = np.clip((1.0 - r) / 0.08, 0, 1) * alpha
    a = np.maximum(a, np.where(np.abs(r - 0.985) < 0.015, 0.9, 0.0) * (r <= 1))
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = (rgb(color) * 255).astype(np.uint8)
    out[..., 3] = (a * 255).astype(np.uint8)
    return out


def ring(origin, u, v, r, n: int = 64) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n)
    return np.asarray(origin) + r * (np.cos(t)[:, None] * np.asarray(u) + np.sin(t)[:, None] * np.asarray(v))


class TofMorphometrics(GLScene):
    """Stage-7 morphometrics, in general."""

    def construct(self) -> None:
        data = np.load(DATA_PATH)
        meta = json.loads(str(data["meta"]))
        T = (lambda x: max(0.3, x * 0.35)) if FAST else (lambda x: x)
        aff = np.asarray(meta["affine"], float)
        sp = np.asarray(meta["spacing_mm"], float)
        cols = {c: i for i, c in enumerate(meta["point_cols"])}
        P = np.asarray(data["paths"], float)
        pid = P[:, len(cols)].astype(int)
        plab = P[:, len(cols) + 1].astype(int)
        paths = [P[pid == k] for k in range(len(meta["paths"]))]
        show = int(meta["showcase"])
        S = paths[show]
        s_mm = S[:, cols["s_mm"]]
        r_cs = S[:, cols["radius_mm"]]
        r_mis = S[:, cols["maximum_inscribed_sphere_radius_mm"]]
        xyz = S[:, :3]
        summary = {int(r["label"]): r for r in meta["summary"] if r.get("label") is not None}

        idx = np.asarray(data["vox"], float)
        world = (aff[:3, :3] @ idx.T).T + aff[:3, 3]
        lab = np.asarray(data["lab"], int)
        vox = Voxels(world, pitch=float(sp[0]), fill=0.9)
        mesh = Mesh(Mesh.from_triangles(np.asarray(data["mesh_v"]), np.asarray(data["mesh_f"]),
                                        colors=colors_of(np.asarray(data["mesh_lab"]))), opacity=0.0, amb=0.45,
                    spec=0.22)
        lines_lab = Tubes([p[:, :3] for p in paths], radius=0.28,
                          colors=[rgb(EICAB_COLORS.get(int(plab[pid == k][0]), "#888888")) * 0.55
                                  for k in range(len(paths))])
        lines_lab.progress = 0.0
        self.gl.add(mesh, vox, lines_lab)

        centre = 0.5 * (world.max(0) + world.min(0))
        extent = float(np.ptp(world, axis=0)[:2].max())
        dist = extent / (2 * np.tan(np.radians(12))) * 1.05
        self.set_camera(target=centre, distance=dist * 0.9, phi=72, theta=96, fov=24, shift=SHIFT)
        self.orbit(ORBIT)

        # ------------------------------------------------------------------ 0 · voxels
        card = self.title_card("Measuring the arteries: TOF morphometrics",
                               "qvtpy stage 7  ·  nvitk.measure.morphometrics (cow_morpho port)  ·  VMTK",
                               "post-processed eICAB labels, 0.5 mm voxels")
        self.voxel_transition(vox, colors_of(lab), np.ones(len(vox)), run_time=T(2.6), delay=world[:, 2], lag=0.7,
                              extra=[FadeIn(card)])
        self.hold(T(2.0))

        # ------------------------------------------------------------------ 1 · surfaces
        # the surface as what it is: vertices (points) joined by triangle edges (wireframe)
        mv = np.asarray(data["mesh_v"], float)
        mf = np.asarray(data["mesh_f"], int)
        mcol = colors_of(np.asarray(data["mesh_lab"]))
        e = np.sort(np.concatenate([mf[:, [0, 1]], mf[:, [1, 2]], mf[:, [2, 0]]]), axis=1)
        e = np.unique(e, axis=0)
        e = e[np.argsort(mv[e].mean(1)[:, 2])]  # revealed bottom to top
        mn = np.asarray(Mesh.from_triangles(mv, mf)["v"][:, 3:6], float)  # vertex normals: the back is hidden
        zoom_d = 30.0 / (2 * np.tan(np.radians(12)))
        wire = Lines(mv[e[:, 0]], mv[e[:, 1]], colors=np.c_[mcol[e[:, 0]] * 0.55, np.full(len(e), 0.7)],
                     opacity=0.0, normals=(mn[e[:, 0]], mn[e[:, 1]]))
        vo = np.argsort(mv[:, 2])
        pts = Points(mv[vo], colors=mcol[vo] * 0.85, size_px=3.0, opacity=0.0, ref_dist=zoom_d, normals=mn[vo])
        self.gl.add(wire, pts)
        cap1 = self.caption("1 · A surface per vessel", [
            ("marching cubes on each label (its largest piece): a triangle mesh", INK, 20),
            (f"{len(mv):,} vertices joined by {len(e):,} edges: the voxel staircase becomes a wall", INK_SOFT,
             18),
        ])
        self.swap(card, cap1)

        def to_points(f: float) -> None:
            pts.opacity = min(1.0, 2.0 * f)
            pts.progress = f
            vox.set(op=np.full(len(vox), 1.0 - f, "f4"))

        self.animate_gl(to_points, run_time=T(2.2))
        vox.visible = False
        wire.opacity = 1.0
        wire.progress = 0.0
        self.animate_gl(lambda f: setattr(wire, "progress", f), run_time=T(2.6))
        # close in on the left ICA's upper bend, ~30 mm across: the triangles show
        ml = np.asarray(data["mesh_lab"])
        ica_v = mv[ml == 1] if (ml == 1).any() else mv
        zoom_t = ica_v[ica_v[:, 2] >= np.percentile(ica_v[:, 2], 75)].mean(0)
        self.orbit(ORBIT * 0.3)
        self.move_camera(run_time=T(2.6), target=zoom_t, distance=zoom_d,
                         shift=(0.0, 0.08))
        self.hold(T(2.0))
        self.move_camera(run_time=T(2.0), target=centre, distance=dist * 0.9, shift=SHIFT)
        self.orbit(ORBIT)

        # ------------------------------------------------------------------ 2 · centerlines
        n_paths = len(paths)
        cap2 = self.caption("2 · Centerlines", [
            ("VMTK: the surface's Voronoi diagram, from the root to every terminal", INK, 20),
            (f"{n_paths} root → terminal paths, resampled every 0.1 mm", INK_SOFT, 18),
        ])
        self.swap(cap1, cap2)

        def glassy(f: float) -> None:
            mesh.opacity = 0.2 * f
            mesh.rim = 1.0
            wire.opacity = 1.0 - f
            pts.opacity = 1.0 - f

        self.animate_gl(glassy, run_time=T(1.4))
        self.gl.remove(wire, pts)
        self.animate_gl(lambda f: setattr(lines_lab, "progress", f), run_time=T(3.2))
        self.hold(T(1.2))

        # ------------------------------------------------------------------ 3 · two radii (showcase)
        name = meta["paths"][show]["vessel"]
        lid = meta["paths"][show]["label"]
        sel_lab = np.asarray(data["mesh_lab"])

        cap3 = self.caption("3 · Calibre: two radii at every point", [
            ("maximum inscribed sphere: the largest ball that fits (VMTK)", MIS_C, 20),
            ("cross-section: cut by the normal plane, 36 rays to the wall, the median", CS_C, 20),
            (f"here along the {name}, {s_mm[-1]:.0f} mm from the floor to its terminal", INK_SOFT, 18),
        ])
        self.swap(cap2, cap3)
        focus = 0.5 * (xyz.max(0) + xyz.min(0))
        ext_s = float(np.ptp(xyz, axis=0).max())
        mrgb = colors_of(sel_lab)
        others = sel_lab != lid

        def dim_others(f: float) -> None:
            a = np.ones(len(sel_lab), "f4")
            a[others] = 1.0 - 0.75 * f
            mesh.v[:, 9] = a
            mesh.dirty_v = True

        self.move_camera(run_time=T(2.0), target=focus, distance=ext_s / (2 * np.tan(np.radians(12))) * 1.75,
                         shift=(-0.24, 0.06))
        self.orbit(ORBIT * 0.5)
        self.animate_gl(dim_others, run_time=T(0.8))
        del mrgb

        # the profile plot, top right
        L = float(s_mm[-1])
        ymax = float(np.ceil(np.nanmax(r_cs[np.isfinite(r_cs)]) + 0.5))
        ax = make_axes(L, ymax)
        xl = Text("arc length (mm)", font=FONT, color=INK_SOFT, font_size=13).next_to(ax.x_axis, DOWN, buff=0.28)
        yl = Text("radius (mm)", font=FONT, color=INK_SOFT, font_size=13).rotate(np.pi / 2).next_to(
            ax.y_axis, LEFT, buff=0.3)
        title = Text(f"{name} radius profile", font=FONT, weight="BOLD", color=INK, font_size=16)
        plot = VGroup(ax, xl, yl)
        title.next_to(ax, UP, buff=0.12).align_to(ax, LEFT)
        panel = VGroup(plot, title)
        panel.to_corner(UR, buff=0.4)
        bg = Rectangle(width=panel.width + 0.4, height=panel.height + 0.3, fill_color="#FFFFFF", fill_opacity=0.9,
                       stroke_width=0).move_to(panel)
        board = VGroup(bg, panel)
        self.fixed(board, keep=True, drop=(title,))

        def curve(y: np.ndarray, color: str, width: float = 2.4) -> VMobject:
            ok = np.isfinite(y)
            yy = np.where(ok, y, np.interp(s_mm, s_mm[ok], y[ok]))
            m = VMobject(stroke_color=color, stroke_width=width)
            m.set_points_as_corners([ax.c2p(a, b) for a, b in zip(s_mm, yy)])
            self.fixed(m, keep=True)
            return m

        full_mis = curve(r_mis, MIS_C)
        full_cs = curve(r_cs, CS_C)
        part_mis = full_mis.copy()
        part_mis.pointwise_become_partial(full_mis, 0, 0.001)
        cursor = Line(ax.c2p(0, 0), ax.c2p(0, ymax), stroke_color=INK_FAINT, stroke_width=1.5)
        self.fixed(cursor, keep=True)
        self.play(FadeIn(board), run_time=T(0.6))
        self.add(part_mis, cursor)

        ball = Spheres(xyz[:1], radius=float(r_mis[0]), color=rgb(MIS_C), subdiv=3)
        ball.set(op=np.array([0.62], "f4"))
        self.gl.add(ball)
        trail = Tubes([xyz], radius=0.35, colors=[rgb(MIS_C)])
        trail.progress = 0.0
        self.gl.add(trail)

        def roll(f: float) -> None:
            k = int(round(f * (len(xyz) - 1)))
            ball.set(centers=xyz[k:k + 1], radius=np.array([max(r_mis[k], 0.05)], "f4"))
            part_mis.pointwise_become_partial(full_mis, 0, max(f, 0.001))
            cursor.put_start_and_end_on(ax.c2p(s_mm[k], 0), ax.c2p(s_mm[k], ymax))
            trail.progress = f

        self.animate_gl(roll, run_time=T(6.5))
        self.gl.remove(ball)

        # cross-sections: plane, contour, 36 rays, median circle
        secs = meta["sections"]
        tex = disc_texture()
        quads, contours, rays, circles = [], [], [], []
        for sct in secs:
            o, u, v = (np.asarray(sct[k], float) for k in ("origin", "u", "v"))
            R = 1.8 * float(sct["radius"]) + 0.8
            corners = np.array([o - R * u - R * v, o + R * u - R * v, o + R * u + R * v, o - R * u + R * v])
            q = Quad(corners, tex, opacity=0.0)
            quads.append(q)
            loops = [np.asarray(lp, float) for lp in sct["loops"] if len(lp) > 3]
            loops = [np.vstack([lp, lp[:1]]) for lp in loops]
            near = [lp for lp in loops if np.min(np.linalg.norm(lp - o, axis=1)) < 3.0 * float(sct["radius"]) + 1]
            c = Tubes(near or loops[:1], radius=0.07, colors=[rgb(INK)] * max(1, len(near or loops[:1])))
            c.opacity = 0.0
            contours.append(c)
            ends = np.asarray(sct["ray_end"], float)
            rr = Tubes([np.vstack([o, e]) for e in ends], radius=0.045, colors=[rgb(RAY_C)] * len(ends))
            rr.progress = 0.0
            rays.append(rr)
            cc = Tubes([ring(o, u, v, float(sct["radius"]))], radius=0.09, colors=[rgb(CS_C)])
            cc.progress = 0.0
            circles.append(cc)
        self.gl.add(*quads, *contours, *rays, *circles)
        dots = VGroup(*[Dot(ax.c2p(s_mm[sct["index"]], sct["radius"]), radius=0.045, color=CS_C) for sct in secs])
        self.fixed(dots, keep=True)
        for k, sct in enumerate(secs):
            i = int(sct["index"])
            fast = k > 1

            def show_sec(f: float, k=k, i=i) -> None:
                quads[k].opacity = min(1.0, 3 * f)
                contours[k].opacity = min(1.0, 3 * f)
                rays[k].progress = np.clip(3 * f - 0.8, 0, 1)
                circles[k].progress = np.clip(3 * f - 1.9, 0, 1)
                cursor.put_start_and_end_on(ax.c2p(s_mm[i], 0), ax.c2p(s_mm[i], ymax))

            self.animate_gl(show_sec, run_time=T(1.1 if fast else 2.4), extra=[FadeIn(dots[k], scale=2)])
            if not fast:
                self.hold(T(0.6))
        part_cs = full_cs.copy()
        part_cs.pointwise_become_partial(full_cs, 0, 0.001)
        self.add(part_cs)

        def draw_cs(f: float) -> None:
            part_cs.pointwise_become_partial(full_cs, 0, max(f, 0.001))
            k = int(round(f * (len(xyz) - 1)))
            cursor.put_start_and_end_on(ax.c2p(s_mm[k], 0), ax.c2p(s_mm[k], ymax))
            for q, c, rr, cc in zip(quads, contours, rays, circles):
                q.opacity = c.opacity = 1.0 - f
                rr.opacity = cc.opacity = 1.0 - f

        cap3b = self.caption("   the cross-section reads wider than the ball", [
            (f"median {np.nanmedian(r_cs):.2f} mm vs {np.nanmedian(r_mis):.2f} mm: the TOF lumen is not round",
             INK, 20),
            ("stenoses are tested on the ball, enlargements on the cut", INK_SOFT, 18),
        ])
        self.swap(cap3, cap3b)
        self.animate_gl(draw_cs, run_time=T(3.0), extra=[FadeOut(dots)])
        self.gl.remove(*quads, *contours, *rays, *circles)
        self.hold(T(1.0))

        # ------------------------------------------------------------------ 4 · shape
        sm = summary.get(lid, {})
        chord = float(sm.get("main_chord_length_mm") or np.linalg.norm(xyz[-1] - xyz[0]))
        tort = float(sm.get("tortuosity_dm") or L / chord)
        cap5 = self.caption("4 · Shape: length, tortuosity, curvature", [
            (f"{name}: path {L:.1f} mm / chord {chord:.1f} mm = tortuosity {tort:.2f}", INK, 20),
            ("curvature κ along the centreline (smoothed, true 1/mm)", INK_SOFT, 18),
            ("siphon mask: bends with κ ≥ 0.20 /mm at 3 mm, ± 5 mm", SIPHON_C, 18),
        ])
        self.swap(cap3b, cap5)
        kap = np.nan_to_num(S[:, cols["curvature_1_per_mm"]], nan=0.0)
        kap_s = np.convolve(kap, np.ones(15) / 15, mode="same")
        siph = S[:, cols["siphon_mask_point"]] > 0
        kcol = sequential(np.clip(kap_s / 0.6, 0, 1), ("#F3E9D2", "#E8B04B", "#D0612A", "#7A1631"))
        curv = Tubes([xyz], radius=0.55, colors=[kcol])
        curv.opacity = 0.0
        chord_t = Tubes([np.linspace(xyz[0], xyz[-1], 2)], radius=0.3, colors=[rgb(INK)])
        chord_t.progress = 0.0
        mask_cols = np.where(siph[:, None], rgb(SIPHON_C), rgb("#D9DDE3"))
        siph_t = Tubes([xyz], radius=0.62, colors=[mask_cols])
        siph_t.opacity = 0.0
        self.gl.add(curv, chord_t, siph_t)
        self.animate_gl(lambda f: setattr(trail, "opacity", 1.0 - f), run_time=T(0.8),
                        extra=[FadeOut(board), FadeOut(part_mis), FadeOut(part_cs), FadeOut(cursor)])
        self.gl.remove(trail)
        self.animate_gl(lambda f: setattr(chord_t, "progress", f), run_time=T(1.4))
        self.hold(T(1.4))
        kbar = self._colorbar(("#F3E9D2", "#E8B04B", "#D0612A", "#7A1631"), "curvature κ (1/mm)", 0.0, 0.6)
        self.animate_gl(lambda f: setattr(curv, "opacity", f), run_time=T(1.2), extra=[FadeIn(kbar)])
        self.hold(T(2.0))

        def to_siphon(f: float) -> None:
            siph_t.opacity = f
            curv.opacity = 1.0 - f

        self.animate_gl(to_siphon, run_time=T(1.2), extra=[FadeOut(kbar)])
        self.hold(T(2.2))

        # ------------------------------------------------------------------ 5 · stenosis (on its own path)
        ls = meta.get("lesion_show")
        k_les = int(ls["path_index"]) if ls and ls.get("path_index") is not None else show
        Q = paths[k_les]
        q_xyz = Q[:, :3]
        q_s = Q[:, cols["s_mm"]]
        q_name = meta["paths"][k_les]["vessel"].replace("_", " ")
        q_lid = meta["paths"][k_les]["label"]
        sig = Q[:, cols["robust_radius_mm"]]
        ref = Q[:, cols["robust_reference_radius_mm"]]
        i0 = int(ls["point_start"]) if ls else 0
        i1 = int(ls["point_end"]) if ls else 0
        lesion_on = bool(ls) and 0 <= i0 < i1 < len(Q)
        Lq = float(q_s[-1])
        ymax2 = float(np.ceil(np.nanmax(np.r_[sig[np.isfinite(sig)], ref[np.isfinite(ref)] * 1.2]) * 2) / 2)
        ax2 = make_axes(Lq, ymax2, xstep=float(10 if Lq < 60 else 20), ystep=0.5 if ymax2 <= 2 else 1.0)
        xl2 = Text("arc length (mm)", font=FONT, color=INK_SOFT, font_size=13).next_to(ax2.x_axis, DOWN, buff=0.28)
        t2 = Text(f"{q_name}: radius vs its healthy reference", font=FONT, weight="BOLD", color=INK, font_size=16)
        t2.next_to(ax2, UP, buff=0.12).align_to(ax2, LEFT)
        pan2 = VGroup(ax2, xl2, t2).to_corner(UR, buff=0.4)
        bg2 = Rectangle(width=pan2.width + 0.4, height=pan2.height + 0.3, fill_color="#FFFFFF", fill_opacity=0.9,
                        stroke_width=0).move_to(pan2)
        ok = np.isfinite(ref) & np.isfinite(sig)
        ss, rf, sg = q_s[ok], ref[ok], sig[ok]
        band = Polygon(*[ax2.c2p(a, b) for a, b in zip(ss, rf * 1.2)],
                       *[ax2.c2p(a, b) for a, b in zip(ss[::-1], rf[::-1] * 0.8)],
                       fill_color=REF_C, fill_opacity=0.16, stroke_width=0)
        ref_line = VMobject(stroke_color=REF_C, stroke_width=2.4)
        ref_line.set_points_as_corners([ax2.c2p(a, b) for a, b in zip(ss, rf)])
        sig_line = VMobject(stroke_color=CS_C, stroke_width=2.2)
        sig_line.set_points_as_corners([ax2.c2p(a, b) for a, b in zip(ss, sg)])
        les_marks = VGroup()
        if lesion_on:
            les_marks.add(Rectangle(width=ax2.c2p(q_s[i1], 0)[0] - ax2.c2p(q_s[i0], 0)[0], height=ax2.y_length,
                                    fill_color=LESION_C, fill_opacity=0.14, stroke_width=0)
                          .move_to(ax2.c2p((q_s[i0] + q_s[i1]) / 2, ymax2 / 2)))
            seg_line = VMobject(stroke_color=LESION_C, stroke_width=3.4)
            okk = np.isfinite(sig[i0:i1 + 1])
            seg_line.set_points_as_corners([ax2.c2p(a, b) for a, b in zip(q_s[i0:i1 + 1][okk], sig[i0:i1 + 1][okk])])
            les_marks.add(seg_line)
        board2 = VGroup(bg2, pan2, band, ref_line, sig_line)
        self.fixed(board2, les_marks, keep=True, drop=(t2,))
        if lesion_on:
            kind = ls.get("kind", "stenosis")
            dpc = abs(float(ls.get("diameter_percent") or 0.0))
            third = (f"{q_name}: a {dpc:.0f} % {'narrowing' if kind == 'stenosis' else 'widening'} over "
                     f"{float(ls.get('length_mm') or 0):.1f} mm  ({ls.get('confidence_class', '')} confidence)")
        else:
            third = "this subject: every run stays in the band"
        cap6 = self.caption("5 · Stenosis and enlargement", [
            ("a robust reference per junction-to-junction stretch (LOWESS of log r)", REF_C, 20),
            ("a lesion: a run ≥ 20 % below it (≥ 3 mm) or above it (≥ 1.5 mm)", INK_SOFT, 18),
            (third, LESION_C if lesion_on else INK, 18),
        ])
        self.swap(cap5, cap6)
        q_focus = q_xyz[i0:i1 + 1].mean(0) if lesion_on else 0.5 * (q_xyz.max(0) + q_xyz.min(0))
        q_ext = float(np.ptp(q_xyz, axis=0).max())
        others_q = sel_lab != q_lid
        q_tube = Tubes([q_xyz], radius=0.3, colors=[rgb("#5A6270")])
        q_tube.progress = 0.0
        les_tube = Tubes([q_xyz[i0:i1 + 1]] if lesion_on else [q_xyz[:2]], radius=0.75, colors=[rgb(LESION_C)])
        les_tube.opacity = 0.0
        self.gl.add(q_tube, les_tube)

        def to_lesion_view(f: float) -> None:
            siph_t.opacity = chord_t.opacity = 1.0 - f
            a = np.ones(len(sel_lab), "f4")
            a[others_q] = 0.25 * (1 - f) + 0.08 * f
            mesh.v[:, 9] = a
            mesh.dirty_v = True
            mesh.opacity = 0.2 + 0.25 * f
            lines_lab.opacity = 1.0 - 0.8 * f

        self.animate_gl(to_lesion_view, run_time=T(1.2))
        self.gl.remove(siph_t, chord_t, curv)
        self.move_camera(run_time=T(2.0), target=q_focus, distance=max(q_ext * 0.9, 45.0) / (2 * np.tan(np.radians(12))),
                         shift=(-0.22, 0.06))
        self.play(FadeIn(bg2), FadeIn(pan2), run_time=T(0.6))
        self.add(sig_line)
        sig_full = sig_line.copy()
        sig_line.pointwise_become_partial(sig_full, 0, 0.001)
        self.animate_gl(lambda f: (setattr(q_tube, "progress", f),
                                   sig_line.pointwise_become_partial(sig_full, 0, max(f, 0.001))), run_time=T(1.8))
        self.play(FadeIn(band), Create(ref_line), run_time=T(1.4))
        if lesion_on:
            self.animate_gl(lambda f: setattr(les_tube, "opacity", f), run_time=T(0.8),
                            extra=[FadeIn(les_marks)])
        self.hold(T(3.4))

        # ------------------------------------------------------------------ 6 · variants (fetal example)
        vm = meta["variants"]
        v_aff = np.asarray(vm["affine"], float)
        v_world = (v_aff[:3, :3] @ np.asarray(data["v_vox"], float).T).T + v_aff[:3, 3]
        v_lab = np.asarray(data["v_lab"], int)
        v_mesh = Mesh(Mesh.from_triangles(np.asarray(data["v_mesh_v"]), np.asarray(data["v_mesh_f"]),
                                          colors=colors_of(np.asarray(data["v_mesh_lab"]))), opacity=0.0, amb=0.45,
                      spec=0.22)
        self.gl.add(v_mesh)
        les = [r for r in vm["lesions"] if r.get("kind") in ("hypoplasia", "fetal")]
        cap7 = self.caption("6 · The circle of Willis, as a whole", [
            ("each vessel checked against the expected topology (here: the fetal example)", INK, 20),
            ("  ·  ".join(f"{r['vessel_name'].replace('_', ' ')}: {r['kind']}" for r in les) or "complete", INK_SOFT,
             18),
            ("an absent or thread-like segment: a variant, not a lesion", INK_FAINT, 18),
        ])
        self.swap(cap6, cap7)

        def main_out(f: float) -> None:
            mesh.opacity = 0.45 * (1 - f)
            lines_lab.opacity = 0.2 * (1 - f)
            q_tube.opacity = les_tube.opacity = 1.0 - f

        self.animate_gl(main_out, run_time=T(1.0), extra=[FadeOut(board2), FadeOut(les_marks)])
        self.gl.remove(q_tube, les_tube)
        c_cow = v_world[np.isin(v_lab, [4, 5, 10, 11, 12, 3])].mean(0)
        self.orbit(ORBIT * 0.4)
        self.move_camera(run_time=T(1.2), target=c_cow, distance=dist * 0.6, phi=50, shift=(0.0, 0.1))
        self.animate_gl(lambda f: setattr(v_mesh, "opacity", f), run_time=T(1.2))
        tags = []
        ref_pts = {6: (5, 4), 12: (12, 10), 14: (10, 10)}  # where to pin each variant: (own label, else neighbour)
        for r in les:
            own, nb = ref_pts.get(int(r["label"]), (int(r["label"]), int(r["label"])))
            pts = v_world[v_lab == own] if (v_lab == own).any() else v_world[v_lab == nb]
            if not len(pts):
                continue
            p = pts.mean(0)
            if int(r["label"]) == 6:  # absent right A1: mirror the left one across the AComm
                comm = v_world[v_lab == 4].mean(0) if (v_lab == 4).any() else p
                p = 2 * comm - pts.mean(0)
            kind = "absent" if r["kind"] == "hypoplasia" and not (v_lab == int(r["label"])).any() else r["kind"]
            if r["kind"] == "fetal":
                kind = "fetal (fed through the RPComm)"
            offs = {6: (-1.2, 0.5), 12: (0.35, -0.4), 14: (1.3, 0.4)}.get(int(r["label"]), (0.0, 0.2))
            tags.append(self.label_3d(f"{r['vessel_name'].replace('_', ' ')}: {kind}", p, color="#B0302A",
                                      font_size=15, offset=offs))
        if tags:
            self.play(*[FadeIn(t) for t in tags], run_time=T(0.8))
        self.hold(T(3.8))
        if tags:
            self.play(*[FadeOut(t) for t in tags], run_time=T(0.5))

        # back to the stenosis example: its surfaces, whole
        def main_in(f: float) -> None:
            v_mesh.opacity = 1.0 - f
            mesh.v[:, 9] = 1.0
            mesh.dirty_v = True
            mesh.opacity = f
            mesh.rim = 0.0

        self.animate_gl(main_in, run_time=T(1.2))
        self.gl.remove(v_mesh)

        # ------------------------------------------------------------------ 8 · summary
        cap8 = self.caption("Per-vessel summary", [
            ("length · mean diameter · tortuosity, and much more per segment and lesion", INK, 20),
            ("vessel_summary · segments · lesions · per-point tables · centerlines (.vtp)", INK_SOFT, 18),
        ])
        self.swap(cap7, cap8)
        self.move_camera(run_time=T(2.0), target=centre, distance=dist * 1.05, phi=70, shift=(-0.3, 0.12))
        self.orbit(ORBIT * 0.6)
        pick = [1, 2, 3, 7, 8, 5, 21, 23, 24, 13, 14]
        rows = []
        for lid_ in pick:
            r = summary.get(lid_)
            if r is None or not r.get("present") or r.get("main_length_mm") is None:
                continue
            rows.append((lid_, r))
        heads = ["vessel", "length", "Ø mean", "tortuosity", "volume"]
        table = VGroup()
        cells = [[Text(h, font=FONT, weight="BOLD", color=INK, font_size=14) for h in heads]]
        for lid_, r in rows:
            cells.append([
                VGroup(Rectangle(width=0.14, height=0.14, fill_color=EICAB_COLORS.get(lid_, INK), fill_opacity=1,
                                 stroke_width=0),
                       Text(r["vessel_name"].replace("_", " "), font=FONT, color=INK, font_size=14)).arrange(
                    RIGHT, buff=0.08),
                Text(f"{r['main_length_mm']:.0f} mm", font=FONT, color=INK, font_size=14),
                Text(f"{r['diameter_mean_mm']:.2f} mm", font=FONT, color=INK, font_size=14),
                Text(f"{r['tortuosity_dm']:.2f}", font=FONT, color=INK, font_size=14),
                Text(f"{r['volume_mm3']:.0f} mm³", font=FONT, color=INK, font_size=14),
            ])
        widths = [max(row[c].width for row in cells) for c in range(len(heads))]
        for k, row in enumerate(cells):
            x = 0.0
            for c, cell in enumerate(row):
                if c == 0:
                    cell.move_to([x + cell.width / 2, -k * 0.3, 0])
                else:
                    cell.move_to([x + widths[c] - cell.width / 2, -k * 0.3, 0])
                x += widths[c] + 0.35
                table.add(cell)
        rule = Line(table.get_corner(UL) + DOWN * 0.22, table.get_corner(UR) + DOWN * 0.22, stroke_color=INK_FAINT,
                    stroke_width=1)
        table.add(rule)
        table.to_corner(UR, buff=0.45)
        table.add_background_rectangle(color="#FFFFFF", opacity=0.9, buff=0.15)
        self.fixed(table, keep=True)
        self.play(FadeIn(table, shift=LEFT * 0.2), run_time=T(1.0))
        self.hold(T(5.5))
        self.play(FadeOut(table), FadeOut(cap8), run_time=T(1.0))
        self.hold(T(0.6))

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
    run_scene(__file__, "TofMorphometrics", "tof_morphometrics")
