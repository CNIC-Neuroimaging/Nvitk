"""Manim animation of the complete eICAB + post-process (qvtpy stage 1), every 0.5 mm voxel.

From the TOF angiogram to the post-processed labels, on the stenosis example: eICAB's
circle-of-Willis labels inside its box, then the nvitk post-process
(:func:`nvitk.pipes.qvtpy.util.eicab.eicab_postprocess.postprocess_eicab_variants`): local
half-maximum re-segmentation (the anterior choroidals go with it), ICAs to the image floor, the
competitive distal growth along the Frangi tree (the expansion), and, quickly on one isolated ICA,
the siphon min-cut (the fetal example's left ICA as stage 1 cut it: this subject's siphons close
no loop), ending on the WB_pp / CW_pp masks.

The 3D content is drawn on the GPU (:mod:`gl_scene`), so every voxel of the masks (~90,000) is a
cube of its own, over a ray-marched MIP of the TOF. Every state comes from
``eicab_full_stages.npz`` (``eicab_full_precompute.py``); this file needs numpy, moderngl and
manim only.

Render (manim environment; ``--no-text`` keeps every caption, legend and inset but transparent)::

    EF_FAST=1 python scripts/anim/eicab_full_manim.py -ql --fps 15
    python scripts/anim/eicab_full_manim.py -qh --fps 30              # -> eicab_full.mp4
    python scripts/anim/eicab_full_manim.py -qk --fps 30              # 4K
    python scripts/anim/eicab_full_manim.py -qh --fps 30 --no-text    # -> eicab_full_notext.mp4
"""

from __future__ import annotations

import json
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # numpy's per-frame work is small: no 32-thread spin
import sys
from pathlib import Path

import numpy as np
from manim import BOLD, DOWN, LEFT, RIGHT, UL, UP, UR, FadeIn, FadeOut, ImageMobject, Rectangle, Text, VGroup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anim_voxels import FONT, INK, INK_FAINT, INK_SOFT, ramp3, rgb, run_scene  # noqa: E402
from gl_render import Spheres, Tubes, VolumeMIP, Voxels  # noqa: E402
from gl_scene import GLScene  # noqa: E402
from vessel_palette import ADDED, CUT, EICAB_COLORS, EICAB_NAMES, GHOST, PATH, REMOVED, colors_of  # noqa: E402

DATA_PATH = Path(os.environ.get("EF_STAGES", str(Path(__file__).with_name("eicab_full_stages.npz"))))
FAST = os.environ.get("EF_FAST", "0") == "1"
ORBIT = 3.2  # deg / s
SHIFT = (0.0, 0.16)  # lifts the vessels clear of the bottom-left captions


class EicabFull(GLScene):
    """eICAB and its post-process, voxel by voxel."""

    def construct(self) -> None:
        data = np.load(DATA_PATH)
        meta = json.loads(str(data["meta"]))
        T = (lambda x: max(0.3, x * 0.35)) if FAST else (lambda x: x)
        aff = np.asarray(meta["affine"], float)
        sp = np.asarray(meta["spacing_mm"], float)

        idx = np.asarray(data["vox"], float)
        world = (aff[:3, :3] @ idx.T).T + aff[:3, 3]
        lab = {k: np.asarray(data[f"lab_{k}"], int) for k in
               ("raw", "noacha", "reseg", "ext", "loopfill", "distal", "wb", "cw")}
        tree = np.asarray(data["tree"], bool)
        owner = np.asarray(data["owner"], int)
        arrival = np.asarray(data["arrival"], float)
        depth = np.asarray(data["depth"], float)
        grow_order = np.asarray(data["grow_order"], float)
        ext_order = np.asarray(data["ext_order"], float)
        v = meta["volume_mm3"]

        vox = Voxels(world, pitch=float(sp[0]), fill=0.9)
        mip = VolumeMIP(np.asarray(data["tof8"]), np.asarray(data["tof_affine"], float), opacity=0.0, lo=0.18,
                        hi=0.8, gamma=1.3)
        self.gl.add(mip, vox)

        # camera: the final WB mask fills the view
        wb_pts = world[lab["wb"] > 0]
        centre = 0.5 * (wb_pts.max(0) + wb_pts.min(0))
        extent = float(np.ptp(wb_pts, axis=0)[:2].max())
        dist = extent / (2 * np.tan(np.radians(12))) * 1.05
        self.set_camera(target=centre, distance=dist * 1.25, phi=72, theta=92, fov=24, shift=SHIFT)

        def style(stage: str, op: float = 1.0) -> tuple:
            lb = lab[stage]
            return colors_of(lb), np.where(lb > 0, op, 0.0).astype("f4"), np.where(lb > 0, 1.0, 0.3).astype("f4")

        def legend(ids: list[int]) -> VGroup:
            items = [VGroup(Rectangle(width=0.16, height=0.16, fill_color=EICAB_COLORS[i], fill_opacity=1,
                                      stroke_width=0),
                            Text(EICAB_NAMES[i], font=FONT, color=INK, font_size=13)).arrange(RIGHT, buff=0.08)
                     for i in ids]
            half = (len(items) + 1) // 2
            cols = VGroup(VGroup(*items[:half]).arrange(DOWN, buff=0.06, aligned_edge=LEFT),
                          VGroup(*items[half:]).arrange(DOWN, buff=0.06, aligned_edge=LEFT))
            g = cols.arrange(RIGHT, buff=0.45, aligned_edge=UP).to_corner(UL, buff=0.35)
            g.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.1)
            self.fixed(g, keep=True)  # the colour key stays in the text-free version
            return g

        def inset(key: str, width: float = 6.0):
            img = ImageMobject(np.asarray(data[key], np.uint8))
            img.width = width
            img.to_corner(UR, buff=0.35)
            return img

        # ---------------------------------------------------------------- 0 · title, TOF MIP
        self.orbit(ORBIT)
        card = self.title_card("From TOF angiography to a labelled arterial tree",
                               "eICAB + nvitk post-process  ·  qvtpy stage 1",
                               "TOF MRA, 0.5 mm voxels  ·  every voxel drawn")
        self.animate_gl(lambda f: setattr(mip, "opacity", 0.9 * f), run_time=T(2.0), extra=[FadeIn(card)])
        self.hold(T(1.5))
        cap0 = self.caption("TOF MR angiography", [
            ("inflowing blood is bright: the arteries stand out of the brain", INK, 20),
            ("shown as a maximum-intensity projection, turning", INK_SOFT, 18),
        ])
        self.swap(card, cap0)
        self.hold(T(2.5))

        # ---------------------------------------------------------------- 1 · eICAB
        lo_v, hi_v = (np.asarray(b, float) for b in meta["eicab_box_vox"])
        corners = np.array([[x, y, z] for x in (lo_v[0], hi_v[0]) for y in (lo_v[1], hi_v[1])
                            for z in (lo_v[2], hi_v[2])])
        cw = (aff[:3, :3] @ corners.T).T + aff[:3, 3]
        edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
        box = Tubes([np.linspace(cw[a], cw[b], 2) for a, b in edges], radius=0.35, colors=[rgb(INK_SOFT)] * 12)
        box.progress = 0.0
        self.gl.add(box)
        cap1 = self.caption("eICAB: a deep-learning labeller", [
            (f"{meta['n_raw_labels']} arteries of the circle of Willis, each with its own label", INK, 20),
            ("only inside its box  ·  masks thicker than the lumen", INK_SOFT, 18),
            ("trunks cut at the box's floor  ·  distal arteries stop short", INK_SOFT, 18),
        ])
        self.swap(cap0, cap1)
        self.animate_gl(lambda f: setattr(box, "progress", f), run_time=T(1.2))
        present = sorted(set(np.unique(lab["raw"])) - {0})
        leg = legend([i for i in present])
        rc, ro, rs = style("raw")
        self.voxel_transition(vox, rc, ro, rs, run_time=T(3.2), delay=world[:, 2], lag=0.75, glow="#FFFFFF",
                              glow_strength=0.5, extra=[FadeIn(leg)])
        raw_pts = world[lab["raw"] > 0]
        c_raw = 0.5 * (raw_pts.max(0) + raw_pts.min(0))
        self.move_camera(run_time=T(1.6), target=c_raw, distance=dist * 0.78,
                         extra=[])
        self.animate_gl(lambda f: setattr(mip, "opacity", 0.9 - 0.55 * f), run_time=T(1.0))
        self.hold(T(1.6))

        # ---------------------------------------------------------------- 2 · re-segmentation
        # (the anterior choroidals are dropped first: they vanish with the peel, uncommented)
        drop = (1.0 - v["reseg"] / max(v["raw"], 1e-9)) * 100.0
        cap3 = self.caption("1 · Local re-segmentation at half maximum", [
            ("T = bg + ½(core − bg), per vessel, from its own intensities", INK, 20),
            ("core: its centreline  ·  bg: a 1–3 mm ring  ·  other labels are barriers", INK_SOFT, 18),
            (f"volume {v['raw']:.0f} → {v['reseg']:.0f} mm³  (−{drop:.0f} %): the mask hugs the lumen",
             REMOVED, 18),
        ])
        self.swap(cap1, cap3)
        self.move_camera(run_time=T(1.4), distance=dist * 0.62)
        img = inset("inset_reseg")
        self.play(FadeIn(img), run_time=T(0.5))
        peel = (lab["raw"] > 0) & (lab["reseg"] == 0)
        grow = (lab["raw"] == 0) & (lab["reseg"] > 0)
        sc, so, ss = style("reseg")
        flash = np.where(peel[:, None], rgb(REMOVED), np.where(grow[:, None], rgb(ADDED), rc))
        self.voxel_transition(vox, flash, run_time=T(0.7))
        # outer shell first: delay by depth inside eICAB's mask
        self.voxel_transition(vox, sc, so, np.where(peel, 0.15, ss), run_time=T(3.6), delay=depth, lag=0.7,
                              extra=[FadeOut(leg)])
        leg = legend(sorted(set(np.unique(lab["reseg"])) - {0}))
        self.play(FadeIn(leg), run_time=T(0.4))
        self.hold(T(1.2))
        self.play(FadeOut(img), run_time=T(0.4))

        # ---------------------------------------------------------------- 3 · ICAs to the floor
        e = meta["extension"]

        def ext_text(lid: str) -> str:
            x = e.get(lid) or {}
            return f"{EICAB_NAMES[int(lid)]} +{x.get('length_mm', 0)} mm"

        cap4 = self.caption("2 · ICAs to the image floor", [
            ("minimum-cost path through a multi-scale Frangi tree (+ bright lumen)", INK, 20),
            ("from the trunk's open end down to a vessel cross-section at the floor", INK_SOFT, 18),
            ("  ·  ".join(ext_text(k) for k in sorted(e)), INK, 18),
        ])
        self.swap(cap3, cap4)
        all_pts = world[(lab["wb"] > 0) | (lab["ext"] > 0)]
        c2 = 0.5 * (all_pts.max(0) + all_pts.min(0))
        self.move_camera(run_time=T(1.6), target=c2, distance=dist * 1.3, phi=80)
        paths = np.asarray(data["ext_paths"], float)
        pls = []
        for k in np.unique(paths[:, 3]) if len(paths) else []:
            p = paths[paths[:, 3] == k][:, :3]
            pls.append((aff[:3, :3] @ p.T).T + aff[:3, 3])
        route = Tubes(pls, radius=0.45, colors=[rgb(PATH)] * len(pls))
        route.progress = 0.0
        self.gl.add(route)
        self.animate_gl(lambda f: setattr(route, "progress", f), run_time=T(2.0))
        new = (lab["loopfill"] > 0) & (lab["reseg"] == 0)  # the loop fill's few voxels come along
        ec, eo, es = style("loopfill")
        self.voxel_transition(vox, ec, eo, es, run_time=T(3.2), delay=np.where(new, ext_order, -1.0), lag=0.8,
                              glow="#FFFFFF", glow_strength=0.6)

        def fade_out(f: float) -> None:
            route.opacity = 1.0 - f
            box.opacity = 1.0 - f

        self.animate_gl(fade_out, run_time=T(0.9))
        self.gl.remove(route, box)
        self.move_camera(run_time=T(1.4), target=centre, distance=dist * 1.15, phi=70)

        # ---------------------------------------------------------------- 4 · distal growth
        added = sum(float(x) for x in meta["distal"].values())
        cap6 = self.caption("3 · Distal growth along the Frangi tree", [
            ("multi-scale Frangi vesselness → a hysteresis tree of tubes", INK, 20),
            ("ACAs, MCAs, PCAs (P2) and SCAs flood it at once: each voxel to the front that reaches it first",
             INK_SOFT, 17),
            (f"+{added / 1000:.1f} cm³; ACA / MCA growth written as their A2 / M2", ADDED, 18),
        ])
        self.swap(cap4, cap6)
        lc, lo_, ls = ec, eo, es
        t_op = np.where(tree, 0.22, lo_)
        t_rgb = np.where(tree[:, None], rgb(GHOST), lc)
        self.voxel_transition(vox, t_rgb, t_op, np.where(tree, 0.7, ls), run_time=T(1.6), delay=-world[:, 2],
                              lag=0.5)
        self.hold(T(0.6))
        # the fronts: tree voxels take their owner's colour in the order the fronts reached them
        won = tree & (owner > 0)
        f_rgb = np.where(won[:, None], colors_of(owner), t_rgb)
        f_op = np.where(won, 0.9, t_op)
        self.voxel_transition(vox, f_rgb, f_op, np.where(won, 0.8, np.where(tree, 0.7, ls)), run_time=T(6.0),
                              delay=np.where(won, arrival, -1.0), lag=0.92, glow="#FFD25A", glow_strength=0.8)
        # the lumen around each claimed path, written as A2 / M2 (no separate step)
        leg2 = legend(sorted(set(np.unique(lab["wb"])) - {0}))
        wc, wo, ws = style("wb")
        newd = (lab["wb"] > 0) & (lab["loopfill"] == 0)
        self.voxel_transition(vox, wc, wo, ws, run_time=T(2.2), delay=np.where(newd, grow_order, -1.0), lag=0.6,
                              extra=[FadeOut(leg), FadeIn(leg2)])
        self.hold(T(1.2))

        # ---------------------------------------------------------------- 5 · siphon correction (quick)
        prev = self._siphon_quick(data, meta, vox, mip, cap6, centre, dist, T)
        self.move_camera(run_time=T(1.6), target=centre, distance=dist * 1.1, phi=70)
        self.animate_gl(lambda f: setattr(mip, "opacity", 0.35 * f), run_time=T(0.5))

        # ---------------------------------------------------------------- 8 · the two masks
        cap8 = self.caption("Post-processed masks", [
            (f"WB_pp: all steps, {v['wb']:.0f} mm³ (stage 7 measures it)", INK, 20),
            (f"CW_pp: without the distal growth, {v['cw']:.0f} mm³", INK_SOFT, 18),
            (f"from eICAB's {v['raw']:.0f} mm³: thinner lumen, full trunks, distal trees", INK_FAINT, 18),
        ])
        self.swap(prev, cap8)
        self.hold(T(1.5))
        cc, co, cs = style("cw")
        dimmed = (lab["wb"] > 0) & (lab["cw"] == 0)
        self.voxel_transition(vox, wc, np.where(dimmed, 0.12, wo), ws, run_time=T(1.4))
        self.hold(T(1.6))
        self.voxel_transition(vox, wc, wo, ws, run_time=T(1.4))
        self.hold(T(2.5))
        self.play(FadeOut(cap8), FadeOut(leg2), run_time=T(1.0))
        self.hold(T(0.5))

    def _siphon_quick(self, data, meta, vox: Voxels, mip: VolumeMIP, prev, centre, dist, T):
        """The ICA siphon min-cut of ``siphon_cut_manim.py``, quickly, on one isolated ICA (the
        fetal example's left ICA as stage 1 cut it: this subject's siphons close no loop)."""
        sm = meta["siphon_demo"]
        P = lambda k: np.asarray(data[f"sip_{k}"])  # noqa: E731
        sp = float(sm["spacing_mm"][0])
        cen = np.asarray(P("blk_centres"), float) * sp
        loop = np.asarray(P("loop_nodes"), float) * sp
        off = centre - 0.5 * (cen.max(0) + cen.min(0))  # shown where the tree was
        cen, loop = cen + off, loop + off

        def at(k: str) -> np.ndarray:
            return np.asarray(P(k), float).reshape(-1, 3) * sp + off

        occ_raw, occ_pre, occ_rep = (np.asarray(P(k), float) > 0 for k in ("occ_raw", "occ_pretess", "occ_repaired"))
        tof_rel = np.asarray(P("tof_rel"), float)
        t = np.clip(tof_rel / max(np.percentile(tof_rel[occ_raw], 97), 1e-9), 0, 1)
        tof_rgb = np.asarray(ramp3("#3B4A63", "#8FA3C2", "#E9EEF6", t), "f4")
        weight = np.asarray(P("weight"), float)
        wn = np.clip(weight / max(np.percentile(weight[occ_pre], 95), 1e-9), 0, 1)
        w_rgb = np.asarray(ramp3("#F6C445", "#A9BCD4", "#1F3F66", wn), "f4")
        slab, source, sink = (np.asarray(P(k), float) > 0 for k in ("slab_frac", "source_frac", "sink_frac"))

        ica = Voxels(cen, pitch=sp, fill=0.9)
        ica.set(rgb=tof_rgb, op=np.zeros(len(cen), "f4"))
        pre = Voxels(at("pretess_removed"), pitch=sp, fill=1.0)
        pre.set(rgb=rgb(CUT), op=np.zeros(len(pre), "f4"), size=np.full(len(pre), 1.6, "f4"))
        cutv = Voxels(at("cut_voxels"), pitch=sp, fill=1.0)
        cutv.set(rgb=rgb(CUT), op=np.zeros(len(cutv), "f4"), size=np.full(len(cutv), 1.25, "f4"))
        ring = Tubes([np.vstack([loop, loop[:1]])], radius=0.35, colors=[rgb("#1B1D23")])
        ring.progress = 0.0
        bridge = np.asarray(P("loop_bridge"), bool)
        arcs_b = Tubes([loop[[i, (i + 1) % len(loop)]] for i in range(len(loop)) if bridge[i] or bridge[(i + 1) % len(loop)]],
                       radius=0.45, colors=None)
        arcs_b.set_colors(np.tile(np.r_[rgb("#8E44AD"), 1.0], (len(arcs_b.v), 1)))
        arcs_b.opacity = 0.0
        contact = Spheres(loop[np.asarray(P("loop_contact"), bool)], radius=0.7, color=rgb("#E8443A"))
        contact.set(op=np.zeros(len(contact), "f4"))
        cl_b = Tubes([at("cl_before")], radius=0.3, colors=[rgb("#DE4133")])
        cl_b.progress = 0.0
        cl_a = Tubes([at("cl_after")], radius=0.35, colors=[rgb("#2E9E4F")])
        cl_a.progress = 0.0
        self.gl.add(ica, pre, cutv, ring, arcs_b, contact, cl_b, cl_a)

        cap = self.caption("4 · ICA siphon correction", [
            ("where a siphon's limbs touch, the mask closes a loop (β₁ = 1)", INK, 20),
            ("cut only the contact: the cheapest set of voxels that opens the ring", INK_SOFT, 18),
            (f"one isolated ICA (the fetal example's {sm['label_name']}: this subject's siphons are open)",
             INK_FAINT, 16),
        ])
        beta = VGroup(Text("β₁", font=FONT, color=INK_SOFT, font_size=26),
                      Text(str(sm["betti_raw"]["beta1"]), font=FONT, weight=BOLD, color=REMOVED, font_size=34))
        beta.arrange(RIGHT, buff=0.18)
        note = Text(" ", font=FONT, color=INK, font_size=18)
        badge = VGroup(beta, note).arrange(DOWN, buff=0.15, aligned_edge=RIGHT).to_corner(UR, buff=0.5)
        self.fixed(badge, keep=True, drop=(note,))  # the β₁ counter stays, the notes go

        def say(text: str, color: str = INK, rt: float = 0.35):
            new = Text(text, font=FONT, color=color, font_size=18).next_to(beta, DOWN, buff=0.22).align_to(beta, RIGHT)
            new.add_background_rectangle(color="#FFFFFF", opacity=0.88, buff=0.08)
            self.fixed(new)
            self.play(FadeOut(badge[1]), FadeIn(new), run_time=T(rt))
            badge.remove(badge[1])
            badge.add(new)

        # isolate: the tree goes, the ICA comes
        u_, s_, vt = np.linalg.svd(loop - loop.mean(0))
        nrm = vt[-1] if vt[-1][0] >= 0 else -vt[-1]
        if sm["label_name"].startswith("L"):
            nrm = -nrm
        th = float(np.degrees(np.arctan2(nrm[1], nrm[0])))
        ph = float(np.clip(np.degrees(np.arccos(np.clip(nrm[2], -1, 1))), 60, 84))
        self.orbit(0.0)
        th = self.cam.theta + ((th - self.cam.theta + 180.0) % 360.0 - 180.0)
        ext = float(np.ptp(cen, axis=0).max())
        m0 = mip.opacity
        op0 = np.array(vox.op)

        def swap_in(f: float) -> None:
            vox.set(op=(op0 * (1 - f)).astype("f4"))
            mip.opacity = m0 * (1 - f)
            ica.set(op=np.where(occ_raw, 0.35 + 0.6 * t, 0.0).astype("f4") * f)

        self.swap(prev, cap, run_time=0.6)
        self.animate_gl(swap_in, run_time=T(1.2))  # the tree fades before the camera moves in
        self.move_camera(run_time=T(1.8), target=loop.mean(0), distance=ext / (2 * np.tan(np.radians(12))) * 1.15,
                         phi=ph, theta=th, shift=(0.0, 0.1), extra=[FadeIn(badge)])
        self.orbit(2.0)
        base_op = np.where(occ_raw, 0.1 + 0.22 * t, 0.0).astype("f4")
        self.animate_gl(lambda f: (setattr(ring, "progress", f), setattr(cl_b, "progress", f),
                                   ica.set(op=(np.where(occ_raw, 0.35 + 0.6 * t, 0) * (1 - f) + base_op * f)
                                           .astype("f4"))), run_time=T(1.4))
        say(f"a centerline shortcuts across: {sm['cl_before_mm']} mm", "#DE4133")
        self.hold(T(0.8))
        # pretess
        say(f"pretess: {sm['pretess_removed']} edge / corner contacts removed", CUT)
        self.animate_gl(lambda f: pre.set(op=np.full(len(pre), min(1.0, 2 * f) * (1 - max(0.0, 2 * f - 1)), "f4")),
                        run_time=T(1.2), extra=[])
        ica.set(op=np.where(occ_pre, 0.1 + 0.22 * t, 0.0).astype("f4"))
        # bridge + contact
        say(f"bridge {sm['bridge_mm']} mm across the contact, curl {sm['curl_mm']} mm", "#8E44AD")
        self.animate_gl(lambda f: (setattr(arcs_b, "opacity", f), contact.set(op=np.full(len(contact), f, "f4"))),
                        run_time=T(0.8))
        self.hold(T(0.6))
        # cost field
        say("cost per voxel: bright lumen dear, wall and contact cheap", "#B7860B")
        w_op = np.where(occ_pre, np.where(wn < 0.25, 0.85, 0.3 + 0.45 * wn), 0.0).astype("f4")
        self.voxel_transition(ica, w_rgb, w_op, run_time=T(1.2),
                              hook=lambda f: setattr(ring, "opacity", 1.0 - f))
        self.hold(T(0.6))
        # open the ring, source / sink
        say("ring opened opposite the contact: source / sink beside it", "#2E9E4F")
        o_rgb = np.where(source[:, None], rgb("#2E9E4F"), np.where(sink[:, None], rgb("#2F7FD6"), w_rgb))
        o_op = np.where(slab, 0.06, np.where(source | sink, 0.9, w_op)).astype("f4")
        self.voxel_transition(ica, o_rgb, o_op, run_time=T(1.0))
        self.hold(T(0.5))
        # the cut
        say(f"minimum vertex cut: {sm['cut_voxels']} voxels = {sm['cut_mm3']} mm³", CUT)
        self.voxel_transition(ica, o_rgb, (o_op * 0.3).astype("f4"), run_time=T(0.6),
                              hook=lambda f: cutv.set(op=np.full(len(cutv), f, "f4")))
        self.hold(T(1.0))
        # accept
        new_b = Text(str(sm["betti_repaired"]["beta1"]), font=FONT, weight=BOLD, color=ADDED, font_size=34)
        new_b.move_to(beta[1])
        self.fixed(new_b, keep=True)
        rep_op = np.where(occ_rep, 0.3 + 0.45 * t, 0.0).astype("f4")

        def accept(f: float) -> None:
            cutv.set(op=np.full(len(cutv), 1 - f, "f4"))
            arcs_b.opacity = 1 - f
            contact.set(op=np.full(len(contact), 1 - f, "f4"))
            cl_b.opacity = 1 - f

        self.voxel_transition(ica, tof_rgb, rep_op, run_time=T(1.0), hook=accept,
                              extra=[FadeOut(beta[1]), FadeIn(new_b)])
        say(f"opened: the centerline follows the curl, {sm['cl_after_mm']} mm", ADDED)
        self.animate_gl(lambda f: setattr(cl_a, "progress", f), run_time=T(1.4))
        self.hold(T(1.0))

        # back to the tree
        def swap_out(f: float) -> None:
            vox.set(op=(op0 * f).astype("f4"))
            ica.set(op=(rep_op * (1 - f)).astype("f4"))
            cl_a.opacity = 1 - f

        self.animate_gl(swap_out, run_time=T(1.2), extra=[FadeOut(badge), FadeOut(new_b)])
        self.gl.remove(ica, pre, cutv, ring, arcs_b, contact, cl_b, cl_a)
        self.orbit(ORBIT)
        return cap



if __name__ == "__main__":
    run_scene(__file__, "EicabFull", "eicab_full")
