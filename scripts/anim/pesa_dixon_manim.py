"""Manim animation of the PESA-Fat Dixon v5 pipeline (stages 1-3) on one subject.

Three Dixon MRI stations (neck and shoulders, abdomen, thighs), four images each (FAT, WATER, fat
fraction, T2*):

- stage 1: TotalSegmentator on each station (organs, paravertebral muscles, vertebrae on the
  FAT and WATER images, thigh muscles, the femur / hip bones);
- stage 2: what is measured: the liver from the WATER image, its vessels out (T2* > 50 ms), each
  kidney's renal pelvis out (convex hull minus the kidney), L3 / L4 eroded to the bone marrow,
  the femur subtracted from the quadriceps, and every mask one voxel in from its border (the
  ``*_VOL`` twins keep it for the volumes);
- stage 3: the means inside each mask: fat fraction, T2*, R2* = 1 / T2*, volume, and the liver's
  water fraction against the paravertebral muscles.

Every state comes from ``pesa_dixon_stages.npz`` (``pesa_dixon_precompute.py``, which rebuilds
stage 2 with its own functions and reproduces the stage-2 masks voxel for voxel); this file needs
numpy, moderngl and manim only. Every voxel is drawn at the Dixon grid (2 × 2 × 3 mm). No subject
identifier is shown.

Render (manim environment)::

    PD_FAST=1 python scripts/anim/pesa_dixon_manim.py -ql --fps 15
    python scripts/anim/pesa_dixon_manim.py -qh --fps 30              # -> pesa_dixon.mp4
    python scripts/anim/pesa_dixon_manim.py -qh --fps 30 --no-text    # -> pesa_dixon_notext.mp4
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anim_voxels import FONT, INK, INK_SOFT, hex_of, rgb, run_scene  # noqa: E402
from gl_render import Mesh, Quad, Voxels  # noqa: E402
from gl_scene import GLScene  # noqa: E402
import pesa_palette as pp  # noqa: E402

DATA_PATH = Path(os.environ.get("PD_STAGES", str(Path(__file__).with_name("pesa_dixon_stages.npz"))))
FAST = os.environ.get("PD_FAST", "0") == "1"
ORBIT = 2.2
SHIFT = (0.34, 0.06)
STATIONS = ("head", "thorax", "legs")

# one structure id per measured mask across the three stations
S_LIVER, S_PANC, S_KL, S_KR, S_PVML, S_PVMR, S_L3, S_L4, S_QL, S_QR, S_BONE = range(1, 12)
STRUCT = {S_LIVER: pp.LIVER, S_PANC: pp.PANCREAS, S_KL: pp.KIDNEY_L, S_KR: pp.KIDNEY_R, S_PVML: pp.PVM_L,
          S_PVMR: pp.PVM_R, S_L3: pp.L3, S_L4: pp.L4, S_QL: pp.QUAD_L, S_QR: pp.QUAD_R, S_BONE: pp.BONE}
NAMES = {S_LIVER: "liver", S_PANC: "pancreas", S_KL: "kidney L", S_KR: "kidney R", S_PVML: "paravertebral L",
         S_PVMR: "paravertebral R", S_L3: "L3 marrow", S_L4: "L4 marrow", S_QL: "quadriceps L",
         S_QR: "quadriceps R"}
# station label -> structure id
TO_S = {"head": {1: S_PVML, 2: S_PVMR},
        "thorax": {1: S_LIVER, 2: S_PANC, 3: S_KL, 4: S_KR, 5: S_PVML, 6: S_PVMR, 7: S_L3, 8: S_L4},
        "legs": {1: S_QL, 2: S_QR}}


class PesaDixon(GLScene):
    """PESA-Fat Dixon v5, stages 1-3."""

    def construct(self) -> None:
        data = np.load(DATA_PATH)
        meta = json.loads(str(data["meta"]))
        s3 = meta["stage3"]
        T = (lambda x: max(0.3, x * 0.35)) if FAST else (lambda x: x)

        # ---- the three stations as one voxel layer
        part, worlds, aff = {}, [], {}
        n0 = 0
        for s in STATIONS:
            a = np.asarray(data[f"{s}_affine"], float)
            v = np.asarray(data[f"{s}_vox"], float)
            aff[s] = a
            worlds.append(v @ a[:3, :3].T + a[:3, 3])
            part[s] = slice(n0, n0 + len(v))
            n0 += len(v)
        world = np.concatenate(worlds)
        N = len(world)
        sp = np.abs(np.diag(aff["thorax"])[:3])
        vox = Voxels(world, pitch=float(sp[0]), fill=0.93, edge_dark=0.16, aspect=sp / sp[0])

        def arr(s: str, key: str, dtype=int) -> np.ndarray:
            k = f"{s}_{key}"
            return np.asarray(data[k]).astype(dtype) if k in data.files else np.zeros(part[s].stop - part[s].start,
                                                                                        dtype)

        def struct_of(s: str, lab: np.ndarray) -> np.ndarray:
            out = np.zeros(len(lab), int)
            for k, v in TO_S[s].items():
                out[lab == k] = v
            return out

        def full(per: dict[str, np.ndarray]) -> np.ndarray:
            out = np.zeros(N, int)
            for s, a in per.items():
                out[part[s]] = a
            return out

        LUT = pp.lut(STRUCT)

        def look(lab: np.ndarray, ghost: np.ndarray | None = None, ghost_op: float = 0.12):
            """rgb, op for a structure-id array (0 hidden, or ghost where *ghost*)."""
            c = LUT[lab % 256].copy()
            op = (lab > 0).astype("f4")
            if ghost is not None:
                g = ghost & (lab == 0)
                c[g] = rgb(pp.GHOST)
                op[g] = ghost_op
            return c, op

        # stage-1 raw classes, one id per voxel and station
        raw = np.zeros(N, int)
        h = arr("head", "s1_total_mr")
        raw[part["head"]] = h
        t_tot, t_vert, t_wat = arr("thorax", "s1_total_mr"), arr("thorax", "s1_vertebrae_mr"), arr(
            "thorax", "s1_total_mr_water")
        raw[part["thorax"]] = np.where(t_tot > 0, t_tot, np.where(t_wat > 0, t_wat, np.where(t_vert > 0, t_vert + 120, 0)))
        l_mus, l_tot = arr("legs", "s1_thigh_shoulder_muscles_mr"), arr("legs", "s1_total_mr")
        raw[part["legs"]] = np.where(l_tot > 0, l_tot + 60, np.where(l_mus > 0, l_mus + 180, 0))
        raw_rgb = pp.categorical(raw)

        # stage-2 states (structure ids over the whole layer)
        th = {k: arr("thorax", k, bool) for k in ("liver_fat", "liver", "vessels", "kidney_left", "kidney_right",
                                                  "pelvis2_left", "pelvis2_right", "hull_left", "hull_right",
                                                  "vert_L3", "vert_L3_ero", "vert_L4", "vert_L4_ero", "bn_L3",
                                                  "bn_L4")}
        t_vol = arr("thorax", "final_vol")
        S1 = t_vol.copy()                       # WATER liver, kidneys whole, raw vertebrae
        S1[(S1 == 0) & th["kidney_left"]] = 3
        S1[(S1 == 0) & th["kidney_right"]] = 4
        S0 = S1.copy()                          # the FAT-contrast liver instead
        S0[(S0 == 1) & ~th["liver_fat"]] = 0
        S0[(S0 == 0) & th["liver_fat"]] = 1
        S2 = S1.copy()
        S2[th["vessels"] & (S2 == 1)] = 0
        S3 = S2.copy()
        S3[(S3 == 3) & th["pelvis2_left"]] = 0
        S3[(S3 == 4) & th["pelvis2_right"]] = 0
        S4a = S3.copy()
        S4a[(S4a == 7) & ~th["vert_L3_ero"]] = 0
        S4a[(S4a == 8) & ~th["vert_L4_ero"]] = 0
        S4 = S3.copy()
        S4[(S4 == 7) & ~th["bn_L3"]] = 0
        S4[(S4 == 8) & ~th["bn_L4"]] = 0
        pre_t = arr("thorax", "pre_erode")
        assert int((S4 != pre_t).sum()) < 50, int((S4 != pre_t).sum())
        S4 = pre_t

        head0 = arr("head", "s1_total_mr")
        head_raw = np.zeros_like(head0)
        head_raw[head0 > 0] = 1 + (np.unique(head0[head0 > 0], return_inverse=True)[1])  # autochthon L / R
        head_raw = np.where(arr("head", "final_vol") > 0, arr("head", "final_vol"), head_raw)
        legs_q0 = arr("legs", "quads_raw")
        skel, skel_dil = arr("legs", "skeleton", bool), arr("legs", "skeleton_dil", bool)

        kept0 = full({"head": struct_of("head", head_raw), "thorax": struct_of("thorax", S0),
                      "legs": struct_of("legs", legs_q0)})
        fin = full({s: struct_of(s, arr(s, "final")) for s in STATIONS})
        fin_vol = full({s: struct_of(s, arr(s, "final_vol")) for s in STATIONS})
        pre = full({"head": struct_of("head", arr("head", "final_vol")), "thorax": struct_of("thorax", S4),
                    "legs": struct_of("legs", arr("legs", "pre_erode"))})

        def th_state(a: np.ndarray) -> np.ndarray:
            out = kept0.copy()
            out[part["thorax"]] = struct_of("thorax", a)
            return out

        # ---- slices and skin
        quads, tex = {}, {}
        for s in STATIONS:
            a = aff[s]
            shp = meta["stations"][s.upper()]["shape"]
            y = a[1, 1] * meta["stations"][s.upper()]["slice_y"] + a[1, 3]
            x0, x1 = a[0, 3] - 0.5 * a[0, 0], a[0, 3] + (shp[0] - 0.5) * a[0, 0]
            z0, z1 = a[2, 3] - 0.5 * a[2, 2], a[2, 3] + (shp[2] - 0.5) * a[2, 2]
            corners = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]])
            tex[s] = {}
            for k in ("FAT", "WATER", "FF", "T2"):
                g = np.asarray(data[f"{s}_tex_{k}"]).transpose(1, 0, 2)
                rgba = np.zeros(g.shape[:2] + (4,), np.uint8)
                rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g[..., 0]
                rgba[..., 3] = g[..., 1]
                tex[s][k] = rgba
            quads[s] = Quad(corners, tex[s]["FAT"], opacity=0.0)
        skins = [Mesh(Mesh.from_triangles(np.asarray(data[f"{s}_body_v"], float), np.asarray(data[f"{s}_body_f"]),
                                          color=rgb(pp.SKIN)), opacity=0.0, rim=0.55, amb=0.6, spec=0.05)
                 for s in STATIONS]
        self.gl.add(*skins, *quads.values(), vox)

        allw = world
        z_lo, z_hi = float(allw[:, 2].min()), float(allw[:, 2].max())
        body_c = np.array([0.5 * (allw[:, 0].min() + allw[:, 0].max()), float(np.median(allw[:, 1])),
                           0.5 * (z_lo + z_hi)])
        fov = 24.0

        def dist_for(height: float) -> float:
            return height / 2 / np.tan(np.radians(fov / 2)) * 1.12

        def centre_of(mask: np.ndarray) -> np.ndarray:
            w = world[mask]
            return 0.5 * (w.min(0) + w.max(0))

        self.set_camera(target=body_c, distance=dist_for(z_hi - z_lo + 90), phi=84, theta=90, fov=fov, shift=SHIFT)
        self.orbit(ORBIT)

        def legend(ids: list[int], corner=UL) -> VGroup:
            items = [VGroup(Rectangle(width=0.16, height=0.16, fill_color=STRUCT[i], fill_opacity=1, stroke_width=0),
                            Text(NAMES[i], font=FONT, color=INK, font_size=13)).arrange(RIGHT, buff=0.08)
                     for i in ids]
            g = VGroup(*items).arrange(DOWN, buff=0.06, aligned_edge=LEFT).to_corner(corner, buff=0.35)
            g.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.1)
            self.fixed(g, keep=True)
            return g

        # ------------------------------------------------------------ title + the stations
        title = self.title_card("PESA-Fat · Dixon MRI",
                                "fat fraction, T2* and volume of organs, muscles and bone marrow",
                                "nvitk.pipes.pesa_fat.dixon_v5 · stages 1-3")

        def show_slices(f: float, op: float = 0.92) -> None:
            for q in quads.values():
                q.opacity = op * f
            for m in skins:
                m.opacity = 0.16 * f

        self.play(FadeIn(title), run_time=T(0.8))
        self.animate_gl(show_slices, run_time=T(2.0))
        self.hold(T(1.2))

        cap = self.caption("Three stations, four images each", [
            ("neck and shoulders · abdomen · thighs, scanned one after the other", INK, 20),
            ("two-point Dixon: a FAT and a WATER image; fat fraction = F / (F + W); T2*", INK_SOFT, 18),
        ])
        self.swap(title, cap)
        names = {"FAT": "FAT", "WATER": "WATER", "FF": "fat fraction", "T2": "T2*"}
        tag = None
        for k in ("FAT", "WATER", "FF", "T2"):
            for s in STATIONS:
                quads[s].set_image(tex[s][k])
            new = Text(names[k], font=FONT, weight="BOLD", color=INK, font_size=22).to_corner(UL, buff=0.5)
            new.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.1)
            self.fixed(new)
            if tag is None:
                self.play(FadeIn(new), run_time=T(0.4))
            else:
                self.play(FadeOut(tag), FadeIn(new), run_time=T(0.4))
            tag = new
            self.hold(T(1.3))
        for s in STATIONS:
            quads[s].set_image(tex[s]["FAT"])
        self.play(FadeOut(tag), run_time=T(0.3))

        # ------------------------------------------------------------ stage 1
        cap1 = self.caption("Stage 1 · TotalSegmentator on every station", [
            ("organs and paravertebral muscles on FAT, the liver again on WATER, the vertebrae on WATER", INK, 18),
            ("the thigh muscles and the femur / hip bones on the thighs", INK_SOFT, 18),
        ])
        self.swap(cap, cap1)
        z = world[:, 2]
        dz = (z.max() - z)  # top to bottom
        vox.set(rgb=raw_rgb, op=np.zeros(N, "f4"))
        self.voxel_transition(vox, op=(raw > 0).astype("f4"), delay=dz, lag=0.6, run_time=T(3.0),
                              extra=[], hook=lambda f: show_slices(1.0, 0.92 - 0.6 * f))
        self.hold(T(1.4))

        cap2 = self.caption("Stage 2 · keep what is measured", [
            ("liver · pancreas · kidneys · paravertebral muscles · L3, L4 · quadriceps", INK, 20),
            ("each structure keeps its biggest connected piece", INK_SOFT, 18),
        ])
        self.swap(cap1, cap2)
        c, op = look(kept0)
        leg_all = legend([S_LIVER, S_PANC, S_KL, S_KR, S_PVML, S_PVMR, S_L3, S_L4, S_QL, S_QR])
        self.voxel_transition(vox, rgb_=c, op=op, run_time=T(2.2), extra=[FadeIn(leg_all)])
        self.hold(T(1.2))

        # ------------------------------------------------------------ THORAX
        tmask = np.zeros(N, bool)
        tmask[part["thorax"]] = kept0[part["thorax"]] > 0
        t_c = centre_of(tmask)
        self.move_camera(run_time=T(2.2), target=t_c, distance=dist_for(380), phi=80,
                         extra=[FadeOut(leg_all)])
        capL = self.caption("Abdomen · the liver from the WATER image", [
            ("on FAT the fatty liver blends with the fat around it; WATER outlines it", INK, 18),
            ("green: gained over the FAT-image liver · red: lost", INK_SOFT, 18),
        ])
        self.swap(cap2, capL)
        quads["thorax"].set_image(tex["thorax"]["WATER"])
        a_now = th_state(S0)
        a_new = th_state(S1)
        gain = (a_new == S_LIVER) & (a_now != S_LIVER)
        lose = (a_now == S_LIVER) & (a_new != S_LIVER)
        c0, op0 = look(a_now)
        c0[gain] = rgb(pp.ADDED)
        op_g = op0.copy()
        op_g[gain] = 1.0
        c0[lose] = rgb(pp.REMOVED)
        self.voxel_transition(vox, rgb_=c0, op=op_g, run_time=T(1.8))
        self.hold(T(0.8))
        c1, op1 = look(a_new)
        self.voxel_transition(vox, rgb_=c1, op=op1, run_time=T(1.4))

        capV = self.caption("The liver's vessels out", [
            (f"liver voxels with T2* above {meta['liver_vessel_t2star_ms']:g} ms are vessels, not tissue", INK, 18),
            ("left out of FF / T2* / R2*; the liver volume keeps them", INK_SOFT, 18),
        ])
        self.swap(capL, capV)
        quads["thorax"].set_image(tex["thorax"]["T2"])
        ves = np.zeros(N, bool)
        ves[part["thorax"]] = th["vessels"]
        cl, opl = look(th_state(S1))
        cl[ves] = rgb(pp.REMOVED)
        opl[(th_state(S1) == S_LIVER)] = np.where(ves, 1.0, 0.18)[th_state(S1) == S_LIVER]
        size = np.where(ves, 2.0, 1.0).astype("f4")
        self.voxel_transition(vox, rgb_=cl, op=opl, size=size, run_time=T(1.6))
        self.hold(T(1.0))
        c2, op2 = look(th_state(S2))
        self.voxel_transition(vox, rgb_=c2, op=op2, size=np.ones(N, "f4"), run_time=T(1.2))

        # kidneys
        kmask = np.zeros(N, bool)
        kmask[part["thorax"]] = th["kidney_left"] | th["kidney_right"]
        self.move_camera(run_time=T(1.8), target=centre_of(kmask), distance=dist_for(170), phi=74)
        capK = self.caption("Kidneys · the renal pelvis out", [
            ("convex hull of each kidney minus the kidney: the pelvis, its biggest piece", INK, 18),
            ("grown by two voxels and cleared, so the urine and fat at the hilum stay out", INK_SOFT, 18),
        ])
        self.swap(capV, capK)
        quads["thorax"].set_image(tex["thorax"]["FAT"])
        hull = np.zeros(N, bool)
        hull[part["thorax"]] = (th["hull_left"] | th["hull_right"]) & ~(th["kidney_left"] | th["kidney_right"])
        pel = np.zeros(N, bool)
        pel[part["thorax"]] = th["pelvis2_left"] | th["pelvis2_right"]
        cur = th_state(S2)
        ch, oph = look(cur)
        ch[hull & (cur == 0)] = rgb(pp.GHOST)
        oph[hull & (cur == 0)] = 0.32
        self.voxel_transition(vox, rgb_=ch, op=oph, run_time=T(1.5))
        cp_, opp = ch.copy(), oph.copy()
        cp_[pel & ((cur == S_KL) | (cur == S_KR) | (cur == 0))] = rgb(pp.REMOVED)
        opp[pel & ((cur == S_KL) | (cur == S_KR))] = 1.0
        opp[pel & (cur == 0)] = 0.5
        self.voxel_transition(vox, rgb_=cp_, op=opp, run_time=T(1.4))
        self.hold(T(0.8))
        c3, op3 = look(th_state(S3))
        self.voxel_transition(vox, rgb_=c3, op=op3, run_time=T(1.4))

        # vertebrae
        vmask = np.zeros(N, bool)
        vmask[part["thorax"]] = th["vert_L3"] | th["vert_L4"]
        self.move_camera(run_time=T(1.6), target=centre_of(vmask), distance=dist_for(140), phi=76,
                         theta=self.cam.theta - 50)
        capB = self.caption("L3 and L4 · the bone marrow", [
            ("each vertebra eroded by three voxels, its biggest piece kept", INK, 18),
            ("the cortex and the discs stay out of the marrow's fat fraction", INK_SOFT, 18),
        ])
        self.swap(capK, capB)
        ca, opa = look(th_state(S4a))
        self.voxel_transition(vox, rgb_=ca, op=opa, run_time=T(1.6))
        c4, op4 = look(th_state(S4))
        self.voxel_transition(vox, rgb_=c4, op=op4, run_time=T(1.0))
        self.hold(T(0.8))

        # ------------------------------------------------------------ LEGS
        lmask = np.zeros(N, bool)
        lmask[part["legs"]] = legs_q0 > 0
        self.move_camera(run_time=T(2.4), target=centre_of(lmask), distance=dist_for(360), phi=80,
                         theta=self.cam.theta + 50)
        capQ = self.caption("Thighs · the femur out of the quadriceps", [
            ("the femur and hip bones from TotalSegmentator, the femurs grown by three voxels", INK, 18),
            ("subtracted, so the bone marrow's fat stays out of the muscle", INK_SOFT, 18),
        ])
        self.swap(capB, capQ)
        lsk = np.zeros(N, bool)
        lsk[part["legs"]] = skel
        ldl = np.zeros(N, bool)
        ldl[part["legs"]] = skel_dil
        cur = th_state(S4)
        lq = np.zeros(N, bool)
        lq[part["legs"]] = legs_q0 > 0
        # a coronal cut through the femurs: what is in front of it hidden, the overlap on the cut face
        lfront = np.zeros(N, bool)
        lfront[part["legs"]] = world[part["legs"], 1] > float(np.median(world[lsk, 1]))
        cq, opq = look(cur)
        opq[lfront & lq] = 0.0
        cq[lsk] = rgb(pp.BONE)
        opq[lsk & ~lfront] = 1.0
        self.voxel_transition(vox, rgb_=cq, op=opq, run_time=T(1.6),
                              hook=lambda f: setattr(quads["legs"], "opacity", 0.32 * (1 - f)))
        hit = ldl & lq
        ring = ldl & ~lsk & ~lq
        cq2, opq2 = cq.copy(), opq.copy()
        cq2[hit | ring] = rgb(pp.REMOVED)
        opq2[ring & ~lfront] = 0.4
        opq2[hit & ~lfront] = 1.0
        self.voxel_transition(vox, rgb_=cq2, op=opq2, run_time=T(1.6))
        self.hold(T(1.0))
        cq3, opq3 = look(pre)
        cq3[lsk] = rgb(pp.BONE)
        opq3[lfront & lq] = 0.0
        opq3[lsk & ~lfront] = 1.0
        self.voxel_transition(vox, rgb_=cq3, op=opq3, run_time=T(1.4))
        self.hold(T(0.6))
        c5, op5 = look(pre)
        self.voxel_transition(vox, rgb_=c5, op=op5, run_time=T(1.2),
                              hook=lambda f: setattr(quads["legs"], "opacity", 0.32 * f))

        # ------------------------------------------------------------ the measurement masks
        self.move_camera(run_time=T(2.6), target=body_c, distance=dist_for(z_hi - z_lo + 90), phi=84,
                         theta=self.cam.theta - 20)
        capE = self.caption("Measurement masks: one voxel in from every border", [
            ("red: eroded, read by FF / T2* / R2* (clear of partial volume at the border)", pp.MEASURE, 18),
            ("blue: the uneroded twin, read by the volume", pp.VOLUME, 18),
        ])
        self.swap(capQ, capE)
        ring_e = (fin_vol > 0) & (fin == 0)
        ce = np.tile(rgb(pp.MEASURE), (N, 1)).astype("f4")
        ce[ring_e] = rgb(pp.VOLUME)
        ope = np.where(fin > 0, 1.0, np.where(ring_e, 0.35, 0.0)).astype("f4")
        self.voxel_transition(vox, rgb_=ce, op=ope, run_time=T(1.8), extra=[])
        self.hold(T(1.6))

        # ------------------------------------------------------------ stage 3
        ffv = np.zeros(N, float)
        t2v = np.zeros(N, float)
        for s in STATIONS:
            ffv[part[s]] = np.asarray(data[f"{s}_ff"], float)
            t2v[part[s]] = np.asarray(data[f"{s}_t2"], float)
        capF = self.caption("Stage 3 · the fat fraction inside every mask", [
            ("mean FF, T2*, R2* = 1 / T2* per structure, volume from the blue twin", INK, 18),
            (f"liver water fraction: liver WATER / paravertebral WATER = {s3['DIXON_LIVER_WF']:.0f} %",
             INK_SOFT, 18),
        ])
        self.swap(capE, capF)
        cf = pp.ramp(ffv / 70.0, pp.FF_STOPS)
        opf = (fin > 0).astype("f4")
        bar = self._colorbar(pp.FF_STOPS, "fat fraction (%)", 0, 70)
        self.voxel_transition(vox, rgb_=cf, op=opf, delay=dz, lag=0.5, run_time=T(2.6), extra=[FadeIn(bar)])
        self.hold(T(1.0))

        rows = [("liver", "LIVER", pp.LIVER), ("pancreas", "PANCREAS", pp.PANCREAS),
                ("kidneys", "KIDNEY_LR", pp.KIDNEY_L), ("paravertebral", "T_PVM_LR", pp.PVM_L),
                ("L3 marrow", "BN_L3", pp.L3), ("L4 marrow", "BN_L4", pp.L4), ("quadriceps", "L_QM_LR", pp.QUAD_L)]
        chart = self._ff_chart(rows, s3)
        self.play(FadeOut(bar), FadeIn(chart, shift=RIGHT * 0.2), run_time=T(0.8))
        self.hold(T(4.0))

        end = self.caption("What stage 3 writes", [
            ("per structure: VOL · FF · T2* · R2* · slices · liver WF", INK, 20),
            ("one row per subject; the batch joins them into its summary codebook", INK_SOFT, 18),
        ])
        self.swap(capF, end)
        self.orbit(ORBIT * 1.6)
        self.hold(T(3.0))
        self.play(FadeOut(end), FadeOut(chart), run_time=T(0.8))
        self.hold(T(0.4))

    def _colorbar(self, stops, label: str, lo: float, hi: float) -> VGroup:
        n = 40
        cs = pp.ramp(np.linspace(0, 1, n), stops)
        cells = VGroup(*[Rectangle(width=0.06, height=0.18, fill_color=hex_of(c), fill_opacity=1, stroke_width=0)
                         for c in cs]).arrange(RIGHT, buff=0)
        lo_t = Text(f"{lo:g}", font=FONT, color=INK_SOFT, font_size=13).next_to(cells, LEFT, buff=0.1)
        hi_t = Text(f"{hi:g}", font=FONT, color=INK_SOFT, font_size=13).next_to(cells, RIGHT, buff=0.1)
        lab = Text(label, font=FONT, color=INK, font_size=14).next_to(cells, UP, buff=0.1)
        g = VGroup(cells, lo_t, hi_t, lab).to_corner(UL, buff=0.5)
        g.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.12)
        self.fixed(g, keep=True)
        return g

    def _ff_chart(self, rows, s3: dict) -> VGroup:
        """Horizontal bars of the mean fat fraction per structure, with T2* and volume beside."""
        vmax = 70.0
        width = 2.6
        head = VGroup(Text("fat fraction (%)", font=FONT, weight="BOLD", color=INK, font_size=14),
                      Text("T2* (ms)", font=FONT, weight="BOLD", color=INK, font_size=14),
                      Text("volume (cc)", font=FONT, weight="BOLD", color=INK, font_size=14))
        lines = VGroup()
        for name, key, col in rows:
            ff = float(s3[f"DIXON_{key}_FF"])
            t2 = float(s3[f"DIXON_{key}_T2"])
            vol = float(s3[f"DIXON_{key}_VOL"])
            nm = Text(name, font=FONT, color=INK, font_size=13)
            b = Rectangle(width=max(width * ff / vmax, 0.02), height=0.17, fill_color=col, fill_opacity=1,
                          stroke_width=0)
            v = Text(f"{ff:.1f}", font=FONT, color=INK_SOFT, font_size=12)
            t = Text(f"{t2:.1f}", font=FONT, color=INK_SOFT, font_size=12)
            vv = Text(f"{vol:.0f}", font=FONT, color=INK_SOFT, font_size=12)
            lines.add(VGroup(nm, b, v, t, vv))
        # columns
        x_name, x_bar, x_t2, x_vol = 0.0, 1.55, 1.55 + width + 0.95, 1.55 + width + 2.15
        for i, g in enumerate(lines):
            y = -0.3 * i
            nm, b, v, t, vv = g
            nm.move_to([x_name, y, 0], aligned_edge=LEFT)
            b.move_to([x_bar, y, 0], aligned_edge=LEFT)
            v.next_to(b, RIGHT, buff=0.08)
            t.move_to([x_t2, y, 0])
            vv.move_to([x_vol, y, 0])
        h0, h1, h2 = head
        h0.move_to([x_bar, 0.34, 0], aligned_edge=LEFT)
        h1.move_to([x_t2, 0.34, 0])
        h2.move_to([x_vol, 0.34, 0])
        g = VGroup(head, lines).to_corner(UL, buff=0.5)
        g.add_background_rectangle(color="#FFFFFF", opacity=0.9, buff=0.16)
        self.fixed(g, keep=True)
        return g


if __name__ == "__main__":
    run_scene(__file__, "PesaDixon", "pesa_dixon")
