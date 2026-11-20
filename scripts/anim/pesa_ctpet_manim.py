"""Manim animation of the PESA-Fat CT-PET v5 pipeline (stages 1-3) on one subject.

One whole-body CT and its FDG PET:

- stage 1: TotalSegmentator on the CT (``total``: organs and bones; ``tissue_4_types``:
  subcutaneous, torso and intermuscular fat; ``thigh_shoulder_muscles``; ``body``);
- stage 2: the fat (visceral and subcutaneous): arms and legs out of the subcutaneous fat, the
  organ interiors cleared with a safety margin (the bladder redrawn from its PET uptake), the
  ureter found on the PET and cleared, the L3-L4 slab; the intermuscular fat split into legs and
  trunk at the quadriceps; the organs and the hemisphere-split muscles (deltoid split left /
  right, the skeleton subtracted); the L3 / L4 bone marrow (eroded, and a density twin below
  400 HU for its volume); every SUV mask eroded against the motion between CT and PET; the
  quadriceps cut where the lowest PET bed leaves them too noisy (the last-bed trim);
- stage 3: SUV (bw) mean / max inside every mask, volumes from the uneroded twins.

Every state comes from ``pesa_ctpet_stages.npz`` (``pesa_ctpet_precompute.py``, which runs the
real stage-2 worker with recorders on its steps); this file needs numpy, moderngl and manim only.
The CT grid is pooled 3 × 3 × 2 (3.5 × 3.5 × 4 mm) for the voxels; the bone marrow is drawn at the
CT grid. No subject identifier is shown.

Render (manim environment)::

    PC_FAST=1 python scripts/anim/pesa_ctpet_manim.py -ql --fps 15
    python scripts/anim/pesa_ctpet_manim.py -qh --fps 30              # -> pesa_ctpet.mp4
    python scripts/anim/pesa_ctpet_manim.py -qh --fps 30 --no-text    # -> pesa_ctpet_notext.mp4
"""

from __future__ import annotations

import json
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UL, UP, UR, DashedLine, Dot, FadeIn, FadeOut, Line, Rectangle, Text,
                   VGroup, VMobject)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anim_voxels import FONT, INK, INK_SOFT, hex_of, rgb, run_scene  # noqa: E402
from gl_render import Mesh, Quad, VolumeMIP, Voxels  # noqa: E402
from gl_scene import GLScene  # noqa: E402
import pesa_palette as pp  # noqa: E402

DATA_PATH = Path(os.environ.get("PC_STAGES", str(Path(__file__).with_name("pesa_ctpet_stages.npz"))))
FAST = os.environ.get("PC_FAST", "0") == "1"
ORBIT = 2.2
ORIGIN_L = np.zeros(3)
CUT = "#E8892B"   # the last-bed cut plane and the slices it takes out of the SUV mask
SHIFT = (0.34, 0.06)

FAT_T = {1: pp.FAT_V, 2: pp.FAT_SC}
IM_T = {1: pp.IM_LEGS, 2: pp.IM_TRUNK}
ORG_T = {1: pp.LIVER, 2: pp.SPLEEN, 3: pp.PANCREAS}
MUS_T = {1: pp.QUAD_L, 2: pp.QUAD_R, 3: pp.PVM_L, 4: pp.PVM_R, 5: pp.DELT_L, 6: pp.DELT_R, 7: pp.TRAP}
MO_T = {1: pp.L4, 2: pp.L3}



def solid(color: str, alpha: float, n: int = 8) -> np.ndarray:
    out = np.zeros((n, n, 4), np.uint8)
    out[..., :3] = (rgb(color) * 255).astype(np.uint8)
    out[..., 3] = int(alpha * 255)
    return out


class PesaCtPet(GLScene):
    """PESA-Fat CT-PET v5, stages 1-3."""

    def construct(self) -> None:
        data = np.load(DATA_PATH)
        meta = json.loads(str(data["meta"]))
        s3 = meta["stage3"]
        T = (lambda x: max(0.3, x * 0.35)) if FAST else (lambda x: x)

        def A(k: str, dtype=int) -> np.ndarray:
            return np.asarray(data[k]).astype(dtype)

        paff = np.asarray(data["affine"], float)
        world = A("vox", float) @ paff[:3, :3].T + paff[:3, 3]
        N = len(world)
        sp = np.abs(np.diag(paff)[:3])
        vox = Voxels(world, pitch=float(sp[0]), fill=0.94, edge_dark=0.14, aspect=sp / sp[0])
        caff = np.asarray(data["ct_affine"], float)
        mw = A("mo_vox", float) @ caff[:3, :3].T + caff[:3, 3]
        csp = np.abs(np.diag(caff)[:3])
        movox = Voxels(mw, pitch=float(csp[0]), fill=0.92, edge_dark=0.2, aspect=csp / csp[0])
        movox.set(op=np.zeros(len(mw), "f4"))

        bone = VolumeMIP(np.asarray(data["bone8"]), np.asarray(data["bone8_affine"], float), color=(0.32, 0.34, 0.4),
                         opacity=0.0, lo=0.05, hi=0.7, gamma=1.2)
        pet = VolumeMIP(np.asarray(data["pet8"]), np.asarray(data["pet8_affine"], float), color=(0.12, 0.08, 0.16),
                        opacity=0.0, lo=0.08, hi=0.6, gamma=1.0)
        skin = Mesh(Mesh.from_triangles(np.asarray(data["body_v"], float), np.asarray(data["body_f"]),
                                        color=rgb(pp.SKIN)), opacity=0.0, rim=0.55, amb=0.6, spec=0.05)
        skel = Mesh(Mesh.from_triangles(np.asarray(data["skel_v"], float), np.asarray(data["skel_f"]),
                                        color=rgb(pp.BONE)), opacity=0.0, amb=0.55, spec=0.1)
        self.gl.add(pet, bone, skin, skel, vox, movox)

        fov = 24.0
        z_lo, z_hi = float(world[:, 2].min()), float(world[:, 2].max())
        body_c = np.array([0.5 * (world[:, 0].min() + world[:, 0].max()), float(np.median(world[:, 1])),
                           0.5 * (z_lo + z_hi)])

        def dist_for(height: float) -> float:
            return height / 2 / np.tan(np.radians(fov / 2)) * 1.12

        def centre_of(mask: np.ndarray, w: np.ndarray = world) -> np.ndarray:
            v = w[mask]
            return 0.5 * (v.min(0) + v.max(0))

        self.set_camera(target=body_c, distance=dist_for(z_hi - z_lo + 80), phi=86, theta=90, fov=fov, shift=SHIFT)
        self.orbit(ORBIT)

        # ---- painting helpers
        def paint(*layers, ghost: np.ndarray | None = None, ghost_op: float = 0.1):
            """rgb, op from (labels, table[, op]) layers, later ones on top."""
            c = np.tile(rgb(pp.GHOST), (N, 1)).astype("f4")
            op = np.zeros(N, "f4")
            if ghost is not None:
                op[ghost] = ghost_op
            for layer in layers:
                lab, table = layer[0], layer[1]
                o = layer[2] if len(layer) > 2 else 1.0
                m = lab > 0
                c[m] = pp.lut(table)[lab[m] % 256]
                op[m] = o
            return c, op

        def legend(items: list[tuple[str, str]]) -> VGroup:
            rows = [VGroup(Rectangle(width=0.16, height=0.16, fill_color=c, fill_opacity=1, stroke_width=0),
                           Text(n, font=FONT, color=INK, font_size=13)).arrange(RIGHT, buff=0.08) for c, n in items]
            g = VGroup(*rows).arrange(DOWN, buff=0.06, aligned_edge=LEFT).to_corner(UL, buff=0.35)
            g.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.1)
            self.fixed(g, keep=True)
            return g

        fat_raw, fat_noext, ext = A("fat_raw"), A("fat_noext"), A("extremity_fat")
        org_raw, org_dil = A("organs_raw") > 0, A("organs_dil") > 0
        bl_pet, uret = A("bladder_pet") > 0, A("ureter") > 0
        fat, fat_b = A("fat"), A("fat_batch")
        im_raw, im_pre, fat_im = A("im_raw"), A("im_pre"), A("fat_im")
        org_vol, org = A("organs_vol"), A("organs")
        mus_raw, mus_vol, mus = A("mus_raw"), A("mus_vol"), A("muscles")
        mus_pre = A("muscles_pretrim")
        delt = A("deltoid") > 0
        suv = A("suv", float)
        z = world[:, 2]
        dz = z.max() - z

        # ------------------------------------------------------------ title + CT / PET
        title = self.title_card("PESA-Fat · CT-PET", "fat depots, organs, muscles and bone marrow: volume and FDG uptake",
                                "nvitk.pipes.pesa_fat.ct_pet_v5 · stages 1-3")
        self.play(FadeIn(title), run_time=T(0.8))

        def ctx(f: float, b: float = 0.85, s: float = 0.14) -> None:
            bone.opacity = b * f
            skin.opacity = s * f

        self.animate_gl(ctx, run_time=T(2.0))
        cap = self.caption("One CT and one PET, head to thighs", [
            (f"CT: the anatomy at {meta['ct_spacing'][0]:.2f} × {meta['ct_spacing'][1]:.2f} × "
             f"{meta['ct_spacing'][2]:.0f} mm", INK, 20),
            ("PET: FDG uptake as SUV (dark = high), resampled onto every CT mask", INK_SOFT, 18),
        ])
        self.swap(title, cap)
        self.animate_gl(lambda f: setattr(pet, "opacity", 0.9 * f), run_time=T(1.8))
        self.hold(T(2.0))

        # ------------------------------------------------------------ stage 1
        cap1 = self.caption("Stage 1 · TotalSegmentator on the CT", [
            ("total: organs, bones, vessels · tissue_4_types: subcutaneous, torso and intermuscular fat", INK, 18),
            ("thigh_shoulder_muscles: quadriceps, deltoids, trapezius · body: trunk and extremities", INK_SOFT, 18),
        ])
        self.swap(cap, cap1)
        s1t, s1f, s1m = A("s1_total"), A("s1_tissue"), A("s1_muscles")
        raw = np.where(s1t > 0, s1t, np.where(s1m > 0, s1m + 140, np.where((s1f > 0) & (s1f != 3), s1f + 200, 0)))
        c_raw = pp.categorical(raw)
        fat_cls = (raw == 201) | (raw == 202) | (raw == 204)
        c_raw[raw == 201] = rgb(pp.FAT_SC)   # tissue_4_types: 1 subcutaneous, 2 torso, 4 intermuscular
        c_raw[raw == 202] = rgb(pp.FAT_V)
        c_raw[raw == 204] = rgb(pp.IM_TRUNK)
        vox.set(rgb=c_raw, op=np.zeros(N, "f4"))
        op_raw = np.where(raw > 0, np.where(fat_cls, 0.12, 1.0), 0.0).astype("f4")
        self.voxel_transition(vox, op=op_raw, delay=dz, lag=0.6, run_time=T(3.2),
                              hook=lambda f: (setattr(pet, "opacity", 0.9 * (1 - f)), setattr(bone, "opacity",
                                                                                               0.85 * (1 - f))))
        self.hold(T(1.6))

        # ------------------------------------------------------------ fat
        cap2 = self.caption("Stage 2 · visceral and subcutaneous fat", [
            ("the torso fat (visceral) and the subcutaneous fat from tissue_4_types", INK, 18),
            ("red: the arms and legs, out of the subcutaneous fat", pp.REMOVED, 18),
        ])
        self.swap(cap1, cap2)
        leg_f = legend([(pp.FAT_V, "visceral fat"), (pp.FAT_SC, "subcutaneous fat")])
        def fat_look(lab, v_op=1.0, sc_op=0.2):
            c_, op_ = paint((lab, FAT_T))
            op_[lab == 2] = sc_op
            op_[lab == 1] = v_op
            return c_, op_

        c, op = fat_look(fat_raw)
        self.voxel_transition(vox, rgb_=c, op=op, run_time=T(2.0), extra=[FadeIn(leg_f)])
        self.hold(T(0.6))
        c[ext > 0] = rgb(pp.REMOVED)
        op = op.copy()
        op[ext > 0] = 0.6
        self.voxel_transition(vox, rgb_=c, op=op, run_time=T(1.2))
        self.hold(T(0.6))
        c, op = fat_look(fat_noext)
        self.voxel_transition(vox, rgb_=c, op=op, run_time=T(1.2))

        trunk_c = centre_of(fat_noext == 1)
        self.move_camera(run_time=T(2.0), target=trunk_c, distance=dist_for(480), phi=80, shift=(0.3, 0.06))
        capO = self.caption("The organs cleared, with a margin", [
            ("liver, kidneys (their convex hulls), bowel, colon and bladder: their insides are not fat", INK, 18),
            ("grown by a margin (kidneys 7, bladder 10 voxels) for the SUV masks; the bladder redrawn from "
             "its PET uptake", INK_SOFT, 16),
        ])
        self.swap(cap2, capO)
        # a coronal cut through the organs: everything in front of it hidden
        front = world[:, 1] > float(np.median(world[org_raw, 1]))
        c, op = fat_look(fat_noext, v_op=1.0, sc_op=1.0)
        op[front] = 0.0
        self.voxel_transition(vox, rgb_=c, op=op, run_time=T(1.4))
        c[org_raw] = rgb(pp.GHOST)
        op[org_raw & ~front] = 1.0
        ring = org_dil & ~org_raw
        c[ring] = rgb(pp.REMOVED)
        op[ring & ~front] = 1.0
        self.voxel_transition(vox, rgb_=c, op=op, run_time=T(1.6))
        self.hold(T(1.0))
        c, op = fat_look(fat, v_op=1.0, sc_op=1.0)
        op[front] = 0.0
        self.voxel_transition(vox, rgb_=c, op=op, run_time=T(1.4))
        self.hold(T(0.6))

        capU = self.caption("The ureter, from the PET", [
            ("excreted FDG makes the ureter bright: traced from each kidney to the bladder", INK, 18),
            ("cleared from the visceral fat of the L3-L4 slab", INK_SOFT, 18),
        ])
        self.swap(capO, capU)
        cu = c.copy()
        opu = np.zeros(N, "f4")
        cu[org_raw] = rgb(pp.GHOST)
        opu[org_raw] = 0.1
        cu[uret] = rgb(pp.URETER)
        opu[uret] = 1.0
        cu[bl_pet] = rgb(pp.BLADDER)
        opu[bl_pet] = 1.0
        sz = np.where(uret, 1.6, 1.0).astype("f4")
        self.voxel_transition(vox, rgb_=cu, op=opu, size=sz, run_time=T(1.6),
                              extra=[], hook=lambda f: setattr(pet, "opacity", 0.55 * f))
        self.hold(T(1.4))

        zb = meta["l3l4_z_world"]
        capB = self.caption("The L3-L4 slab", [
            ("the same fat between the top of L3 and the bottom of L4: the batch depots", INK, 18),
            (f"visceral {s3['GRASA_V_BATCH_VOL']:.0f} cc · subcutaneous {s3['GRASA_SC_BATCH_VOL']:.0f} cc in the slab",
             INK_SOFT, 18),
        ])
        self.swap(capU, capB)
        planes = []
        x0, x1 = world[:, 0].min() - 20, world[:, 0].max() + 20
        y0, y1 = world[:, 1].min() - 20, world[:, 1].max() + 20
        for zz in zb:
            q = Quad(np.array([[x0, y0, zz], [x1, y0, zz], [x1, y1, zz], [x0, y1, zz]]), solid(pp.L3, 0.22), opacity=0.0)
            planes.append(q)
        self.gl.add(*planes)
        cb, opb = paint((fat, FAT_T, 0.08), (fat_b, FAT_T, 1.0))
        self.voxel_transition(vox, rgb_=cb, op=opb, size=np.ones(N, "f4"), run_time=T(1.8),
                              hook=lambda f: [setattr(q, "opacity", f) for q in planes] and None,
                              extra=[])
        self.animate_gl(lambda f: setattr(pet, "opacity", 0.55 * (1 - f)), run_time=T(0.4))
        self.hold(T(1.2))

        # ------------------------------------------------------------ intermuscular fat
        self.move_camera(run_time=T(2.2), target=body_c, distance=dist_for(z_hi - z_lo + 80), phi=84,
                         shift=SHIFT, extra=[FadeOut(leg_f)])
        capI = self.caption("Intermuscular fat · legs or trunk", [
            ("legs: the slices the quadriceps span, plus every fat piece touching them", INK, 18),
            ("trunk: the rest above; arms beside the trunk go with it; then each voxel takes its "
             "neighbours' majority", INK_SOFT, 16),
        ])
        self.swap(capB, capI)
        qz = meta["quads_z_world"]
        for q, zz in zip(planes, qz):
            q.set_corners(np.array([[x0, y0, zz], [x1, y0, zz], [x1, y1, zz], [x0, y1, zz]]))
            q.set_image(solid(pp.QUAD_L, 0.18))
        quads = np.where(np.isin(mus_vol, [1, 2]), mus_vol, 0)
        ci, opi = paint((quads, MUS_T, 0.25), (im_raw, {1: pp.GHOST}, 0.9))
        leg_i = legend([(pp.IM_LEGS, "intermuscular · legs"), (pp.IM_TRUNK, "intermuscular · trunk"),
                        (pp.QUAD_L, "quadriceps")])
        self.voxel_transition(vox, rgb_=ci, op=opi, run_time=T(1.8),
                              hook=lambda f: [setattr(q, "opacity", f) for q in planes] and None)
        ci2, opi2 = paint((quads, MUS_T, 0.25), (im_pre, IM_T, 0.95))
        self.voxel_transition(vox, rgb_=ci2, op=opi2, delay=world[:, 2], lag=0.5, run_time=T(2.0),
                              extra=[FadeIn(leg_i)])
        sw = (im_pre != fat_im) & (fat_im > 0)
        ci3, opi3 = paint((quads, MUS_T, 0.25), (fat_im, IM_T, 0.95))
        szi = np.where(sw, 1.8, 1.0).astype("f4")
        self.voxel_transition(vox, rgb_=ci3, op=opi3, size=szi, run_time=T(1.4))
        self.hold(T(1.0))
        self.voxel_transition(vox, size=np.ones(N, "f4"), run_time=T(0.5),
                              hook=lambda f: [setattr(q, "opacity", 1 - f) for q in planes] and None)

        # ------------------------------------------------------------ organs and muscles
        capM = self.caption("Organs and muscles", [
            ("liver (kidney hulls cleared), spleen, pancreas · quadriceps, paravertebral, deltoids, trapezius", INK, 18),
            ("the deltoid split left / right by its two pieces; the skeleton subtracted from every muscle",
             INK_SOFT, 16),
        ])
        self.swap(capI, capM)
        mus_b = mus_raw.copy()
        mus_b[delt & (mus_b == 6)] = 5  # bilateral first
        cm, opm = paint((org_vol, ORG_T), (mus_b, MUS_T))
        leg_m = legend([(pp.LIVER, "liver"), (pp.SPLEEN, "spleen"), (pp.PANCREAS, "pancreas"),
                        (pp.QUAD_L, "quadriceps L"), (pp.QUAD_R, "quadriceps R"), (pp.PVM_L, "paravertebral L"),
                        (pp.PVM_R, "paravertebral R"), (pp.DELT_L, "deltoid L"), (pp.DELT_R, "deltoid R"),
                        (pp.TRAP, "trapezius")])
        self.play(FadeOut(leg_i), run_time=T(0.3))
        self.voxel_transition(vox, rgb_=cm, op=opm, run_time=T(2.0), extra=[FadeIn(leg_m)])
        cm2, opm2 = paint((org_vol, ORG_T), (mus_vol, MUS_T))
        self.voxel_transition(vox, rgb_=cm2, op=opm2, run_time=T(1.2))
        self.animate_gl(lambda f: setattr(skel, "opacity", 0.55 * f), run_time=T(1.4))
        self.hold(T(1.6))
        self.animate_gl(lambda f: setattr(skel, "opacity", 0.55 * (1 - f)), run_time=T(0.8))

        # ------------------------------------------------------------ bone marrow (CT grid)
        mo_raw, mo_iso, mo_fin, mo_vol = A("mo_raw"), A("mo_iso"), A("mo_final"), A("mo_vol")
        mc = centre_of(mo_raw > 0, mw)
        self.move_camera(run_time=T(2.2), target=mc, distance=dist_for(150), phi=78, shift=(0.25, 0.06),
                         extra=[FadeOut(leg_m)])
        capMO = self.caption("L3 and L4 · the bone marrow", [
            (f"each vertebra eroded by {meta['bn_erode']} voxels, then {meta['mo_xy_erode']} more in-plane: "
             "the SUV mask", INK, 18),
            (f"blue: the vertebra's voxels below {meta['mo_vol_max_hu']:.0f} HU, the volume twin", pp.VOLUME, 18),
        ])
        self.swap(capM, capMO)
        self.voxel_transition(vox, op=np.zeros(N, "f4"), run_time=T(1.0),
                              hook=lambda f: setattr(skel, "opacity", 0.3 * f))
        mlut = pp.lut(MO_T)
        movox.set(rgb=mlut[mo_raw], op=np.zeros(len(mw), "f4"))
        self.voxel_transition(movox, op=np.ones(len(mw), "f4"), run_time=T(1.2))
        self.voxel_transition(movox, op=np.where(mo_iso > 0, 1.0, 0.12).astype("f4"), run_time=T(1.4))
        self.voxel_transition(movox, op=np.where(mo_fin > 0, 1.0, np.where(mo_raw > 0, 0.06, 0.0)).astype("f4"),
                              size=np.where(mo_fin > 0, 1.0, 1.0).astype("f4"), run_time=T(1.4))
        self.hold(T(0.8))
        cmo = mlut[mo_raw].copy()
        cmo[(mo_vol > 0) & (mo_fin == 0)] = rgb(pp.VOLUME)
        opmo = np.where(mo_fin > 0, 1.0, np.where(mo_vol > 0, 0.35, 0.04)).astype("f4")
        self.voxel_transition(movox, rgb_=cmo, op=opmo, run_time=T(1.6))
        self.hold(T(1.2))

        # ------------------------------------------------------------ SUV masks: the erosion
        self.move_camera(run_time=T(2.6), target=trunk_c, distance=dist_for(520), phi=82, shift=(0.3, 0.06))
        capE = self.caption("SUV masks: eroded against motion", [
            ("CT and PET are minutes apart: breathing and bowel move; each SUV mask keeps clear of its border",
             INK, 16),
            ("red: the SUV mask (liver, spleen, muscles 4 voxels in) · blue: the uneroded volume twin",
             INK_SOFT, 16),
        ])
        self.swap(capMO, capE)
        self.voxel_transition(movox, op=np.zeros(len(mw), "f4"), run_time=T(0.6),
                              hook=lambda f: setattr(skel, "opacity", 0.3 * (1 - f)))
        ring_o = (org_vol > 0) & (org == 0)
        ring_m = (mus_vol > 0) & (mus_pre == 0)
        ce = np.tile(rgb(pp.MEASURE), (N, 1)).astype("f4")
        ce[ring_o | ring_m] = rgb(pp.VOLUME)
        ope = np.where((org > 0) | (mus_pre > 0), 1.0, np.where(ring_o | ring_m, 0.3, 0.0)).astype("f4")
        self.voxel_transition(vox, rgb_=ce, op=ope, run_time=T(2.0))
        self.hold(T(1.6))

        # ------------------------------------------------------------ the last PET bed: quadriceps trim
        edge = meta["pet_edge"]
        quad_pre = np.isin(mus_pre, [1, 2])
        quad_vol = np.isin(mus_vol, [1, 2])
        cut = quad_pre & ~np.isin(mus, [1, 2])
        qc = centre_of(quad_vol)
        qz = meta["quads_z_world"]
        qc[2] = qz[0] + 0.4 * (qz[1] - qz[0])
        self.move_camera(run_time=T(2.4), target=qc, distance=dist_for(qz[1] - qz[0] + 260), phi=80,
                         shift=(-0.12, 0.2))
        rel0 = float(np.asarray(data["edge_rel"])[0])
        capT = self.caption("The last PET bed · the quadriceps trimmed", [
            (f"the PET is acquired in {edge['n_beds']} beds of {edge['axial_fov']:.0f} mm, {edge['step']:.0f} mm apart: "
             "two beds overlap everywhere but at the ends", INK, 16),
            (f"at mid-thigh only the lowest bed counts: its sensitivity falls to 0, the noise rises (×{rel0:.1f} "
             "at the last slice)", INK_SOFT, 16),
        ])
        self.swap(capE, capT)
        cq = np.tile(rgb(pp.VOLUME), (N, 1)).astype("f4")
        cq[quad_pre] = rgb(pp.MEASURE)
        opq = np.where(quad_pre, 1.0, np.where(quad_vol, 0.12, 0.0)).astype("f4")
        z_c, z_p = float(edge["leg_bed_centre_mm"]), float(edge["z_plane_mm"])
        qx0, qx1 = world[quad_vol, 0].min() - 40, world[quad_vol, 0].max() + 40
        qy0, qy1 = world[quad_vol, 1].min() - 40, world[quad_vol, 1].max() + 40

        def plane(zz: float, color: str, alpha: float) -> Quad:
            return Quad(np.array([[qx0, qy0, zz], [qx1, qy0, zz], [qx1, qy1, zz], [qx0, qy1, zz]]),
                        solid(color, alpha), opacity=0.0)

        centre_q, cut_q = plane(z_c, pp.GHOST, 0.35), plane(z_p, CUT, 0.3)
        self.gl.add(centre_q, cut_q)
        plot = self._edge_plot(data, edge)
        self.voxel_transition(vox, rgb_=cq, op=opq, run_time=T(1.8),
                              hook=lambda f: setattr(pet, "opacity", 0.45 * f))
        self.play(FadeIn(plot), run_time=T(0.8))
        self.animate_gl(lambda f: setattr(centre_q, "opacity", f), run_time=T(0.8))
        self.hold(T(1.6))
        capT2 = self.caption(f"{edge['cut_distance_mm']:.0f} mm below that bed's centre the noise passes ×{edge['noise_limit']:g}", [
            (f"×{edge['noise_limit']:g} the two-bed level: the {edge['pet_slices_removed']} PET slices below leave the "
             "quadriceps SUV mask", INK, 16),
            ("the plane from the PET DICOM headers (FrameReferenceTime groups) kept by stage 0 in PT_beds.json",
             INK_SOFT, 16),
        ])
        self.swap(capT, capT2)
        self.animate_gl(lambda f: setattr(cut_q, "opacity", f), run_time=T(0.8))
        cr = cq.copy()
        cr[cut] = rgb(CUT)
        self.voxel_transition(vox, rgb_=cr, size=np.where(cut, 1.5, 1.0).astype("f4"), delay=-world[:, 2],
                              lag=0.4, run_time=T(1.6))
        self.hold(T(1.2))
        cv = cq.copy()
        cv[cut] = rgb(pp.VOLUME)
        self.voxel_transition(vox, rgb_=cv, op=np.where(cut, 0.45, opq).astype("f4"), size=np.ones(N, "f4"),
                              run_time=T(1.4))

        def pct(key: str) -> str:
            a, b = edge["quad_full"]["LR"][key], edge["quad_trim"]["LR"][key]
            d = 100.0 * (b / a - 1.0)
            return "unchanged" if abs(d) < 0.05 else f"{d:+.1f} %"

        capT3 = self.caption("What the trim changes", [
            (f"quadriceps SUVmean {pct('mean')} · SUVmax {pct('max')} · SUV95p {pct('p95')} · SUV99p {pct('p99')}",
             INK, 17),
            ("red: the SUV mask · blue: MUSCLES_VOL keeps the cut slices, so the volumes do not change", pp.VOLUME, 16),
        ])
        self.swap(capT2, capT3)
        self.hold(T(2.6))
        self.animate_gl(lambda f: (setattr(centre_q, "opacity", 1 - f), setattr(cut_q, "opacity", 1 - f),
                                   setattr(pet, "opacity", 0.45 * (1 - f))), run_time=T(0.8),
                        extra=[FadeOut(plot)])
        capE = capT3

        # ------------------------------------------------------------ stage 3
        self.move_camera(run_time=T(2.2), target=body_c, distance=dist_for(z_hi - z_lo + 80), phi=86, shift=SHIFT)
        capS = self.caption("Stage 3 · SUV inside every mask", [
            ("the PET (SUV, body weight) read inside each eroded mask: mean, max, 95th and 99th percentiles", INK, 18),
            ("volumes from the uneroded twins · intermuscular and body: volume only", INK_SOFT, 18),
        ])
        self.swap(capE, capS)
        cs = pp.ramp(suv / 3.0, pp.SUV_STOPS)
        ops = np.where((org > 0) | (mus > 0), 1.0, 0.0).astype("f4")
        bar = self._colorbar(pp.SUV_STOPS, "SUV (bw)", 0, 3)
        self.voxel_transition(vox, rgb_=cs, op=ops, delay=dz, lag=0.5, run_time=T(2.6), extra=[FadeIn(bar)])
        self.voxel_transition(movox, rgb_=pp.ramp(A("mo_suv", float) / 3.0, pp.SUV_STOPS),
                              op=(mo_fin > 0).astype("f4"), run_time=T(0.6))
        self.hold(T(1.4))
        rows = [("liver", "HIGADO", "HIGADO_VOL", pp.LIVER), ("spleen", "BAZO", "BAZO_VOL", pp.SPLEEN),
                ("L3-L4 marrow", "MO", ("MO_L3_VOL", "MO_L4_VOL"), pp.L3),
                ("pancreas", "PANCREAS", "PANCREAS_VOL", pp.PANCREAS),
                ("paravertebral", "PARAVERTEBRAL_LR", ("PARAVERTEBRAL_L_VOL", "PARAVERTEBRAL_R_VOL"), pp.PVM_L),
                ("trapezius", "TRAPECIOS", "TRAPECIOS_VOL", pp.TRAP),
                ("quadriceps", "CUADRICEPS_LR", ("CUADRICEPS_L_VOL", "CUADRICEPS_R_VOL"), pp.QUAD_L),
                ("deltoids", "DELTOIDES_LR", ("DELTOIDES_L_VOL", "DELTOIDES_R_VOL"), pp.DELT_L),
                ("visceral fat", "GRASA_V", "GRASA_V_VOL", pp.FAT_V),
                ("subcutaneous fat", "GRASA_SC", "GRASA_SC_VOL", pp.FAT_SC)]
        chart = self._suv_chart(rows, s3)
        self.play(FadeOut(bar), FadeIn(chart, shift=RIGHT * 0.2), run_time=T(0.8))
        self.hold(T(4.5))

        end = self.caption("What stage 3 writes", [
            ("per mask: SUVmax · SUVmean · SUV95p · SUV99p, and volume · slices", INK, 20),
            (f"intermuscular fat: legs {s3['GRASA_IM_PIERNAS_VOL']:.0f} cc · trunk {s3['GRASA_IM_TORAX_VOL']:.0f} cc"
             f" · body {s3['CORP_VOL'] / 1000:.1f} L", INK_SOFT, 18),
        ])
        self.swap(capS, end)
        self.orbit(ORBIT * 1.6)
        self.hold(T(3.0))
        self.play(FadeOut(end), FadeOut(chart), run_time=T(0.8))
        self.hold(T(0.4))

    def _edge_plot(self, data, edge: dict) -> VGroup:
        """Noise of the lowest PET slices against their height: the triangular-sensitivity model
        (line) and the measured noise (dots), relative to the two-bed overlap level."""
        W, Hh = 2.6, 3.1
        z = np.asarray(data["edge_z"], float)
        rel, model = np.asarray(data["edge_rel"], float), np.asarray(data["edge_model"], float)
        z0 = float(edge["z_first"])
        top = 150.0
        n_lo, n_hi = 0.8, 3.2

        def P(n: float, zz: float) -> np.ndarray:
            return np.array([W * (n - n_lo) / (n_hi - n_lo), Hh * (zz - z0) / top, 0.0])

        keep = (z - z0) <= top
        axes = VGroup(Line(P(n_lo, z0), P(n_hi, z0), color=INK_SOFT, stroke_width=1.5),
                      Line(P(n_lo, z0), P(n_lo, z0 + top), color=INK_SOFT, stroke_width=1.5))
        ticks = VGroup()
        for n in (1, 2, 3):
            ticks.add(Text(f"×{n}", font=FONT, color=INK_SOFT, font_size=12).next_to(P(n, z0), DOWN, buff=0.08))
        for d in (0, 50, 100, 150):
            ticks.add(Text(f"{d}", font=FONT, color=INK_SOFT, font_size=12).next_to(P(n_lo, z0 + d), LEFT, buff=0.08))
        band = Rectangle(width=W, height=Hh * (edge["z_plane_mm"] - z0) / top, fill_color=CUT,
                         fill_opacity=0.14, stroke_width=0).move_to(P(n_lo, z0), aligned_edge=DOWN + LEFT)
        curve = VMobject(color=INK, stroke_width=2.2)
        pts = [P(min(n, n_hi), zz) for n, zz in zip(model[keep], z[keep])]
        curve.set_points_as_corners(pts)
        dots = VGroup(*[Dot(P(n, zz), radius=0.025, color=INK_SOFT)
                        for n, zz in zip(rel[keep], z[keep]) if np.isfinite(n) and n_lo <= n <= n_hi])
        lim = DashedLine(P(edge["noise_limit"], z0), P(edge["noise_limit"], z0 + top), color=pp.MEASURE,
                         stroke_width=1.5, dash_length=0.06)
        cen = DashedLine(P(n_lo, edge["leg_bed_centre_mm"]), P(n_hi, edge["leg_bed_centre_mm"]), color=INK_SOFT,
                         stroke_width=1.5, dash_length=0.06)
        cutl = Line(P(n_lo, edge["z_plane_mm"]), P(n_hi, edge["z_plane_mm"]), color=CUT, stroke_width=2.2)
        lab_c = Text("leg-bed centre", font=FONT, color=INK_SOFT, font_size=12).next_to(cen.get_end(), UP, buff=0.05)
        lab_c.align_to(cen, RIGHT)
        lab_cut = Text(f"cut · {edge['cut_distance_mm']:.0f} mm below", font=FONT, color=CUT,
                       font_size=12).next_to(cutl.get_end(), UP, buff=0.05).align_to(cutl, RIGHT)
        lab_lim = Text(f"×{edge['noise_limit']:g}", font=FONT, color=pp.MEASURE, font_size=12).next_to(
            lim.get_end(), UP, buff=0.05)
        key = VGroup(VGroup(Line(ORIGIN_L, ORIGIN_L + RIGHT * 0.3, color=INK, stroke_width=2.2),
                            Text("model", font=FONT, color=INK, font_size=12)).arrange(RIGHT, buff=0.08),
                     VGroup(Dot(radius=0.03, color=INK_SOFT),
                            Text("measured", font=FONT, color=INK, font_size=12)).arrange(RIGHT, buff=0.08)
                     ).arrange(RIGHT, buff=0.25)
        title = Text("PET noise at the scan end", font=FONT, weight="BOLD", color=INK, font_size=14)
        xl = Text("noise / two-bed level", font=FONT, color=INK_SOFT, font_size=12)
        yl = Text("mm above the last slice", font=FONT, color=INK_SOFT, font_size=12).rotate(np.pi / 2)
        body = VGroup(band, axes, ticks, curve, dots, lim, cen, cutl, lab_c, lab_cut, lab_lim)
        xl.next_to(ticks, DOWN, buff=0.08).set_x(P(0.5 * (n_lo + n_hi), z0)[0])
        yl.next_to(ticks, LEFT, buff=0.1).set_y(P(n_lo, z0 + top / 2)[1])
        title.next_to(body, UP, buff=0.32).align_to(axes, LEFT)
        key.next_to(title, DOWN, buff=0.08).align_to(axes, LEFT)
        g = VGroup(body, xl, yl, title, key).to_corner(UR, buff=0.45)
        g.add_background_rectangle(color="#FFFFFF", opacity=0.9, buff=0.14)
        self.fixed(g, keep=True, drop=(title, xl, yl, lab_c, lab_cut, lab_lim, key))
        return g

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

    def _suv_chart(self, rows, s3: dict) -> VGroup:
        """Horizontal bars of SUVmean per mask, with SUVmax and the volume beside."""
        vmax, width = 2.5, 2.4
        head = VGroup(Text("SUVmean", font=FONT, weight="BOLD", color=INK, font_size=14),
                      Text("SUVmax", font=FONT, weight="BOLD", color=INK, font_size=14),
                      Text("volume (cc)", font=FONT, weight="BOLD", color=INK, font_size=14))
        lines = VGroup()
        for name, key, volk, col in rows:
            mean = float(s3[f"{key}_SUVmean"])
            mx = float(s3[f"{key}_SUVMAX"])
            vol = sum(float(s3[k]) for k in (volk if isinstance(volk, tuple) else (volk,)))
            nm = Text(name, font=FONT, color=INK, font_size=13)
            b = Rectangle(width=max(width * mean / vmax, 0.02), height=0.17, fill_color=col, fill_opacity=1,
                          stroke_width=0)
            v = Text(f"{mean:.2f}", font=FONT, color=INK_SOFT, font_size=12)
            t = Text(f"{mx:.1f}", font=FONT, color=INK_SOFT, font_size=12)
            vv = Text(f"{vol:,.0f}", font=FONT, color=INK_SOFT, font_size=12)
            lines.add(VGroup(nm, b, v, t, vv))
        x_name, x_bar, x_mx, x_vol = 0.0, 1.75, 1.75 + width + 0.95, 1.75 + width + 2.1
        for i, g in enumerate(lines):
            y = -0.29 * i
            nm, b, v, t, vv = g
            nm.move_to([x_name, y, 0], aligned_edge=LEFT)
            b.move_to([x_bar, y, 0], aligned_edge=LEFT)
            v.next_to(b, RIGHT, buff=0.08)
            t.move_to([x_mx, y, 0])
            vv.move_to([x_vol, y, 0])
        h0, h1, h2 = head
        h0.move_to([x_bar, 0.34, 0], aligned_edge=LEFT)
        h1.move_to([x_mx, 0.34, 0])
        h2.move_to([x_vol, 0.34, 0])
        g = VGroup(head, lines).to_corner(UL, buff=0.5)
        g.add_background_rectangle(color="#FFFFFF", opacity=0.9, buff=0.16)
        self.fixed(g, keep=True)
        return g


if __name__ == "__main__":
    run_scene(__file__, "PesaCtPet", "pesa_ctpet")
