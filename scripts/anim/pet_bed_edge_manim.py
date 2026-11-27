"""Manim animation: the noisy inferior edge of a whole-body PET and the proposed quadriceps trim.

Explains, on one PESA-Health CT-PET subject, why the last slices of the PET (cut through the thighs)
are noisy and what the proposed per-subject trim of the quadriceps SUV mask would do:

1. the whole-body PET ends half-way down the thighs;
2. step-and-shoot beds (164 mm field, 100 mm step, 210 s each, head to legs);
3. ``FrameReferenceTime`` groups the slices by bed; bed centres from ``(7053,10D5)``;
4. one sensitivity triangle per bed, their sum (0.78 in the overlaps, 0.11 in the last slice);
5. the thighs: axial slices from the inferior end, measured noise vs ``sqrt(0.78 / sens)``;
6. the quadriceps masks (red = measure, eroded; blue = volumetry);
7. the rule (noise <= x1.25 -> 41 mm below the leg-bed centre), the trim and its effect;
8. why the number of removed slices changes between subjects;
9. what we know, what we infer or assume, what to confirm with the technologists.

Everything comes from ``pet_bed_edge_stages.npz`` (``pet_bed_edge_precompute.py``); this file only
draws (numpy + manim). No subject identifier is shown.

Render (manim environment; ``--no-text`` keeps every text element but transparent)::

    PB_FAST=1 python scripts/anim/pet_bed_edge_manim.py -ql --fps 15
    python scripts/anim/pet_bed_edge_manim.py -qh --fps 30              # -> pet_bed_edge.mp4
    python scripts/anim/pet_bed_edge_manim.py -qh --fps 30 --no-text    # -> pet_bed_edge_notext.mp4
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys
from pathlib import Path

import numpy as np
from manim import (
    BOLD,
    DOWN,
    LEFT,
    RIGHT,
    UL,
    UR,
    UP,
    AnimationGroup,
    Create,
    DashedLine,
    Dot,
    DoubleArrow,
    FadeIn,
    Group,
    FadeOut,
    ImageMobject,
    LaggedStart,
    Line,
    Polygon,
    Rectangle,
    ReplacementTransform,
    Scene,
    Text,
    Transform,
    VGroup,
    VMobject,
    config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anim_voxels import FONT, INK, INK_FAINT, INK_SOFT, no_text, run_scene  # noqa: E402

DATA_PATH = Path(os.environ.get("PB_STAGES", str(Path(__file__).with_name("pet_bed_edge_stages.npz"))))
FAST = os.environ.get("PB_FAST", "0") == "1"

config.background_color = "#FFFFFF"

BED_A, BED_B = "#2F7FD6", "#158C81"   # alternating beds / groups
SENS = "#1F3F66"
MEASURED = "#2F7FD6"
MODEL = "#EFA43A"
MEAS = "#E8443A"      # measure mask (red, as in the QC report)
VOLM = "#2F7FD6"      # volumetry mask (blue)
CUT = "#D6336C"
CENTRE = "#158C81"
GOOD = "#2E9E4F"

# ---------------------------------------------------------------- data
D = np.load(DATA_PATH)
Z = D["z"]
Z0 = float(Z[0])
NK = len(Z)
SENS_K = D["sens"]
CENTRES = D["centres"]
GROUPS = D["groups"]
N_CUT = int(D["n_cut"])
Z_CUT = float(D["z_cut"])
CUT_D = float(D["cut_distance"])
REL, PRED = D["rel"], D["pred"]


def es(x: float, nd: int = 1) -> str:
    """Spanish decimal comma."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    s = es(abs(x), nd)
    return "0 %" if abs(x) < 0.5 * 10**-nd else ("+" if x > 0 else "−") + s + " %"


def gray_r(a: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    v = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    g = (255 * (1 - v)).astype(np.uint8)
    return np.dstack([g, g, g])


def gray(a: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    v = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    g = (255 * v).astype(np.uint8)
    return np.dstack([g, g, g])


def overlay(mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    c = color.lstrip("#")
    rgb = [int(c[i : i + 2], 16) for i in (0, 2, 4)]
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[mask] = [*rgb, int(255 * alpha)]
    return out


def image(arr: np.ndarray, height: float, centre) -> ImageMobject:
    """Image with k (rows of *arr*) going up: row 0 of the array at the bottom."""
    im = ImageMobject(np.ascontiguousarray(arr[::-1]))
    im.set_resampling_algorithm(0)  # nearest: keep the PET grain visible
    im.height = height
    im.move_to(np.array([centre[0], centre[1], 0.0]))
    return im


def T(s: str, size: float = 18, color: str = INK, bold: bool = False) -> Text:
    """Text; transparent with ``--no-text`` (timing unchanged)."""
    t = Text(s, font=FONT, font_size=size, color=color, weight=BOLD if bold else "NORMAL")
    if no_text():
        t.set_opacity(0.0)
    return t


def caption(head: str, lines: list[str], corner, width: float, size: float = 17) -> VGroup:
    h = T(head, 23, INK, bold=True)
    body = VGroup(*[T(s, size, INK_SOFT if s.startswith("·") else INK) for s in lines]).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
    block = VGroup(h, body).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
    if block.width > width:
        block.scale_to_fit_width(width)
    block.move_to(corner, aligned_edge=UL)
    return block


# ---------------------------------------------------------------- whole-body geometry
WB_H = 7.2
WB_PX = WB_H / NK
WB_X = -5.0
WB_YB = -3.72


def wb_y(z: float) -> float:
    return WB_YB + ((z - Z0) / 2.0 + 0.5) * WB_PX


MIP_W = D["pet_mip"].shape[1] * WB_PX
MIP_L, MIP_R = WB_X - MIP_W / 2, WB_X + MIP_W / 2
SX0, SX1 = -1.45, 1.25            # sensitivity axis: 0 .. 1
CAP_WB = np.array([1.85, 3.72, 0])


def sx(s: float) -> float:
    return SX0 + (SX1 - SX0) * s


# ---------------------------------------------------------------- legs geometry
KL = 96                     # slices shown in the leg views (192 mm)
_cols = np.where(D["vol_mip"].any(0))[0]
C0, C1 = max(int(_cols.min()) - 14, 0), min(int(_cols.max()) + 14, D["pet_cor"].shape[1])
LG_H = 3.9
LG_PX = LG_H / KL
LG_W = (C1 - C0) * LG_PX
LG_XL = -6.55
LG_X = LG_XL + LG_W / 2
LG_YB = -1.05
NX0, NX1 = 1.85, 6.75       # noise plot: rel 0.75 .. 2.75
R0, R1 = 0.75, 2.75


def lg_y(k: float) -> float:
    """y of the lower edge of slice *k* (k may be fractional)."""
    return LG_YB + k * LG_PX


def lg_x(col: float) -> float:
    return LG_XL + (col - C0 + 0.5) * LG_PX


def nx(r: float) -> float:
    return NX0 + (NX1 - NX0) * (np.clip(r, R0, R1) - R0) / (R1 - R0)


class PetBedEdge(Scene):
    def wait_(self, t: float) -> None:
        self.wait(t * (0.35 if FAST else 1.0))

    def construct(self) -> None:
        self.title_card()
        mip, quad_wb = self.whole_body()
        beds, centres = self.beds(mip)
        self.sensitivity(mip, quad_wb, beds, centres)
        legs = self.legs()
        self.masks_and_rule(legs)
        self.cohort()
        self.summary()

    # ------------------------------------------------------------ 0. title
    def title_card(self) -> None:
        t1 = T("SUV del cuádriceps en el borde inferior del PET", 34, INK, bold=True)
        t2 = T("por qué los últimos cortes son ruidosos y cuánto recortar en cada sujeto", 22, INK_SOFT)
        t3 = T("PET-TC de cuerpo entero · PESA-Health · protocolo PESA PET V5", 18, INK_SOFT)
        g = VGroup(t1, t2, t3).arrange(DOWN, buff=0.3)
        self.play(FadeIn(g, shift=UP * 0.15), run_time=1.0)
        self.wait_(2.5)
        self.play(FadeOut(g), run_time=0.6)

    # ------------------------------------------------------------ 1. the volume
    def whole_body(self):
        mip = D["pet_mip"]
        vmax = float(np.percentile(mip, 99.0))
        mob = image(gray_r(mip, 0, vmax), WB_H, (WB_X, WB_YB + WB_H / 2))
        kq = D["vol_mip"].shape[0]
        qa = np.zeros(mip.shape, bool)
        qa[:kq] = D["vol_mip"]
        quad = image(overlay(qa, VOLM, 0.45), WB_H, (WB_X, WB_YB + WB_H / 2))
        end = Line([MIP_L - 0.05, WB_YB, 0], [MIP_R + 0.05, WB_YB, 0], color=CUT, stroke_width=5)
        cap = caption(
            "El PET termina a mitad de muslo",
            ["El último corte atraviesa el cuádriceps (azul).",
             "En esos cortes la imagen es más ruidosa,",
             "y el SUV del cuádriceps se mide con todos",
             "los vóxeles de su máscara.",
             "· vista: MIP coronal del PET, piernas abajo"],
            CAP_WB, 5.0)
        self.play(FadeIn(mob), run_time=1.0)
        self.play(FadeIn(cap), run_time=0.6)
        self.play(FadeIn(quad), Create(end), run_time=1.0)
        self.wait_(4.0)
        self.play(FadeOut(cap), FadeOut(end), run_time=0.5)
        self.cap = None
        return mob, quad

    # ------------------------------------------------------------ 2-3. beds and groups
    def beds(self, mip):
        nb = len(CENTRES)
        rects, dots = [], []
        for i, c in enumerate(CENTRES):
            x0, x1 = (-2.95, -2.5) if i % 2 == 0 else (-2.4, -1.95)
            col = BED_A if i % 2 == 0 else BED_B
            r = Rectangle(width=x1 - x0, height=164.0 / 2 * WB_PX, stroke_color=col, stroke_width=2,
                          fill_color=col, fill_opacity=0.18)
            r.move_to([(x0 + x1) / 2, wb_y(c), 0])
            rects.append(r)
            dots.append(Dot([(x0 + x1) / 2, wb_y(c), 0], radius=0.04, color=col))
        cap = caption(
            f"{nb} camas, de la cabeza a las piernas",
            ["Cada cama ve 164 mm del cuerpo (campo axial)",
             "durante 210 s; la mesa avanza 100 mm",
             "y se repite. Dos camas vecinas se solapan 64 mm.",
             "La cama de las piernas es la última.",
             "· FieldOfViewDimensions (0018,1149) = [764, 164]",
             "· ActualFrameDuration (0018,1242) = 210 s"],
            CAP_WB, 5.0)
        self.play(FadeIn(cap), run_time=0.6)
        clock = T("cama 1", 18, INK).move_to([-2.45, 3.82, 0])
        self.play(FadeIn(clock), run_time=0.2)
        order = list(range(nb))[::-1]       # acquisition: head (top) first
        for n, i in enumerate(order):
            new = T(f"cama {n + 1} · {es((n + 1) * 3.5, 1)} min", 18, INK).move_to([-2.45, 3.82, 0])
            self.play(FadeIn(rects[i]), Transform(clock, new), run_time=0.45 if not FAST else 0.2)
        # step / overlap / field on the two lowest beds
        c0, c1 = CENTRES[0], CENTRES[1]
        xa = -3.05
        fov = DoubleArrow([xa, wb_y(c0 - 82), 0], [xa, wb_y(c0 + 82), 0], buff=0, stroke_width=2,
                          tip_length=0.1, color=INK)
        ov = DoubleArrow([-1.8, wb_y(c1 - 82), 0], [-1.8, wb_y(c0 + 82), 0], buff=0, stroke_width=2,
                         tip_length=0.1, color=CUT)
        l_fov = T("164 mm", 15, INK).next_to(fov, LEFT, buff=0.06).shift(RIGHT * 0.0)
        l_ov = T("solape 64 mm", 15, CUT).next_to(ov, RIGHT, buff=0.08)
        st = DoubleArrow([-1.8, wb_y(c0), 0], [-1.8, wb_y(c1), 0], buff=0, stroke_width=2, tip_length=0.1,
                         color=CENTRE).shift(RIGHT * 0.0)
        l_st = T("paso 100 mm", 15, CENTRE).next_to(st, RIGHT, buff=0.08)
        self.play(FadeIn(VGroup(*dots[:2])), Create(fov), FadeIn(l_fov), run_time=0.7)
        self.play(Create(st), FadeIn(l_st), run_time=0.6)
        self.wait_(1.0)
        self.play(FadeOut(VGroup(st, l_st)), Create(ov), FadeIn(l_ov), run_time=0.6)
        self.wait_(2.5)
        self.play(FadeOut(VGroup(fov, l_fov, ov, l_ov, clock)), FadeOut(cap), run_time=0.5)

        # FrameReferenceTime groups on the MIP
        bands, labels = [], []
        for g, (a, b) in enumerate(GROUPS):
            col = BED_A if g % 2 == 0 else BED_B
            y0 = WB_YB + a * WB_PX
            y1 = WB_YB + (b + 1) * WB_PX
            band = Rectangle(width=MIP_W, height=y1 - y0, stroke_width=0, fill_color=col, fill_opacity=0.16)
            band.move_to([WB_X, (y0 + y1) / 2, 0])
            bands.append(band)
        n_leg = int(GROUPS[0][1] - GROUPS[0][0] + 1)
        n_mid = int(GROUPS[1][1] - GROUPS[1][0] + 1)
        lab_leg = T(f"{n_leg} cortes", 15, INK).move_to([MIP_R - 0.55, WB_YB + n_leg * WB_PX / 2, 0])
        lab_mid = T(f"{n_mid} cortes = 100 mm", 15, INK).move_to(
            [MIP_R - 1.0, WB_YB + (GROUPS[1][0] + n_mid / 2) * WB_PX, 0])
        for lab in (lab_leg, lab_mid):
            lab.add_background_rectangle(color="#FFFFFF", opacity=0.8, buff=0.04)
        cap = caption(
            "Cada corte sabe de qué cama viene",
            ["FrameReferenceTime (0054,1300) es igual en todos",
             "los cortes de una cama: agrupa el volumen en camas.",
             "Cada grupo tiene 100 mm (el paso), no 164:",
             "en el solape el corte se asigna a una sola cama.",
             "El centro de cada cama queda 21 mm por encima",
             "del inicio de su grupo (tramo constante del",
             "tag privado (7053,10D5)). El de la cama de",
             "piernas se extrapola: 100 mm bajo la siguiente.",
             "· (7053,100E) y las horas de cama cambian",
             "· en los mismos cortes"],
            CAP_WB, 5.0)
        self.play(LaggedStart(*[FadeIn(b) for b in bands], lag_ratio=0.08), FadeIn(cap), run_time=1.4)
        self.play(FadeIn(lab_leg), FadeIn(lab_mid), FadeIn(VGroup(*dots[2:])), run_time=0.6)
        self.wait_(5.5)
        self.play(FadeOut(VGroup(*bands, lab_leg, lab_mid)), FadeOut(cap), run_time=0.6)
        return rects, dots

    # ------------------------------------------------------------ 4. sensitivity
    def sensitivity(self, mip, quad, rects, dots):
        y_bot, y_top = WB_YB, WB_YB + WB_H
        axis = Line([SX0, y_bot, 0], [SX1 + 0.1, y_bot, 0], color=INK, stroke_width=1.5)
        ticks = VGroup()
        for s, txt in ((0, "0"), (0.5, "0,5"), (0.78, "0,78"), (1.0, "1")):
            ticks.add(Line([sx(s), y_bot, 0], [sx(s), y_bot - 0.07, 0], color=INK, stroke_width=1.5))
            ticks.add(T(txt, 13, INK).move_to([sx(s), y_bot - 0.2, 0]))
        ax_lab = T("sensibilidad", 15, INK).move_to([sx(0.5), y_bot + 0.0, 0]).shift(DOWN * 0.42)
        tris = []
        for i, c in enumerate(CENTRES):
            col = BED_A if i % 2 == 0 else BED_B
            tri = Polygon([sx(0), wb_y(c - 82), 0], [sx(1), wb_y(c), 0], [sx(0), wb_y(c + 82), 0],
                          stroke_color=col, stroke_width=1.6, fill_color=col, fill_opacity=0.08)
            tris.append(tri)
        cap = caption(
            "Sensibilidad: un triángulo por cama",
            ["Un punto en el centro de la cama lo ven todos",
             "los detectores; en el borde, casi ninguno:",
             "sens(d) = 1 − d / 82 mm (modelo 3D clásico).",
             "Sumadas, las camas dan ≈ 1 en sus centros",
             "y 0,78 en los solapes (el peor nivel interior).",
             "Abajo no hay cama vecina que compense."],
            CAP_WB, 5.0)
        self.play(FadeIn(cap), Create(axis), FadeIn(ticks), FadeIn(ax_lab), FadeOut(VGroup(*dots)), run_time=0.8)
        self.play(*[ReplacementTransform(r, t) for r, t in zip(rects, tris)], run_time=1.6)
        k = np.arange(NK)
        ys = WB_YB + (k + 0.5) * WB_PX
        curve = VMobject(stroke_color=SENS, stroke_width=4)
        curve.set_points_as_corners([[sx(s), y, 0] for s, y in zip(SENS_K, ys)])
        ref = DashedLine([sx(0.78), y_bot, 0], [sx(0.78), y_top, 0], color=INK_SOFT, stroke_width=1.5, dash_length=0.06)
        self.play(Create(curve), run_time=2.2)
        self.play(Create(ref), run_time=0.5)
        self.wait_(1.5)
        s0 = float(SENS_K[0])
        d0 = float(CENTRES[0] - Z0)
        pt = Dot([sx(s0), ys[0], 0], radius=0.07, color=CUT)
        lab = T(f"último corte: a {d0:.0f} mm del centro,", 15, CUT)
        lab2 = T(f"sens = 1 − {d0:.0f}/82 = {es(s0, 2)}", 15, CUT)
        lab3 = T(f"ruido esperado √(0,78/{es(s0, 2)}) ≈ ×{es(np.sqrt(0.78 / s0), 1)}", 15, CUT)
        lg = VGroup(lab, lab2, lab3).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([sx(1.0) + 1.95, ys[0] + 0.55, 0])
        lg.add_background_rectangle(color="#FFFFFF", opacity=0.9, buff=0.05)
        self.play(FadeIn(pt, scale=1.6), FadeIn(lg), run_time=0.8)
        self.wait_(4.5)
        self.wb_group = VGroup(axis, ticks, ax_lab, *tris, curve, ref, pt, lg)
        self.play(FadeOut(cap), run_time=0.4)
        # zoom box on the thighs
        box = Rectangle(width=MIP_W, height=KL * WB_PX, color=CUT, stroke_width=3).move_to(
            [WB_X, WB_YB + KL * WB_PX / 2, 0])
        self.play(Create(box), run_time=0.6)
        self.zoom_box = box
        self.mip, self.quad_wb = mip, quad

    # ------------------------------------------------------------ 5. thighs: noise
    def legs(self):
        pet = D["pet_cor"][:KL, C0:C1]
        thigh = pet[: KL][D["vol_cor"][:KL, C0:C1] > 0]
        vmax = float(np.percentile(thigh, 99.5))
        cor = image(gray_r(pet, 0, vmax), LG_H, (LG_X, LG_YB + LG_H / 2))
        frame = Rectangle(width=LG_W, height=LG_H, color=CUT, stroke_width=3).move_to(cor.get_center())
        self.play(FadeOut(self.wb_group), FadeOut(self.quad_wb), run_time=0.5)
        self.play(ReplacementTransform(self.zoom_box, frame), FadeOut(self.mip), FadeIn(cor), run_time=1.2)
        self.play(FadeOut(frame), run_time=0.3)

        head = T("Menos sensibilidad → menos cuentas → más ruido", 24, INK, bold=True).move_to([-6.95, 3.5, 0], aligned_edge=LEFT)
        self.play(FadeIn(head), run_time=0.5)

        # axial thumbnails from the inferior end
        ap = D["axial_pet"]
        ref = float(D["axial_ref"])
        thumbs, labs, ticks = [], [], []
        for n, k in enumerate(D["axial_k"]):
            im = ImageMobject(gray_r(ap[n], 0, ref))
            im.set_resampling_algorithm(0)
            im.width = 2.05
            im.move_to([-5.75 + n * 2.3, -2.05, 0])
            thumbs.append(im)
            s = float(SENS_K[k])
            lab = VGroup(T(f"corte {k} · sens {es(s, 2)}", 13, INK),
                         T(f"ruido medido ×{es(REL[k], 1)}", 13, MEASURED)).arrange(DOWN, buff=0.05)
            lab.next_to(im, DOWN, buff=0.06)
            labs.append(lab)
            y = lg_y(k + 0.5)
            ticks.append(VGroup(Line([LG_XL - 0.12, y, 0], [LG_XL, y, 0], color=INK, stroke_width=2),
                                T(str(k), 11, INK).move_to([LG_XL - 0.25, y, 0])))
        self.play(LaggedStart(*[FadeIn(t) for t in thumbs], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(l) for l in labs], lag_ratio=0.15),
                  LaggedStart(*[FadeIn(t) for t in ticks], lag_ratio=0.15), run_time=2.0)
        sub = T("Mismos cortes, misma ventana: el grano gordo y los puntos negros sueltos de los primeros cortes son ruido, no anatomía.",
                15, INK_SOFT).move_to([-6.95, -3.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(sub), run_time=0.5)
        self.wait_(4.0)

        # noise plot aligned with the coronal rows
        yb, yt = lg_y(0), lg_y(KL)
        xaxis = Line([NX0, yb, 0], [NX1, yb, 0], color=INK, stroke_width=1.5)
        yaxis = Line([NX0, yb, 0], [NX0, yt, 0], color=INK, stroke_width=1.5)
        tk = VGroup()
        for r in (1.0, 1.5, 2.0, 2.5):
            tk.add(Line([nx(r), yb, 0], [nx(r), yb - 0.07, 0], color=INK, stroke_width=1.5))
            tk.add(T("×" + es(r, 1).replace(",0", ""), 13, INK).move_to([nx(r), yb - 0.2, 0]))
        for k in (0, 20, 40, 60, 80):
            tk.add(T(str(k), 12, INK_SOFT).move_to([NX0 - 0.2, lg_y(k + 0.5), 0]))
        xl = T("ruido relativo al solape", 14, INK).move_to([(NX0 + NX1) / 2, yb - 0.45, 0])
        yl = T("corte", 13, INK_SOFT).move_to([NX0 - 0.2, yt + 0.18, 0])
        n = min(len(REL), KL)
        meas = VMobject(stroke_color=MEASURED, stroke_width=3.5)
        meas.set_points_as_corners([[nx(REL[k]), lg_y(k + 0.5), 0] for k in range(n) if np.isfinite(REL[k])])
        model = VMobject(stroke_color=MODEL, stroke_width=3.5)
        model.set_points_as_corners([[nx(PRED[k]), lg_y(k + 0.5), 0] for k in range(n)])
        leg = VGroup(
            VGroup(Line(ORIGIN3, RIGHT * 0.35, color=MEASURED, stroke_width=3.5), T("medido", 13, INK)).arrange(RIGHT, buff=0.08),
            VGroup(Line(ORIGIN3, RIGHT * 0.35, color=MODEL, stroke_width=3.5), T("modelo √(0,78 / sens)", 13, INK)).arrange(RIGHT, buff=0.08),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([NX1 - 1.2, yt - 0.2, 0])
        self.play(Create(xaxis), Create(yaxis), FadeIn(tk), FadeIn(xl), FadeIn(yl), run_time=0.7)
        new_sub = T("Ruido medido = desviación del residuo (corte − mediana 5×5) / media, dentro del cuerpo; 1 = nivel del solape.",
                    15, INK_SOFT).move_to([-6.95, -3.6, 0], aligned_edge=LEFT)
        self.play(FadeOut(sub), run_time=0.3)
        sub = new_sub
        self.play(Create(meas), FadeIn(leg[0]), FadeIn(sub), run_time=2.0)
        self.play(Create(model), FadeIn(leg[1]), run_time=1.5)
        self.wait_(4.0)
        self.legs_parts = dict(cor=cor, head=head, thumbs=thumbs, labs=labs, ticks=ticks, sub=sub,
                               plot=VGroup(xaxis, yaxis, tk, xl, yl, meas, model, leg), vmax=vmax)
        return self.legs_parts

    # ------------------------------------------------------------ 6-7. masks and the rule
    def masks_and_rule(self, P):
        ct = D["ct_cor"][:KL, C0:C1]
        ctm = image(gray(ct, -160, 240), LG_H, (LG_X, LG_YB + LG_H / 2))
        vol = D["vol_cor"][:KL, C0:C1] > 0
        meas = D["meas_cor"][:KL, C0:C1] > 0
        keep = np.zeros_like(meas)
        keep[N_CUT:] = True
        o_vol = image(overlay(vol, VOLM, 0.40), LG_H, (LG_X, LG_YB + LG_H / 2))
        o_meas_keep = image(overlay(meas & keep, MEAS, 0.62), LG_H, (LG_X, LG_YB + LG_H / 2))
        o_meas_cut = image(overlay(meas & ~keep, MEAS, 0.62), LG_H, (LG_X, LG_YB + LG_H / 2))
        o_meas_gone = image(overlay(meas & ~keep, INK_FAINT, 0.55), LG_H, (LG_X, LG_YB + LG_H / 2))

        head = T("Las máscaras del cuádriceps", 24, INK, bold=True).move_to([-6.95, 3.5, 0], aligned_edge=LEFT)
        f_meas = int(min(D["first_meas_L"], D["first_meas_R"]))
        f_vol = int(min(D["first_vol_L"], D["first_vol_R"]))
        info = VGroup(
            VGroup(Rectangle(width=0.3, height=0.18, fill_color=MEAS, fill_opacity=0.7, stroke_width=0),
                   T(f"medida (SUV): erosionada, empieza en el corte {f_meas}", 15, INK)).arrange(RIGHT, buff=0.1),
            VGroup(Rectangle(width=0.3, height=0.18, fill_color=VOLM, fill_opacity=0.5, stroke_width=0),
                   T(f"volumen: sin erosionar, empieza en el corte {f_vol}", 15, INK)).arrange(RIGHT, buff=0.1),
            T("La erosión ya quita los cortes peores (0–4); quedan los cortes 5–15, todavía ruidosos.", 15, INK_SOFT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([-6.95, -2.2, 0], aligned_edge=UL)
        self.play(FadeOut(Group(*P["thumbs"])), FadeOut(VGroup(*P["labs"])), FadeOut(P["sub"]),
                  ReplacementTransform(P["head"], head), run_time=0.6)
        self.play(FadeOut(P["cor"]), FadeIn(ctm), run_time=0.8)
        self.play(FadeIn(o_vol), run_time=0.6)
        self.play(FadeIn(o_meas_keep), FadeIn(o_meas_cut), FadeIn(info), run_time=0.8)
        # where the SUVmax of each side sits
        dots = VGroup()
        for side, lab, where in (("L", "SUVmax izq.", UL), ("R", "SUVmax dcho.", UR)):
            k, _row, col = D[f"max_{side}"][0]
            p = [lg_x(col), lg_y(k + 0.5), 0]
            d = Dot(p, radius=0.07, color="#FFD43B", stroke_color=INK, stroke_width=1.5)
            t = T(f"{lab} (corte {k}, sens {es(SENS_K[k], 2)})", 13, INK)
            t.add_background_rectangle(color="#FFFFFF", opacity=0.85, buff=0.04)
            t.next_to(d, where, buff=0.04)
            dots.add(VGroup(d, t))
        self.play(FadeIn(dots), run_time=0.6)
        self.wait_(5.0)

        # the rule
        head2 = T("La regla: no medir donde el ruido supera ×1,25", 24, INK, bold=True).move_to([-6.95, 3.5, 0], aligned_edge=LEFT)
        kc = (CENTRES[0] - Z0) / 2.0 + 0.5
        y_c = lg_y(kc)
        y_cut = lg_y(N_CUT)
        cline = DashedLine([LG_XL, y_c, 0], [NX1, y_c, 0], color=CENTRE, stroke_width=2.5, dash_length=0.08)
        clab = T("centro de la cama de piernas", 13, CENTRE).move_to([NX1, y_c + 0.15, 0], aligned_edge=RIGHT)
        kline = Line([LG_XL, y_cut, 0], [NX1, y_cut, 0], color=CUT, stroke_width=3.5)
        klab = T(f"plano de corte: {CUT_D:.0f} mm por debajo", 13, CUT).move_to([NX1, y_cut + 0.15, 0], aligned_edge=RIGHT)
        arr = DoubleArrow([LG_XL + LG_W + 0.25, y_c, 0], [LG_XL + LG_W + 0.25, y_cut, 0], buff=0, stroke_width=2.5,
                          tip_length=0.1, color=CUT)
        alab = T(f"{CUT_D:.0f} mm", 13, CUT).next_to(arr, RIGHT, buff=0.05)
        vline = DashedLine([nx(1.25), lg_y(0), 0], [nx(1.25), lg_y(KL), 0], color=CUT, stroke_width=1.5, dash_length=0.06)
        deriv = VGroup(
            T("ruido ≤ ×1,25  ⇔  sens ≥ 0,78 / 1,25² = 0,50  ⇔  d ≤ 82 · (1 − 0,50) = 41 mm del centro de la cama", 15, INK),
            T(f"Con un tag estándar:  z corte = (1.er corte del 2.º grupo de FrameReferenceTime) − 120 mm = {Z_CUT:.0f} mm", 15, INK),
            T(f"En este sujeto quedan por debajo los cortes 0–{N_CUT - 1}: {N_CUT} cortes fuera de la máscara de medida.", 15, CUT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([-6.95, -2.2, 0], aligned_edge=UL)
        self.play(FadeOut(info), ReplacementTransform(head, head2), run_time=0.6)
        self.play(Create(cline), FadeIn(clab), run_time=0.7)
        self.play(Create(vline), FadeIn(deriv[0]), run_time=0.8)
        self.play(Create(arr), FadeIn(alab), Create(kline), FadeIn(klab), run_time=1.0)
        self.play(FadeIn(deriv[1]), run_time=0.5)
        self.wait_(3.0)
        self.play(FadeIn(deriv[2]), FadeOut(o_meas_cut), FadeIn(o_meas_gone), run_time=1.2)
        self.wait_(3.0)

        # effect on the statistics (percent; the SUV factor is a per-series constant)
        sL, sR = D["stats_L"], D["stats_R"]
        cols = ["", "SUVmax", "SUVmean", "p95", "p99", "vóxeles"]
        rows = [["izquierdo"] + [pct(100 * (sL[1, j] / sL[0, j] - 1), 2 if j == 1 else 1) for j in range(5)],
                ["derecho"] + [pct(100 * (sR[1, j] / sR[0, j] - 1), 2 if j == 1 else 1) for j in range(5)]]
        widths = [1.4, 1.15, 1.15, 1.0, 1.0, 1.1]
        table = VGroup()
        for r, row in enumerate([cols] + rows):
            x = -6.95
            for c, txt in enumerate(row):
                t = T(txt, 15, INK, bold=(r == 0 or c == 0))
                t.move_to([x + widths[c] / 2, -2.25 - r * 0.32, 0])
                table.add(t)
                x += widths[c]
        note = VGroup(
            T("Volumen (máscara azul): sin cambios.", 15, VOLM),
            T("Aquí el SUVmax no estaba en la zona ruidosa.", 15, INK_SOFT),
            T("En otro sujeto sí: SUVmax −4,9 % y −16,1 %.", 15, INK_SOFT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-0.15, -2.1, 0], aligned_edge=UL)
        head3 = T("Efecto del recorte en este sujeto", 24, INK, bold=True).move_to([-6.95, 3.5, 0], aligned_edge=LEFT)
        self.play(FadeOut(deriv), ReplacementTransform(head2, head3), run_time=0.5)
        self.play(FadeIn(table), run_time=0.8)
        self.play(FadeIn(note), run_time=0.6)
        self.wait_(6.0)
        self.play(FadeOut(VGroup(head3, table, note, cline, clab, kline, klab, arr, alab, vline, dots, P["plot"],
                                 *P["ticks"])),
                  FadeOut(ctm), FadeOut(o_vol), FadeOut(o_meas_keep), FadeOut(o_meas_gone), run_time=0.8)

    # ------------------------------------------------------------ 8. why it changes between subjects
    def cohort(self):
        C = D["cohort"]          # distance of slice 0 below the leg-bed centre, removed, beds, shown
        head = T("¿Por qué no se quitan los mismos cortes en todos?", 24, INK, bold=True).move_to([-6.95, 3.5, 0], aligned_edge=LEFT)
        y_c, y_e = 2.45, -2.35
        mm = (y_c - y_e) / 82.0

        def yd(d):
            return y_c - d * mm

        xs = np.linspace(-3.6, 4.6, len(C))
        cutl = DashedLine([xs[0] - 1.1, yd(CUT_D), 0], [xs[-1] + 0.9, yd(CUT_D), 0], color=CUT, stroke_width=2.5, dash_length=0.1)
        cutt = T(f"plano de corte ({CUT_D:.0f} mm)", 14, CUT).next_to(cutl, RIGHT, buff=0.1).shift(LEFT * 0.0)
        cutt.move_to([xs[-1] + 1.0, yd(CUT_D) + 0.2, 0], aligned_edge=RIGHT)
        axis = VGroup(Line([xs[0] - 1.25, y_c, 0], [xs[0] - 1.25, y_e, 0], color=INK, stroke_width=1.5))
        for d, txt in ((0, "0 mm · centro"), (20, "20 mm"), (40, "40 mm"), (60, "60 mm"), (82, "82 mm · borde")):
            axis.add(Line([xs[0] - 1.3, yd(d), 0], [xs[0] - 1.2, yd(d), 0], color=INK, stroke_width=1.5))
            axis.add(T(txt, 12, INK).move_to([xs[0] - 1.38, yd(d), 0], aligned_edge=RIGHT))
        axis.add(T("cama de piernas", 12, INK_SOFT).move_to([xs[0] - 1.25, y_c + 0.45, 0], aligned_edge=RIGHT).rotate(0))
        cols = VGroup()
        names = "ABCDE"
        for n, (d0, nrem, nb, shown) in enumerate(C):
            x = xs[n]
            w = 1.1
            # the leg bed's sensitivity: width proportional to 1 - d/82
            tri = Polygon([x - w / 2, y_c, 0], [x + w / 2, y_c, 0], [x, y_e, 0], stroke_width=0,
                          fill_color=BED_A, fill_opacity=0.18)
            body = Rectangle(width=w, height=y_c - yd(d0), stroke_color=INK_FAINT, stroke_width=1.5,
                             fill_opacity=0).move_to([x, (y_c + yd(d0)) / 2, 0])
            end = Line([x - w / 2 - 0.1, yd(d0), 0], [x + w / 2 + 0.1, yd(d0), 0], color=INK, stroke_width=5)
            parts = [tri, body, end]
            if d0 > CUT_D:
                rem = Rectangle(width=w, height=(d0 - CUT_D) * mm, stroke_width=0, fill_color=CUT,
                                fill_opacity=0.45).move_to([x, (yd(CUT_D) + yd(d0)) / 2, 0])
                parts.append(rem)
            top = T(f"sujeto {names[n]}" + (" (el del vídeo)" if shown else ""), 14, INK, bold=bool(shown)).move_to([x, y_c + 0.45, 0])
            bot1 = T(f"último corte a {d0:.0f} mm", 13, INK).move_to([x, y_e - 0.35, 0])
            bot2 = T(f"quita {int(nrem)} cortes", 14, CUT if nrem else GOOD, bold=True).move_to([x, y_e - 0.65, 0])
            cols.add(VGroup(*parts, top, bot1, bot2))
        sub = VGroup(
            T("El plano está fijo respecto a la cama, no al final del volumen. Lo que cambia es dónde cae el último corte:", 15, INK),
            T("el escáner usa el mínimo de camas que cubre el rango planificado y reparte el sobrante entre los dos extremos (inferido).", 15, INK_SOFT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([-6.95, -3.45, 0], aligned_edge=LEFT)
        self.play(FadeIn(head), Create(axis), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(c) for c in cols], lag_ratio=0.2), run_time=2.2)
        self.play(Create(cutl), FadeIn(cutt), run_time=0.7)
        self.play(FadeIn(sub), run_time=0.6)
        self.wait_(7.0)
        self.play(FadeOut(VGroup(head, axis, cols, cutl, cutt, sub)), run_time=0.6)

    # ------------------------------------------------------------ 9. summary
    def summary(self):
        head = T("Qué sabemos, qué suponemos y qué hay que confirmar", 26, INK, bold=True).move_to([0, 3.4, 0])
        blocks = [
            ("Sabemos (DICOM o medido)", GOOD, [
                "camas por FrameReferenceTime",
                "164 mm, paso 100 mm, 210 s,",
                "de cabeza a piernas",
                "ruido ×1,7–2,5 en el último corte",
                "(4 de 5 sujetos)",
                "recorte de 41 mm: p95/p99 ≤ 1,1 %,",
                "media ≈ +1 %",
                "SUVmax cambia en 2 de 6 máscaras",
            ]),
            ("Inferimos o asumimos", MODEL, [
                "centro de cama = tramo constante",
                "de (7053,10D5), 21 mm sobre el grupo",
                "sensibilidad triangular, ruido ∝ 1/√sens",
                "PET reconstruido solo donde hay TC",
                "mínimo de camas, sobrante repartido",
                "umbral ×1,25 → 41 mm (elección)",
            ]),
            ("Confirmar con los técnicos", CUT, [
                "cómo se planifica el rango",
                "y se reparte el sobrante",
                "qué son los tags privados 7053",
                "tratamiento de los planos del borde",
                "si se puede alargar 4–5 cm",
                "por debajo del cuádriceps",
            ]),
        ]
        cols = VGroup()
        for title, col, items in blocks:
            t = T(title, 19, col, bold=True)
            body = VGroup(*[T(("· " if not s.startswith(("de ", "(4", "media", "y se", "por deb", "de (7053")) else "  ") + s, 15, INK)
                            for s in items]).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
            cols.add(VGroup(t, body).arrange(DOWN, aligned_edge=LEFT, buff=0.25))
        cols.arrange(RIGHT, buff=0.6, aligned_edge=UP).move_to([0, 0.1, 0])
        self.play(FadeIn(head), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.1) for c in cols], lag_ratio=0.35), run_time=2.0)
        self.wait_(9.0)
        self.play(FadeOut(VGroup(head, cols)), run_time=0.8)


ORIGIN3 = np.zeros(3)

if __name__ == "__main__":
    run_scene(__file__, "PetBedEdge", "pet_bed_edge")
