"""Manim animation of the qvtpy distal vessel expansion (``blood_flood``).

Renders a voxelised vessel segment and walks through every stage of
:func:`nvitk.segmentation.blood_flood.blood_flood` — Frangi vesselness, the
GMM/hysteresis tree, marker-connected components, the ICA/basilar barrier,
vesselness thinning, and finally the watershed flood that grows the stage-3
proximal seeds out into the distal branches.

Every voxel state shown here comes from ``blood_flood_stages.npz``, which
``blood_flood_precompute.py`` produces by running the real nvitk primitives over
a synthetic phantom. Nothing in this file re-implements the algorithm; it only
draws the arrays. That split also keeps this script numpy-only, so it can run in
a manim environment that has no nvitk installed.

Render::

    BF_FAST=1 manim -ql --fps 30 blood_flood_manim.py BloodFloodDistalExpansion
    manim -qh --fps 30 blood_flood_manim.py BloodFloodDistalExpansion
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import *

# --- Configuration ------------------------------------------------------------

DATA_PATH = Path(
    os.environ.get("BF_STAGES", str(Path(__file__).with_name("blood_flood_stages.npz")))
)
# Preview mode: decimate voxels and shorten beats so layout/timing can be checked
# in a couple of minutes instead of a couple of hours.
FAST = os.environ.get("BF_FAST", "0") == "1"

config.background_color = "#FFFFFF"

FONT = "DejaVu Sans"

INK = "#1B1D23"
INK_SOFT = "#8B9199"
INK_FAINT = "#B9BEC6"

GHOST = "#93A2BC"          # CD volume, unprocessed
VESS_LO = "#CBD6E6"        # low Frangi response
VESS_HI = "#25618F"        # high Frangi response
TREE = "#5E6E8A"           # binary hysteresis tree
HIGH_SEED = "#EFA43A"      # V > high  (hysteresis anchors)
SEED = "#F0761B"           # stage-3 marker labels
MCA = "#DE4133"            # watershed label 1
ACA = "#158C81"            # watershed label 2
BARRIER = "#2A2E39"        # dilated ICA / basilar wall
GREEN = "#2E9E4F"

OBJECT_WIDTH = 7.6         # scene units across the longest voxel axis
CUBE_FILL_RATIO = 0.86     # cube side / voxel pitch, leaves the reference's hairline gaps
STROKE_W = 0.5


def _hex(rgb: np.ndarray) -> str:
    r, g, b = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def _rgb(color: str) -> np.ndarray:
    c = color.lstrip("#")
    return np.array([int(c[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _smoothstep(f: np.ndarray) -> np.ndarray:
    return f * f * (3.0 - 2.0 * f)


class VoxelField:
    """A cloud of cubes whose fill/stroke is driven from numpy state arrays.

    Manim re-rasterises the whole scene whenever any mobject changes, so the only
    thing worth optimising is how much work the per-frame updater does. States are
    therefore quantised and only cubes whose quantised style actually moved get
    touched, and colours are cached by hex string.
    """

    def __init__(self, coords: np.ndarray, pitch: float) -> None:
        self.coords = coords
        n = len(coords)
        side = pitch * CUBE_FILL_RATIO
        centre = (coords.max(axis=0) + coords.min(axis=0)) / 2.0
        self.cubes: list[Cube] = []
        for p in coords:
            cube = Cube(side_length=side, fill_opacity=0.0, fill_color=GHOST)
            cube.set_stroke(color=GHOST, width=STROKE_W, opacity=0.0)
            cube.move_to((np.asarray(p, dtype=float) - centre) * pitch)
            self.cubes.append(cube)
        self.group = VGroup(*self.cubes)

        self.rgb = np.tile(_rgb(GHOST), (n, 1))
        self.op = np.zeros(n)
        self.sop = np.zeros(n)
        self._quant = np.full((n, 5), -1, dtype=np.int16)
        self._colors: dict[tuple[int, int, int], ManimColor] = {}

    def _color(self, key: tuple[int, int, int]) -> ManimColor:
        col = self._colors.get(key)
        if col is None:
            col = ManimColor(_hex(np.asarray(key, dtype=float) / 63.0))
            self._colors[key] = col
        return col

    def apply(self, rgb: np.ndarray, op: np.ndarray, sop: np.ndarray) -> None:
        q = np.empty((len(self.cubes), 5), dtype=np.int16)
        q[:, :3] = np.clip(np.rint(rgb * 63.0), 0, 63)
        q[:, 3] = np.clip(np.rint(op * 48.0), 0, 48)
        q[:, 4] = np.clip(np.rint(sop * 48.0), 0, 48)
        for i in np.nonzero((q != self._quant).any(axis=1))[0]:
            i = int(i)
            col = self._color((int(q[i, 0]), int(q[i, 1]), int(q[i, 2])))
            cube = self.cubes[i]
            cube.set_fill(col, opacity=float(q[i, 3]) / 48.0)
            cube.set_stroke(color=col, opacity=float(q[i, 4]) / 48.0)
        self._quant = q
        self.rgb, self.op, self.sop = rgb, op, sop

    def transition(
        self,
        scene: Scene,
        rgb: np.ndarray,
        op: np.ndarray,
        sop: np.ndarray,
        *,
        run_time: float = 2.0,
        delay: np.ndarray | None = None,
        lag: float = 0.0,
        glow: str | None = None,
        glow_strength: float = 1.0,
        hook=None,
    ) -> None:
        """Interpolate to a target style, optionally as a delayed wave with a glow."""
        rgb0, op0, sop0 = self.rgb.copy(), self.op.copy(), self.sop.copy()
        rgb1, op1, sop1 = np.asarray(rgb), np.asarray(op), np.asarray(sop)

        if delay is None or lag <= 0.0:
            d = np.zeros(len(self.cubes))
        else:
            d = np.asarray(delay, dtype=float)
            span = float(d.max() - d.min())
            d = (d - d.min()) / span * lag if span > 0 else np.zeros_like(d)
        width = max(1e-6, 1.0 - float(d.max()))
        glow_rgb = None if glow is None else _rgb(glow)

        tracker = ValueTracker(0.0)
        scene.add(tracker)

        def update(_: Mobject) -> None:
            f = _smoothstep(np.clip((tracker.get_value() - d) / width, 0.0, 1.0))
            fc = f[:, None]
            cur = rgb0 + (rgb1 - rgb0) * fc
            if glow_rgb is not None:
                g = (4.0 * f * (1.0 - f) * glow_strength)[:, None]
                cur = cur + (glow_rgb - cur) * g
            self.apply(cur, op0 + (op1 - op0) * f, sop0 + (sop1 - sop0) * f)
            if hook is not None:
                hook(f)

        self.group.add_updater(update)
        scene.play(tracker.animate.set_value(1.0), run_time=run_time, rate_func=linear)
        self.group.remove_updater(update)
        scene.remove(tracker)
        self.apply(rgb1, op1, sop1)


class BloodFloodDistalExpansion(ThreeDScene):
    """The full stage-by-stage walkthrough."""

    # -- helpers ---------------------------------------------------------------

    def fixed(self, *mobs: Mobject) -> None:
        """Pin mobjects to the camera frame without showing them yet."""
        self.add_fixed_in_frame_mobjects(*mobs)
        self.remove(*mobs)

    def swap(self, old: Mobject | None, new: Mobject | None, run_time: float = 0.85) -> None:
        if old is not None:
            self.play(FadeOut(old, shift=DOWN * 0.12), run_time=run_time * 0.45)
        if new is not None:
            self.play(FadeIn(new, shift=UP * 0.12), run_time=run_time * 0.55)

    def caption(
        self,
        step: str,
        lines: list[tuple[str, str, float]],
    ) -> VGroup:
        """Bottom-left text block: a step heading over colour-coded detail lines."""
        head = Text(step, font=FONT, weight=BOLD, color=INK, font_size=25)
        body = VGroup(
            *[Text(t, font=FONT, color=c, font_size=s) for t, c, s in lines]
        ).arrange(DOWN, buff=0.17, aligned_edge=LEFT)
        block = VGroup(head, body).arrange(DOWN, buff=0.26, aligned_edge=LEFT)
        block.to_corner(DL, buff=0.55)
        self.fixed(block)
        return block

    # -- the mini vesselness histogram ----------------------------------------

    def histogram_panel(self, meta: dict) -> VGroup:
        counts = np.asarray(meta["hist_counts"], dtype=float)
        edges = np.asarray(meta["hist_edges"], dtype=float)
        lowt, hight = float(meta["lowt"]), float(meta["hight"])

        w, h = 3.3, 1.15
        heights = (counts / max(counts.max(), 1e-9)) ** 0.32
        heights = np.maximum(heights, 0.03)

        def x_of(v: float) -> float:
            return (float(v) - edges[0]) / (edges[-1] - edges[0]) * w

        bars = VGroup()
        bw = w / len(counts)
        for i, hh in enumerate(heights):
            if hh <= 0.0:
                continue
            bar = Rectangle(
                width=bw * 0.86,
                height=max(hh * h, 1e-3),
                stroke_width=0,
                fill_color=INK_FAINT,
                fill_opacity=1.0,
            )
            bar.move_to(np.array([i * bw + bw / 2.0, hh * h / 2.0, 0.0]))
            bars.add(bar)

        axis = Line(ORIGIN, RIGHT * w, stroke_width=1.6, color=INK_SOFT)

        def rule(v: float, color: str, label: str, up: float) -> VGroup:
            x = x_of(v)
            ln = DashedLine(
                np.array([x, 0.0, 0.0]),
                np.array([x, h * up, 0.0]),
                dash_length=0.045,
                stroke_width=2.2,
                color=color,
            )
            tx = Text(label, font=FONT, color=color, font_size=17)
            tx.next_to(ln.get_end(), UP, buff=0.06)
            return VGroup(ln, tx)

        low_rule = rule(lowt, GREEN, f"low {lowt:.3f}", 0.95)
        high_rule = rule(hight, SEED, f"high {hight:.3f}", 0.62)

        title = Text("Frangi vesselness  V(x)", font=FONT, color=INK_SOFT, font_size=17)
        title.next_to(axis, DOWN, buff=0.12).align_to(axis, LEFT)

        panel = VGroup(bars, axis, low_rule, high_rule, title)
        panel.to_corner(DR, buff=0.6)
        self.fixed(panel)
        return panel

    # -- main ------------------------------------------------------------------

    def construct(self) -> None:
        data = np.load(DATA_PATH)
        meta = json.loads(str(data["meta"]))

        display = np.asarray(data["display"], dtype=bool)
        coords = np.argwhere(display)
        if FAST:
            keep = np.random.default_rng(0).random(len(coords)) < 0.30
            coords = coords[keep]
        idx = tuple(coords.T)

        def sel(key: str) -> np.ndarray:
            return np.asarray(data[key])[idx]

        pitch = OBJECT_WIDTH / float(np.ptp(coords, axis=0).max())
        field = VoxelField(coords, pitch)

        intensity = sel("intensity").astype(float)
        vess = sel("vesselness").astype(float)
        tree_hyst = sel("tree_hyst").astype(bool)
        tree_cc = sel("tree_cc").astype(bool)
        dropped = sel("dropped_cc").astype(bool)
        barrier = sel("barrier").astype(bool)
        tree_barrier = sel("tree_barrier").astype(bool)
        tree_final = sel("tree_final").astype(bool)
        removed_thin = sel("removed_by_thin").astype(bool)
        mask_high = sel("mask_high").astype(bool)
        markers = sel("markers").astype(int)
        labels = sel("labels").astype(int)
        order = sel("order").astype(float)

        n = len(coords)
        i_norm = np.clip(
            (intensity - intensity.min()) / max(np.ptp(intensity), 1e-9), 0.0, 1.0
        )
        v_norm = np.clip(vess / max(vess.max(), 1e-9), 0.0, 1.0)

        def const(color: str) -> np.ndarray:
            return np.tile(_rgb(color), (n, 1))

        def blend(c0: str, c1: str, t: np.ndarray) -> np.ndarray:
            return _rgb(c0) + (_rgb(c1) - _rgb(c0)) * t[:, None]

        # ---------------------------------------------------------------- style
        # Every stage is one (rgb, opacity, stroke-opacity) triple over the field.

        ghost_rgb = const(GHOST)
        ghost_op = 0.055 + 0.20 * i_norm
        ghost_sop = 0.10 + 0.20 * i_norm

        vess_rgb = blend(VESS_LO, VESS_HI, v_norm)
        vess_op = 0.045 + 0.62 * v_norm
        vess_sop = 0.08 + 0.42 * v_norm

        self.set_camera_orientation(
            phi=70 * DEGREES, theta=-99 * DEGREES, frame_center=[0.0, 0.0, -0.95]
        )
        self.add(field.group)
        self.begin_ambient_camera_rotation(rate=0.013)

        T = (lambda x: max(0.5, x * 0.28)) if FAST else (lambda x: x)

        # -------------------------------------------------------------- 0 title
        title = Text("Distal vessel expansion", font=FONT, weight=BOLD,
                     color=INK, font_size=34)
        subtitle = Text("nvitk.segmentation.blood_flood   ·   qvtpy stage 4",
                        font=FONT, color=INK_SOFT, font_size=21)
        card = VGroup(title, subtitle).arrange(DOWN, buff=0.2, aligned_edge=LEFT)
        card.to_corner(DL, buff=0.55)
        self.fixed(card)

        self.play(FadeIn(card), run_time=T(1.0))
        field.transition(self, ghost_rgb, ghost_op, ghost_sop,
                         run_time=T(2.6), delay=coords[:, 0].astype(float), lag=0.75)
        self.wait(T(1.0))

        # ------------------------------------------------------------ 1 CD volume
        cap = self.caption(
            "1 · Bright-blood volume",
            [
                ("I(x)  —  4D-flow complex difference", INK, 21),
                ("arteries, veins and parenchymal blobs", INK_SOFT, 19),
                ("are all bright: intensity alone cannot separate them", INK_SOFT, 19),
            ],
        )
        self.swap(card, cap)
        self.wait(T(2.4))

        # ----------------------------------------------------- 2 Frangi vesselness
        cap2 = self.caption(
            "2 · Frangi vesselness",
            [
                ("V(x) = max over σ of  Frangi(λ₁, λ₂, λ₃)", INK, 21),
                ("σ ∈ {0.5, 1.0, 1.5, 2.0, 2.5} voxels", INK_SOFT, 19),
                ("tubular → 1        blob / noise → 0", VESS_HI, 19),
            ],
        )
        self.swap(cap, cap2)
        field.transition(self, vess_rgb, vess_op, vess_sop, run_time=T(3.0))
        self.wait(T(2.0))

        # -------------------------------------------- 3 GMM + hysteresis tree
        hist = self.histogram_panel(meta)
        cap3 = self.caption(
            "3 · GMM + hysteresis tree",
            [
                ("3-component GMM on V > 0", INK, 21),
                (f"low  = μ₂ + 3.5·σ₂ = {meta['lowt']:.3f}", GREEN, 19),
                (f"high = μ₃ + 0.5·σ₃ = {meta['hight']:.3f}", SEED, 19),
                ("tree = components of {V > low} that touch {V > high}", INK_SOFT, 19),
            ],
        )
        self.swap(cap2, cap3)
        self.play(FadeIn(hist), run_time=T(0.8))

        # anchors first: V > high
        anchor_rgb = np.where(mask_high[:, None], _rgb(HIGH_SEED), vess_rgb)
        anchor_op = np.where(mask_high, 0.88, vess_op * 0.35)
        anchor_sop = np.where(mask_high, 0.85, vess_sop * 0.35)
        field.transition(self, anchor_rgb, anchor_op, anchor_sop, run_time=T(1.6))
        self.wait(T(1.2))

        # then grow the low-threshold components that touch them
        tree_rgb = np.where(tree_hyst[:, None], _rgb(TREE), _rgb(GHOST))
        tree_op = np.where(tree_hyst, 0.52, 0.035)
        tree_sop = np.where(tree_hyst, 0.55, 0.06)
        field.transition(self, tree_rgb, tree_op, tree_sop, run_time=T(2.6),
                         delay=-v_norm, lag=0.6, glow=HIGH_SEED, glow_strength=0.55)
        self.wait(T(1.6))

        # --------------------------------------------- 4 marker-connected CCs
        cap4 = self.caption(
            "4 · Keep marker-connected components",
            [
                ("seeds = stage-3 MCA / ACA labels", SEED, 21),
                (f"{meta['n_cc_kept']} of {meta['n_cc_total']} components touch a seed", INK, 19),
                (f"−{meta['n_dropped_cc']} voxels: vein and blobs drop out", INK_SOFT, 19),
            ],
        )
        self.swap(cap3, cap4)

        seed_mask = markers != 0
        m_rgb = np.where(seed_mask[:, None], _rgb(SEED), tree_rgb)
        m_op = np.where(seed_mask, 0.92, tree_op)
        m_sop = np.where(seed_mask, 0.9, tree_sop)
        field.transition(self, m_rgb, m_op, m_sop, run_time=T(1.4))
        self.wait(T(1.0))

        cc_rgb = np.where(seed_mask[:, None], _rgb(SEED),
                          np.where(tree_cc[:, None], _rgb(TREE), _rgb(GHOST)))
        cc_op = np.where(seed_mask, 0.92, np.where(tree_cc, 0.52, 0.028))
        cc_sop = np.where(seed_mask, 0.9, np.where(tree_cc, 0.55, 0.05))
        field.transition(self, cc_rgb, cc_op, cc_sop, run_time=T(2.2),
                         delay=np.where(dropped, 1.0, 0.0), lag=0.35)
        self.wait(T(1.4))

        # ------------------------------------------------------- 5 hard barrier
        cap5 = self.caption(
            "5 · Hard barrier",
            [
                ("ICA / basilar, dilated by 1 voxel", BARRIER, 21),
                ("tree ← (tree ∧ ¬barrier) ∨ seeds", INK, 19),
                (f"−{meta['n_removed_barrier']} voxels; seeds are forced back in", INK_SOFT, 19),
            ],
        )
        self.swap(cap4, cap5)

        b_rgb = np.where(barrier[:, None], _rgb(BARRIER), cc_rgb)
        b_op = np.where(barrier, 0.80, cc_op)
        b_sop = np.where(barrier, 0.92, cc_sop)
        field.transition(self, b_rgb, b_op, b_sop, run_time=T(1.4))
        self.wait(T(0.9))

        tb_rgb = np.where(barrier[:, None], _rgb(BARRIER),
                          np.where(seed_mask[:, None], _rgb(SEED),
                                   np.where(tree_barrier[:, None], _rgb(TREE), _rgb(GHOST))))
        tb_op = np.where(barrier, 0.80,
                         np.where(seed_mask, 0.92, np.where(tree_barrier, 0.52, 0.028)))
        tb_sop = np.where(barrier, 0.92,
                          np.where(seed_mask, 0.9, np.where(tree_barrier, 0.55, 0.05)))
        field.transition(self, tb_rgb, tb_op, tb_sop, run_time=T(1.6))
        self.wait(T(1.2))

        # --------------------------------------------------- 6 vesselness thinning
        cap6 = self.caption(
            "6 · Vesselness thinning",
            [
                (f"drop the weak Frangi shell: V < p55 = {meta['thin_threshold']:.3f}", INK, 21),
                ("seeds are protected from thinning", SEED, 19),
                (f"−{meta['n_removed_thin']} voxels", INK_SOFT, 19),
            ],
        )
        self.swap(cap5, cap6)

        tf_rgb = np.where(barrier[:, None], _rgb(BARRIER),
                          np.where(seed_mask[:, None], _rgb(SEED),
                                   np.where(tree_final[:, None], _rgb(TREE), _rgb(GHOST))))
        tf_op = np.where(barrier, 0.52,
                         np.where(seed_mask, 0.92, np.where(tree_final, 0.55, 0.026)))
        tf_sop = np.where(barrier, 0.62,
                          np.where(seed_mask, 0.9, np.where(tree_final, 0.58, 0.045)))
        field.transition(self, tf_rgb, tf_op, tf_sop, run_time=T(2.2),
                         delay=np.where(removed_thin, 1.0, 0.0), lag=0.35)
        self.wait(T(1.4))

        # ------------------------------------------------------ 7 watershed flood
        self.play(FadeOut(hist), run_time=T(0.6))
        cap7 = self.caption(
            "7 · Watershed flood",
            [
                ("seeds flood the binary tree, watershed on −EDT", INK, 21),
                ("the front follows the lumen and never leaves the tree", INK_SOFT, 19),
            ],
        )
        legend = VGroup(
            VGroup(Square(0.17, fill_color=MCA, fill_opacity=1, stroke_width=0),
                   Text("MCA", font=FONT, color=INK, font_size=19)).arrange(RIGHT, buff=0.13),
            VGroup(Square(0.17, fill_color=ACA, fill_opacity=1, stroke_width=0),
                   Text("ACA", font=FONT, color=INK, font_size=19)).arrange(RIGHT, buff=0.13),
        ).arrange(RIGHT, buff=0.6)
        legend.next_to(cap7, DOWN, buff=0.22).align_to(cap7, LEFT)
        legend.shift(UP * 0.05)
        self.fixed(legend)
        self.swap(cap6, cap7)
        self.play(FadeIn(legend), run_time=T(0.5))

        labeled = labels != 0
        label_rgb = np.where(labels[:, None] == meta["label_mca"], _rgb(MCA),
                             np.where(labels[:, None] == meta["label_aca"], _rgb(ACA), tf_rgb))
        flood_rgb = np.where(labeled[:, None], label_rgb, tf_rgb)
        flood_op = np.where(labeled, 0.9, tf_op)
        flood_sop = np.where(labeled, 0.88, tf_sop)
        # Seeds are already claimed at t=0; everything else waits its geodesic turn.
        flood_delay = np.where(labeled, np.maximum(order, 0.0), 0.0)

        counter = VGroup(
            Integer(0, font_size=30, color=INK, edge_to_fix=RIGHT),
            Text("voxels claimed", font=FONT, color=INK_SOFT, font_size=19),
        ).arrange(RIGHT, buff=0.16)
        counter.to_corner(DR, buff=0.7)
        self.fixed(counter)
        self.play(FadeIn(counter), run_time=T(0.4))

        n_labeled = int(meta["n_labeled"])
        scale = n_labeled / max(1, int(labeled.sum()))

        def tick(f: np.ndarray) -> None:
            claimed = int(round(float(np.count_nonzero(f[labeled] > 0.5)) * scale))
            counter[0].set_value(claimed)
            # set_value rebuilds glyph submobjects, which are not yet frame-fixed.
            self.add_fixed_in_frame_mobjects(counter)

        field.transition(self, flood_rgb, flood_op, flood_sop, run_time=T(10.0),
                         delay=flood_delay, lag=0.88, glow=HIGH_SEED,
                         glow_strength=0.75, hook=tick)
        counter[0].set_value(n_labeled)
        self.add_fixed_in_frame_mobjects(counter)
        self.wait(T(1.6))

        # ------------------------------------------------------------- 8 result
        cap8 = self.caption(
            "Distal expansion complete",
            [
                (f"{meta['n_markers']} seed voxels  →  {n_labeled} labelled", INK, 21),
                (f"{n_labeled / max(1, meta['n_markers']):.1f}× extension into the distal branches",
                 INK_SOFT, 19),
                ("watershed splits the shared distal field between labels", INK_SOFT, 19),
            ],
        )
        self.swap(cap7, cap8)
        self.wait(T(3.2))
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(cap8), FadeOut(legend), FadeOut(counter), run_time=T(1.0))
        self.wait(T(0.6))
