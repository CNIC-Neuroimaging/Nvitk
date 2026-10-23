"""Shared voxel-cloud rendering for the nvitk Manim animations (numpy + manim only).

:class:`VoxelField` draws a set of cubes whose colour, opacity, stroke and *size* are driven
from numpy state arrays and interpolated between states (optionally as a delayed wave with a
glow). Size encodes how much of a display block a mask fills, so thinning and growth read as
cubes shrinking and appearing. :class:`CaptionScene` adds the caption / swap helpers the scenes
share. The style follows ``blood_flood_manim.py``: white background, ink text, a bottom-left
caption per step.

Scenes import this module from their own folder (``sys.path`` is set by the scene file), so it
needs nothing but numpy and manim.
"""

from __future__ import annotations

import numpy as np
from manim import (
    BOLD,
    DL,
    DOWN,
    LEFT,
    UP,
    Cube,
    FadeIn,
    FadeOut,
    ManimColor,
    Mobject,
    Text,
    ThreeDScene,
    ValueTracker,
    VGroup,
    linear,
)

FONT = "DejaVu Sans"

INK = "#1B1D23"
INK_SOFT = "#8B9199"
INK_FAINT = "#B9BEC6"

CUBE_FILL_RATIO = 0.86  # cube side / voxel pitch: the hairline gaps of the reference style
STROKE_W = 0.5


def rgb(color: str) -> np.ndarray:
    """``#RRGGBB`` → float RGB in [0, 1]."""
    c = color.lstrip("#")
    return np.array([int(c[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def hex_of(values: np.ndarray) -> str:
    """Float RGB in [0, 1] → ``#RRGGBB``."""
    r, g, b = (np.clip(values, 0.0, 1.0) * 255.0).astype(int)
    return f"#{r:02X}{g:02X}{b:02X}"


def smoothstep(f: np.ndarray) -> np.ndarray:
    """Cubic ease, 0 → 1."""
    return f * f * (3.0 - 2.0 * f)


def ramp(c0: str, c1: str, t: np.ndarray) -> np.ndarray:
    """Per-element colour between *c0* and *c1* at *t* ∈ [0, 1]."""
    return rgb(c0) + (rgb(c1) - rgb(c0)) * np.clip(t, 0.0, 1.0)[:, None]


def ramp3(c0: str, c1: str, c2: str, t: np.ndarray) -> np.ndarray:
    """Three-stop colour ramp at *t* ∈ [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    lo = ramp(c0, c1, np.clip(t * 2.0, 0.0, 1.0))
    hi = ramp(c1, c2, np.clip(t * 2.0 - 1.0, 0.0, 1.0))
    return np.where((t < 0.5)[:, None], lo, hi)


class VoxelField:
    """A cloud of cubes whose fill / stroke / size come from numpy state arrays.

    Manim re-rasterises the scene whenever a mobject changes, so the per-frame updater only
    touches cubes whose quantised style moved, and colours are cached.

    Parameters
    ----------
    positions
        ``(N, 3)`` cube centres in scene units.
    side
        Full cube side (scene units) at size 1.
    """

    SIZE_LEVELS = 24

    def __init__(self, positions: np.ndarray, side: float, base_color: str = "#93A2BC") -> None:
        n = len(positions)
        self.side = float(side) * CUBE_FILL_RATIO
        self.cubes: list[Cube] = []
        for p in positions:
            cube = Cube(side_length=self.side, fill_opacity=0.0, fill_color=base_color)
            cube.set_stroke(color=base_color, width=STROKE_W, opacity=0.0)
            cube.move_to(np.asarray(p, dtype=float))
            self.cubes.append(cube)
        self.group = VGroup(*self.cubes)
        self.rgb = np.tile(rgb(base_color), (n, 1))
        self.op = np.zeros(n)
        self.sop = np.zeros(n)
        self.size = np.ones(n)
        self._quant = np.full((n, 5), -1, dtype=np.int16)
        self._size_q = np.full(n, self.SIZE_LEVELS, dtype=np.int16)  # cubes are built at size 1
        self._colors: dict[tuple[int, int, int], ManimColor] = {}

    def __len__(self) -> int:
        return len(self.cubes)

    def _color(self, key: tuple[int, int, int]) -> ManimColor:
        col = self._colors.get(key)
        if col is None:
            col = ManimColor(hex_of(np.asarray(key, dtype=float) / 63.0))
            self._colors[key] = col
        return col

    def apply(self, rgb_: np.ndarray, op: np.ndarray, sop: np.ndarray, size: np.ndarray | None = None) -> None:
        """Set the style of every cube (only the changed ones are touched)."""
        size = self.size if size is None else np.asarray(size, dtype=float)
        q = np.empty((len(self.cubes), 5), dtype=np.int16)
        q[:, :3] = np.clip(np.rint(rgb_ * 63.0), 0, 63)
        q[:, 3] = np.clip(np.rint(op * 48.0), 0, 48)
        q[:, 4] = np.clip(np.rint(sop * 48.0), 0, 48)
        sq = np.clip(np.rint(size * self.SIZE_LEVELS), 1, self.SIZE_LEVELS).astype(np.int16)
        for i in np.nonzero((q != self._quant).any(axis=1) | (sq != self._size_q))[0]:
            i = int(i)
            cube = self.cubes[i]
            if sq[i] != self._size_q[i]:
                cube.scale(float(sq[i]) / float(self._size_q[i]))
            col = self._color((int(q[i, 0]), int(q[i, 1]), int(q[i, 2])))
            cube.set_fill(col, opacity=float(q[i, 3]) / 48.0)
            cube.set_stroke(color=col, opacity=float(q[i, 4]) / 48.0)
        self._quant = q
        self._size_q = sq
        self.rgb, self.op, self.sop, self.size = rgb_, op, sop, size

    def transition(
        self,
        scene: ThreeDScene,
        rgb_: np.ndarray,
        op: np.ndarray,
        sop: np.ndarray,
        size: np.ndarray | None = None,
        *,
        run_time: float = 2.0,
        delay: np.ndarray | None = None,
        lag: float = 0.0,
        glow: str | None = None,
        glow_strength: float = 1.0,
        hook=None,
        extra: list | None = None,
    ) -> None:
        """Interpolate to a target style, optionally as a delayed wave with a glow. *extra*
        animations play alongside."""
        rgb0, op0, sop0, size0 = self.rgb.copy(), self.op.copy(), self.sop.copy(), self.size.copy()
        rgb1, op1, sop1 = np.asarray(rgb_), np.asarray(op), np.asarray(sop)
        size1 = size0 if size is None else np.asarray(size, dtype=float)
        if delay is None or lag <= 0.0:
            d = np.zeros(len(self.cubes))
        else:
            d = np.asarray(delay, dtype=float)
            span = float(d.max() - d.min())
            d = (d - d.min()) / span * lag if span > 0 else np.zeros_like(d)
        width = max(1e-6, 1.0 - float(d.max()))
        glow_rgb = None if glow is None else rgb(glow)
        tracker = ValueTracker(0.0)
        scene.add(tracker)

        def update(_: Mobject) -> None:
            f = smoothstep(np.clip((tracker.get_value() - d) / width, 0.0, 1.0))
            fc = f[:, None]
            cur = rgb0 + (rgb1 - rgb0) * fc
            if glow_rgb is not None:
                g = (4.0 * f * (1.0 - f) * glow_strength)[:, None]
                cur = cur + (glow_rgb - cur) * g
            self.apply(cur, op0 + (op1 - op0) * f, sop0 + (sop1 - sop0) * f, size0 + (size1 - size0) * f)
            if hook is not None:
                hook(f)

        self.group.add_updater(update)
        scene.play(tracker.animate.set_value(1.0), *(extra or []), run_time=run_time, rate_func=linear)
        self.group.remove_updater(update)
        scene.remove(tracker)
        self.apply(rgb1, op1, sop1, size1)


class CaptionScene(ThreeDScene):
    """ThreeDScene with the shared caption helpers."""

    def fixed(self, *mobs: Mobject) -> None:
        """Pin mobjects to the camera frame without showing them yet."""
        self.add_fixed_in_frame_mobjects(*mobs)
        self.remove(*mobs)

    def swap(self, old: Mobject | None, new: Mobject | None, run_time: float = 0.85) -> None:
        """Fade *old* out (down) and *new* in (up)."""
        if old is not None:
            self.play(FadeOut(old, shift=DOWN * 0.12), run_time=run_time * 0.45)
        if new is not None:
            self.play(FadeIn(new, shift=UP * 0.12), run_time=run_time * 0.55)

    def caption(self, step: str, lines: list[tuple[str, str, float]], corner=DL) -> VGroup:
        """Bottom-left text block: a step heading over colour-coded detail lines."""
        head = Text(step, font=FONT, weight=BOLD, color=INK, font_size=25)
        body = VGroup(*[Text(t, font=FONT, color=c, font_size=s) for t, c, s in lines]).arrange(
            DOWN, buff=0.15, aligned_edge=LEFT
        )
        block = VGroup(head, body).arrange(DOWN, buff=0.24, aligned_edge=LEFT)
        block.to_corner(corner, buff=0.5)
        # A white backing keeps the text readable over the voxels behind it.
        block.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.16)
        self.fixed(block)
        return block


__all__ = [
    "CUBE_FILL_RATIO",
    "CaptionScene",
    "FONT",
    "INK",
    "INK_FAINT",
    "INK_SOFT",
    "VoxelField",
    "hex_of",
    "ramp",
    "ramp3",
    "rgb",
    "smoothstep",
]
