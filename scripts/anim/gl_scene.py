"""Manim scenes whose 3D content is drawn on the GPU (:mod:`gl_render`).

:class:`GLScene` is a 2D manim ``Scene``: every frame, its camera's background *is* the
GPU-rendered 3D frame, and manim draws its text (captions, legends, insets, plots) over it. The
look and helpers follow :mod:`anim_voxels` (white page, ink text, bottom-left captions, ``swap``,
``--no-text`` keeping every text element but transparent).

The 3D side is driven from the scene:

- ``self.gl``: the :class:`gl_render.GLRenderer` (``self.gl.add(layer)``);
- ``self.cam``: its orbit camera; :meth:`GLScene.move_camera` animates it,
  :meth:`GLScene.orbit` turns it slowly about the vertical axis while everything else plays;
- :meth:`GLScene.animate_gl` runs any per-frame function of the animation's progress (alongside
  ordinary manim animations), and :meth:`GLScene.voxel_transition` is
  :meth:`anim_voxels.VoxelField.transition` for :class:`gl_render.Voxels` (delayed wave + glow);
- :meth:`GLScene.label_3d` pins a text label to a world point.

A driver mobject with an updater keeps every mobject "moving", so manim never caches a static
frame, which would freeze the 3D background.
"""

from __future__ import annotations

import numpy as np
from manim import (
    BOLD,
    Axes,
    DL,
    DOWN,
    LEFT,
    UP,
    Camera,
    FadeIn,
    FadeOut,
    Mobject,
    Scene,
    Text,
    ValueTracker,
    VGroup,
    config,
    linear,
    smooth,
)

from anim_voxels import FONT, INK, INK_SOFT, no_text, rgb, smoothstep  # noqa: F401  (re-exported)
from gl_render import GLRenderer, OrbitCamera, Voxels

config.background_color = "#FFFFFF"


class _GLCamera(Camera):
    """Manim camera whose background is the GPU frame of the moment."""

    gl_source = None

    def reset(self):
        if self.gl_source is not None:
            frame = self.gl_source()
            if frame.shape[:2] == (self.pixel_height, self.pixel_width):
                self.background = frame
        return super().reset()


def ease(f: float) -> float:
    return float(smooth(float(np.clip(f, 0.0, 1.0))))


class GLScene(Scene):
    """A manim scene with a GPU 3D layer underneath (module docstring)."""

    def __init__(self, **kwargs):
        # manim's partial-movie cache hashes the mobjects, not the GPU layer: a re-render after a
        # change to the 3D state alone would splice in stale clips
        config.disable_caching = True
        super().__init__(camera_class=_GLCamera, **kwargs)

    # -- set-up ---------------------------------------------------------------------------------
    def setup(self) -> None:
        super().setup()
        self.gl = GLRenderer(config.pixel_width, config.pixel_height, samples=8)
        self.cam: OrbitCamera = self.gl.camera
        self.orbit_rate = 0.0  # deg / s about the vertical axis
        self._orbit_acc = 0.0
        self._frame_hooks: list = []
        self.renderer.camera.gl_source = self.gl.render
        driver = Mobject()

        def drive(_m: Mobject, dt: float) -> None:
            if self.orbit_rate:
                self.cam.theta += self.orbit_rate * dt
                self._orbit_acc += self.orbit_rate * dt
            for hook in list(self._frame_hooks):
                hook(dt)

        driver.add_updater(drive)
        self.add(driver)
        self._driver = driver

    def add_frame_hook(self, fn) -> None:
        """``fn(dt)`` every frame (a continuously moving 3D element: particles, a pulse)."""
        self._frame_hooks.append(fn)

    def remove_frame_hook(self, fn) -> None:
        if fn in self._frame_hooks:
            self._frame_hooks.remove(fn)

    # -- camera ---------------------------------------------------------------------------------
    def orbit(self, rate_deg_s: float) -> None:
        """Turn the camera about the vertical axis at *rate_deg_s* (0 stops)."""
        self.orbit_rate = float(rate_deg_s)

    def set_camera(self, **params) -> None:
        self.cam.set(**params)

    def move_camera(self, run_time: float = 2.0, extra: list | None = None, **target) -> None:
        """Animate the camera to *target* (any of target, distance, phi, theta, fov, shift). The
        ambient orbit keeps adding to theta meanwhile."""
        start = self.cam.params()
        acc0 = self._orbit_acc

        def fn(f: float) -> None:
            e = ease(f)
            for k, v in target.items():
                a = np.asarray(start[k], float)
                val = a + (np.asarray(v, float) - a) * e
                if k == "theta":
                    val = val + (self._orbit_acc - acc0)
                self.cam.set(**{k: val})

        self.animate_gl(fn, run_time=run_time, extra=extra)

    # -- generic 3D animation ------------------------------------------------------------------
    def animate_gl(self, fn, run_time: float = 1.0, extra: list | None = None, rate=linear) -> None:
        """Call ``fn(f)`` with the progress f ∈ [0, 1] every frame of a *run_time* animation;
        *extra* manim animations play alongside."""
        tracker = ValueTracker(0.0)
        holder = Mobject()

        def upd(_m: Mobject) -> None:
            fn(float(rate(float(np.clip(tracker.get_value(), 0.0, 1.0)))))

        holder.add_updater(upd)
        self.add(tracker, holder)
        self.play(tracker.animate.set_value(1.0), *(extra or []), run_time=run_time, rate_func=linear)
        holder.remove_updater(upd)
        self.remove(tracker, holder)
        fn(1.0)

    def hold(self, seconds: float) -> None:
        """Wait, rendering every frame (the orbit and the frame hooks go on)."""
        if seconds > 0:
            self.wait(seconds)

    # -- voxels ---------------------------------------------------------------------------------
    def voxel_transition(self, vox: Voxels, rgb_=None, op=None, size=None, *, run_time: float = 2.0,
                         delay: np.ndarray | None = None, lag: float = 0.0, glow: str | None = None,
                         glow_strength: float = 1.0, extra: list | None = None, hook=None) -> None:
        """Interpolate *vox* to a target style, optionally as a delayed wave (*delay* per voxel,
        spread over *lag* of the run) with a passing *glow* colour."""
        rgb0, op0, size0 = np.array(vox.rgb, "f4"), np.array(vox.op, "f4"), np.array(vox.size, "f4")
        rgb1 = rgb0 if rgb_ is None else np.broadcast_to(np.asarray(rgb_, "f4"), rgb0.shape)
        op1 = op0 if op is None else np.broadcast_to(np.asarray(op, "f4"), op0.shape)
        size1 = size0 if size is None else np.broadcast_to(np.asarray(size, "f4"), size0.shape)
        if delay is None or lag <= 0.0:
            d = np.zeros(len(vox), "f4")
        else:
            d = np.asarray(delay, "f4")
            span = float(d.max() - d.min())
            d = (d - d.min()) / span * lag if span > 0 else np.zeros_like(d)
        width = max(1e-6, 1.0 - float(d.max()))
        glow_rgb = None if glow is None else rgb(glow).astype("f4")
        moving = np.flatnonzero((np.abs(rgb1 - rgb0).sum(1) > 1e-4) | (np.abs(op1 - op0) > 1e-4)
                                | (np.abs(size1 - size0) > 1e-4))

        def fn(t: float) -> None:
            f = smoothstep(np.clip((t - d[moving]) / width, 0.0, 1.0)).astype("f4")
            fc = f[:, None]
            cur = rgb0[moving] + (rgb1[moving] - rgb0[moving]) * fc
            if glow_rgb is not None:
                g = (4.0 * f * (1.0 - f) * glow_strength)[:, None]
                cur = cur + (glow_rgb - cur) * g
            r_, o_, s_ = rgb0.copy(), op0.copy(), size0.copy()
            r_[moving] = cur
            o_[moving] = op0[moving] + (op1[moving] - op0[moving]) * f
            s_[moving] = size0[moving] + (size1[moving] - size0[moving]) * f
            vox.set(r_, o_, s_)
            if hook is not None:
                hook(t)

        self.animate_gl(fn, run_time=run_time, extra=extra)
        vox.set(np.array(rgb1), np.array(op1), np.array(size1))

    # -- text -----------------------------------------------------------------------------------
    def fixed(self, *mobs: Mobject, keep: bool = False, drop: tuple = ()) -> None:
        """Register on-screen text without showing it. With :func:`no_text` it is made fully
        transparent (fading it in shows nothing, the timing stays), unless *keep*: a plot, a colour
        key or a chart the picture needs is kept, all but its *drop* parts (a title)."""
        if no_text():
            for m in (drop if keep else mobs):
                m.set_opacity(0.0)

    def swap(self, old: Mobject | None, new: Mobject | None, run_time: float = 0.85) -> None:
        if old is not None:
            self.play(FadeOut(old, shift=DOWN * 0.12), run_time=run_time * 0.45)
        if new is not None:
            self.play(FadeIn(new, shift=UP * 0.12), run_time=run_time * 0.55)

    def caption(self, step: str, lines: list[tuple[str, str, float]], corner=DL) -> VGroup:
        head = Text(step, font=FONT, weight=BOLD, color=INK, font_size=25)
        body = VGroup(*[Text(t, font=FONT, color=c, font_size=s) for t, c, s in lines]).arrange(
            DOWN, buff=0.15, aligned_edge=LEFT)
        block = VGroup(head, body).arrange(DOWN, buff=0.24, aligned_edge=LEFT) if lines else VGroup(head)
        block.to_corner(corner, buff=0.5)
        block.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.16)
        self.fixed(block)
        return block

    def title_card(self, title: str, sub: str, who: str) -> VGroup:
        t = Text(title, font=FONT, weight=BOLD, color=INK, font_size=36)
        s = Text(sub, font=FONT, color=INK_SOFT, font_size=19)
        w = Text(who, font=FONT, color=INK_SOFT, font_size=18)
        card = VGroup(t, s, w).arrange(DOWN, buff=0.18, aligned_edge=LEFT).to_corner(DL, buff=0.55)
        card.add_background_rectangle(color="#FFFFFF", opacity=0.86, buff=0.16)
        self.fixed(card)
        return card

    # -- 3D-anchored labels ---------------------------------------------------------------------
    def to_frame(self, world: np.ndarray) -> np.ndarray:
        """World mm (n, 3) -> manim frame coordinates (n, 3)."""
        ndc = self.gl.project(world)
        return np.c_[ndc[:, 0] * config.frame_width / 2.0, ndc[:, 1] * config.frame_height / 2.0,
                     np.zeros(len(ndc))]

    def label_3d(self, text: str, world: np.ndarray, color: str = INK, font_size: float = 16,
                 offset=(0.0, 0.18), backing: bool = True) -> VGroup:
        """A text label that follows the world point *world* every frame (add/FadeIn it)."""
        lab = Text(text, font=FONT, color=color, font_size=font_size, weight=BOLD)
        g = VGroup(lab)
        if backing:
            g.add_background_rectangle(color="#FFFFFF", opacity=0.75, buff=0.05)
        w = np.asarray(world, float).reshape(1, 3)
        off = np.array([offset[0], offset[1], 0.0])

        def follow(m: Mobject) -> None:
            m.move_to(self.to_frame(w)[0] + off)

        follow(g)
        g.add_updater(follow)
        self.fixed(g)
        return g


def make_axes(xmax: float, ymax: float, xstep: float = 20, ystep: float = 1, x_length: float = 5.3,
              y_length: float = 2.3, ymin: float = 0.0) -> Axes:
    """Axes with Text tick labels (no LaTeX needed); with *ymin* < 0 the x ticks go under the bottom edge."""
    ax = Axes(x_range=[0, xmax, xstep], y_range=[ymin, ymax, ystep], x_length=x_length, y_length=y_length,
              tips=False, axis_config={"color": INK_SOFT, "stroke_width": 1.5})
    nums = VGroup()
    for x in np.arange(0, xmax + 1e-9, xstep):
        nums.add(Text(f"{x:g}", font=FONT, color=INK_SOFT, font_size=12).next_to(ax.c2p(x, ymin), DOWN, buff=0.08))
    for y in np.arange(ymin if ymin < 0 else ystep, ymax + 1e-9, ystep):
        nums.add(Text(f"{y:g}", font=FONT, color=INK_SOFT, font_size=12).next_to(ax.c2p(0, y), LEFT, buff=0.08))
    ax.add(nums)
    return ax


__all__ = ["GLScene", "ease", "make_axes"]
