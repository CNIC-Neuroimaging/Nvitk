"""GPU renderer for the dense nvitk animations (numpy + moderngl only, headless EGL).

The Cairo voxel clouds of :mod:`anim_voxels` re-rasterise every cube in Python each frame
(~1.1 s per 1080p frame for 1,600 cubes). The scenes built on this module draw their 3D content on
the GPU instead, at the data's own resolution (10⁴–10⁵ voxels, meshes, tubes, glyphs, a volume
MIP) in a few milliseconds per frame, and manim only composites its text over the result
(:mod:`gl_scene`).

Everything is in world millimetres (scanner RAS: +z superior). A :class:`GLRenderer` owns an MSAA
framebuffer, an orbit camera and a list of layers; :meth:`GLRenderer.render` returns an RGBA
``uint8`` frame, top row first. Opaque geometry is drawn first, then every translucent layer
back-to-front (instances and triangles sorted by depth each frame).

Layers
------
:class:`Voxels`   instanced cubes: per voxel centre, colour, opacity and size (0-1 of the pitch)
:class:`Mesh`     a lit triangle mesh (vertex colours), e.g. a vessel surface
:class:`Tubes`    polylines as lit tubes, revealed along their arc length (``progress``)
:class:`Spheres`  instanced spheres (particles, inscribed spheres, markers)
:class:`Arrows`   instanced arrows (velocity vectors)
:class:`Lines`    1-px line segments (wireframes)
:class:`Points`   round point sprites (mesh vertices, point clouds)
:class:`Quad`     a textured rectangle in 3D (a slice, a cross-section map, a plane)
:class:`VolumeMIP` a ray-marched maximum-intensity projection of a 3D volume
"""

from __future__ import annotations

import math

import numpy as np

try:  # the scenes import this module only in the manim environment, which has moderngl
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None

BG = (1.0, 1.0, 1.0)


# ----------------------------------------------------------------------------------------------
# small linear algebra
# ----------------------------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(np.asarray(target, float) - np.asarray(eye, float))
    s = _normalize(np.cross(f, up))
    if np.linalg.norm(s) < 1e-9:
        s = _normalize(np.cross(f, np.array([0.0, 1.0, 0.0])))
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ np.asarray(eye, float)
    return m


def perspective(fov_deg: float, aspect: float, near: float, far: float, shift=(0.0, 0.0)) -> np.ndarray:
    """Perspective projection with an NDC lens shift (*shift* moves the image, not the view)."""
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    m[0, 2] = -shift[0]
    m[1, 2] = -shift[1]
    return m


def _mat(m: np.ndarray) -> bytes:
    return np.asarray(m, dtype="f4").T.tobytes()  # GLSL is column-major


class OrbitCamera:
    """Orbit camera about *target*: *phi* = angle from +z (deg), *theta* = azimuth from +x (deg),
    *distance* in mm, vertical field of view *fov* (deg). *shift* moves the image on screen in
    NDC units (positive y = up), so content can sit above the captions while the camera turns."""

    def __init__(self, target=(0.0, 0.0, 0.0), distance: float = 400.0, phi: float = 70.0, theta: float = 90.0,
                 fov: float = 24.0, shift=(0.0, 0.0)) -> None:
        self.target = np.asarray(target, float)
        self.distance = float(distance)
        self.phi = float(phi)
        self.theta = float(theta)
        self.fov = float(fov)
        self.shift = np.asarray(shift, float)

    def params(self) -> dict:
        return {"target": self.target.copy(), "distance": self.distance, "phi": self.phi, "theta": self.theta,
                "fov": self.fov, "shift": self.shift.copy()}

    def set(self, **p) -> None:
        for k, v in p.items():
            setattr(self, k, np.asarray(v, float) if k in ("target", "shift") else float(v))

    @property
    def eye(self) -> np.ndarray:
        ph, th = math.radians(self.phi), math.radians(self.theta)
        d = np.array([math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)])
        return self.target + self.distance * d

    def view(self) -> np.ndarray:
        return look_at(self.eye, self.target, np.array([0.0, 0.0, 1.0]))

    def proj(self, aspect: float) -> np.ndarray:
        near = max(self.distance * 0.05, 1.0)
        return perspective(self.fov, aspect, near, self.distance * 6.0 + 2000.0, shift=tuple(self.shift))

    def forward(self) -> np.ndarray:
        return _normalize(self.target - self.eye)


# ----------------------------------------------------------------------------------------------
# shaders
# ----------------------------------------------------------------------------------------------

_LIGHT = """
uniform vec3 light_dir;   // view space, towards the light
vec3 shade(vec3 base, vec3 n, float amb, float spec) {
    float d = max(dot(n, light_dir), 0.0);
    float back = max(dot(n, -light_dir), 0.0) * 0.12;
    vec3 h = normalize(light_dir + vec3(0.0, 0.0, 1.0));
    float s = pow(max(dot(n, h), 0.0), 36.0) * spec;
    return base * (amb + (1.0 - amb) * d + back) + vec3(s);
}
"""

_CUBE_VS = """
#version 330
uniform mat4 view; uniform mat4 proj; uniform float side; uniform int pass_mode;
in vec3 in_pos; in vec3 in_norm;
in vec3 i_center; in vec3 i_rgb; in float i_op; in float i_size;
out vec3 v_n; out vec3 v_local; out vec3 v_ln; out vec3 v_rgb; out float v_op;
void main() {
    bool skip = i_op < 0.004 || (pass_mode == 0 && i_op < 0.995) || (pass_mode == 1 && i_op >= 0.995);
    vec4 vp = view * vec4(i_center + in_pos * side * i_size, 1.0);
    gl_Position = skip ? vec4(2.0, 2.0, 2.0, 1.0) : proj * vp;
    v_n = mat3(view) * in_norm; v_local = in_pos; v_ln = in_norm; v_rgb = i_rgb; v_op = i_op;
}
"""

_CUBE_FS = """
#version 330
uniform float edge_dark;
in vec3 v_n; in vec3 v_local; in vec3 v_ln; in vec3 v_rgb; in float v_op;
out vec4 frag;
""" + _LIGHT + """
void main() {
    vec3 c = shade(v_rgb, normalize(v_n), 0.56, 0.10);
    vec3 m = abs(v_local) * 2.0 * (1.0 - abs(v_ln));
    float e = max(max(m.x, m.y), m.z);
    c *= 1.0 - edge_dark * smoothstep(0.80, 0.99, e);
    frag = vec4(c, v_op);
}
"""

_MESH_VS = """
#version 330
uniform mat4 view; uniform mat4 proj; uniform mat4 model;
in vec3 in_pos; in vec3 in_norm; in vec4 in_rgba; in float in_s;
out vec3 v_n; out vec4 v_rgba; out float v_s; out vec3 v_vp;
void main() {
    vec4 vp = view * model * vec4(in_pos, 1.0);
    gl_Position = proj * vp;
    v_n = mat3(view) * mat3(model) * in_norm; v_rgba = in_rgba; v_s = in_s; v_vp = vp.xyz;
}
"""

_MESH_FS = """
#version 330
uniform float opacity; uniform float progress; uniform float amb; uniform float spec; uniform float rim;
in vec3 v_n; in vec4 v_rgba; in float v_s; in vec3 v_vp;
out vec4 frag;
""" + _LIGHT + """
void main() {
    if (v_s > progress) discard;
    vec3 n = normalize(v_n);
    if (!gl_FrontFacing) n = -n;
    vec3 c = shade(v_rgba.rgb, n, amb, spec);
    float a = v_rgba.a * opacity;
    if (rim > 0.0) {  // see-through shells: edges denser than faces
        float r = 1.0 - abs(dot(n, normalize(-v_vp)));
        a *= mix(1.0, 0.35 + 1.6 * r * r, rim);
    }
    frag = vec4(c, clamp(a, 0.0, 1.0));
}
"""

_SPHERE_VS = """
#version 330
uniform mat4 view; uniform mat4 proj; uniform int pass_mode;
in vec3 in_pos;
in vec3 i_center; in float i_radius; in vec3 i_rgb; in float i_op;
out vec3 v_n; out vec3 v_rgb; out float v_op;
void main() {
    bool skip = i_op < 0.004 || i_radius <= 0.0 || (pass_mode == 0 && i_op < 0.995) || (pass_mode == 1 && i_op >= 0.995);
    vec4 vp = view * vec4(i_center + in_pos * i_radius, 1.0);
    gl_Position = skip ? vec4(2.0, 2.0, 2.0, 1.0) : proj * vp;
    v_n = mat3(view) * in_pos; v_rgb = i_rgb; v_op = i_op;
}
"""

_SPHERE_FS = """
#version 330
in vec3 v_n; in vec3 v_rgb; in float v_op;
out vec4 frag;
""" + _LIGHT + """
void main() { frag = vec4(shade(v_rgb, normalize(v_n), 0.50, 0.25), v_op); }
"""

_ARROW_VS = """
#version 330
uniform mat4 view; uniform mat4 proj; uniform int pass_mode;
in vec3 in_pos; in vec3 in_norm;
in vec3 i_origin; in vec3 i_vec; in float i_width; in vec3 i_rgb; in float i_op;
out vec3 v_n; out vec3 v_rgb; out float v_op;
void main() {
    float len = length(i_vec);
    bool skip = len < 1e-6 || i_op < 0.004 || (pass_mode == 0 && i_op < 0.995) || (pass_mode == 1 && i_op >= 0.995);
    vec3 w = i_vec / max(len, 1e-6);
    vec3 a = abs(w.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 u = normalize(cross(a, w)); vec3 v = cross(w, u);
    // the head keeps its proportions; the shaft stretches with the length
    float head = min(len * 0.45, i_width * 3.2);
    float z = in_pos.z < 0.5 ? in_pos.z / 0.5 * (len - head) : (len - head) + (in_pos.z - 0.5) / 0.5 * head;
    vec3 p = i_origin + w * z + (u * in_pos.x + v * in_pos.y) * i_width;
    vec3 n = normalize(u * in_norm.x + v * in_norm.y + w * in_norm.z);
    vec4 vp = view * vec4(p, 1.0);
    gl_Position = skip ? vec4(2.0, 2.0, 2.0, 1.0) : proj * vp;
    v_n = mat3(view) * n; v_rgb = i_rgb; v_op = i_op;
}
"""

_QUAD_VS = """
#version 330
uniform mat4 view; uniform mat4 proj;
in vec3 in_pos; in vec2 in_uv;
out vec2 v_uv;
void main() { gl_Position = proj * view * vec4(in_pos, 1.0); v_uv = in_uv; }
"""

_QUAD_FS = """
#version 330
uniform sampler2D tex; uniform float opacity;
in vec2 v_uv; out vec4 frag;
void main() { vec4 c = texture(tex, v_uv); frag = vec4(c.rgb, c.a * opacity); }
"""

_MIP_VS = """
#version 330
uniform mat4 view; uniform mat4 proj; uniform mat4 model;   // model: unit cube [0,1]^3 -> world
in vec3 in_pos;
out vec3 v_world;
void main() { vec4 w = model * vec4(in_pos, 1.0); v_world = w.xyz; gl_Position = proj * view * w; }
"""

_MIP_FS = """
#version 330
uniform sampler3D vol; uniform mat4 inv_model; uniform vec3 eye; uniform float opacity;
uniform float lo; uniform float hi; uniform float gamma; uniform vec3 color; uniform int steps;
uniform vec3 clip_lo; uniform vec3 clip_hi;
in vec3 v_world; out vec4 frag;
void main() {
    vec3 o = (inv_model * vec4(eye, 1.0)).xyz;
    vec3 e = (inv_model * vec4(v_world, 1.0)).xyz;
    vec3 d = e - o;
    vec3 inv = 1.0 / d;
    vec3 t0 = (clip_lo - o) * inv, t1 = (clip_hi - o) * inv;
    vec3 tmin = min(t0, t1), tmax = max(t0, t1);
    float ta = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
    float tb = min(min(tmax.x, tmax.y), tmax.z);
    if (tb <= ta) discard;
    float m = 0.0;
    for (int i = 0; i < steps; i++) {
        float t = ta + (tb - ta) * (float(i) + 0.5) / float(steps);
        m = max(m, texture(vol, o + d * t).r);
    }
    float a = pow(clamp((m - lo) / max(hi - lo, 1e-6), 0.0, 1.0), gamma) * opacity;
    if (a < 0.002) discard;
    frag = vec4(color, a);
}
"""

_LINE_VS = """
#version 330
uniform mat4 view; uniform mat4 proj;
in vec3 in_pos; in vec4 in_rgba; in vec3 in_norm;
out vec4 v_rgba; out float v_facing;
void main() {
    vec4 vp = view * vec4(in_pos, 1.0);
    gl_Position = proj * vp;
    v_rgba = in_rgba;
    v_facing = dot(mat3(view) * in_norm, normalize(-vp.xyz));
}
"""

_LINE_FS = """
#version 330
uniform float opacity; uniform int cull;
in vec4 v_rgba; in float v_facing; out vec4 frag;
void main() {
    if (cull == 1 && v_facing < -0.05) discard;
    frag = vec4(v_rgba.rgb, v_rgba.a * opacity);
}
"""

_POINT_VS = """
#version 330
uniform mat4 view; uniform mat4 proj; uniform float size_px; uniform float ref_dist;
in vec3 in_pos; in vec4 in_rgba; in vec3 in_norm;
out vec4 v_rgba; out float v_facing;
void main() {
    vec4 vp = view * vec4(in_pos, 1.0);
    gl_Position = proj * vp;
    gl_PointSize = clamp(size_px * ref_dist / max(-vp.z, 1e-3), 1.0, 64.0);
    v_rgba = in_rgba;
    v_facing = dot(mat3(view) * in_norm, normalize(-vp.xyz));
}
"""

_POINT_FS = """
#version 330
uniform float opacity; uniform int cull;
in vec4 v_rgba; in float v_facing; out vec4 frag;
void main() {
    if (cull == 1 && v_facing < -0.05) discard;
    vec2 d = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(d, d);
    if (r2 > 1.0) discard;
    float shade = 0.75 + 0.25 * sqrt(1.0 - r2);
    frag = vec4(v_rgba.rgb * shade, v_rgba.a * opacity * smoothstep(1.0, 0.8, r2));
}
"""


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------

def _cube_geometry() -> tuple[np.ndarray, np.ndarray]:
    """Unit cube centred at 0, 24 vertices (flat normals), 36 indices."""
    verts, idx = [], []
    for axis in range(3):
        for sgn in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sgn
            u = np.zeros(3)
            u[(axis + 1) % 3] = 1.0
            v = np.cross(n, u)
            base = len(verts)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = 0.5 * n + 0.5 * a * u + 0.5 * b * v
                verts.append(np.r_[p, n])
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.asarray(verts, "f4"), np.asarray(idx, "i4")


def icosphere(subdiv: int = 2) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + 5 ** 0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    verts = [np.asarray(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache: dict = {}
        nf = []

        def mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = verts[a] + verts[b]
                verts.append(p / np.linalg.norm(p))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.asarray(verts, "f4"), np.asarray(f, "i4").ravel()


def _arrow_geometry(k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Arrow along +z in a parametric frame: z in [0, 0.5] = shaft, [0.5, 1] = head (the vertex
    shader maps it to the real length); x/y in units of the arrow width."""
    verts, idx = [], []
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    r_shaft, r_head = 0.5, 1.45
    # shaft side
    base = len(verts)
    for z in (0.0, 0.5):
        for i in range(k):
            verts.append([r_shaft * c[i], r_shaft * s[i], z, c[i], s[i], 0.0])
    for i in range(k):
        j = (i + 1) % k
        idx += [base + i, base + j, base + k + j, base + i, base + k + j, base + k + i]
    # head underside disk
    base = len(verts)
    verts.append([0, 0, 0.5, 0, 0, -1])
    for i in range(k):
        verts.append([r_head * c[i], r_head * s[i], 0.5, 0, 0, -1])
    for i in range(k):
        idx += [base, base + 1 + (i + 1) % k, base + 1 + i]
    # cone
    base = len(verts)
    slope = r_head / 0.5
    for i in range(k):
        n = _normalize(np.array([c[i], s[i], slope * 0.35]))
        verts.append([r_head * c[i], r_head * s[i], 0.5, *n])
        verts.append([0.0, 0.0, 1.0, *n])
    for i in range(k):
        j = (i + 1) % k
        idx += [base + 2 * i, base + 2 * j, base + 2 * i + 1]
    return np.asarray(verts, "f4"), np.asarray(idx, "i4")


def tube_mesh(polyline: np.ndarray, radius, sides: int = 10, colors: np.ndarray | None = None,
              caps: bool = True) -> dict[str, np.ndarray]:
    """A tube around *polyline* (n, 3): per-vertex position, normal, RGBA and normalised arc
    length ``s`` ∈ [0, 1] (for :class:`Tubes` ``progress``). *radius* scalar or per point;
    *colors* (n, 3|4) per point."""
    p = np.asarray(polyline, float)
    n = len(p)
    r = np.broadcast_to(np.asarray(radius, float), (n,))
    seg = np.diff(p, axis=0)
    L = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
    s = L / L[-1] if L[-1] > 0 else np.zeros(n)
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0], t[-1] = p[1] - p[0], p[-1] - p[-2]
    t = _normalize(t)
    # parallel transport
    a = np.array([0.0, 0.0, 1.0]) if abs(t[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _normalize(np.cross(t[0], a))
    us = np.zeros_like(p)
    for i in range(n):
        u = u - t[i] * np.dot(u, t[i])
        nu = np.linalg.norm(u)
        u = u / nu if nu > 1e-9 else _normalize(np.cross(t[i], a))
        us[i] = u
    vs = np.cross(t, us)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.cos(ang)[None, :, None] * us[:, None, :] + np.sin(ang)[None, :, None] * vs[:, None, :]
    pos = p[:, None, :] + ring * r[:, None, None]
    if colors is None:
        col = np.ones((n, 4))
    else:
        col = np.asarray(colors, float)
        if col.shape[1] == 3:
            col = np.c_[col, np.ones(n)]
    verts = np.concatenate([pos.reshape(-1, 3), ring.reshape(-1, 3), np.repeat(col, sides, 0),
                            np.repeat(s, sides)[:, None]], 1)
    i = np.arange(n - 1)[:, None]
    j = np.arange(sides)[None, :]
    a0 = i * sides + j
    a1 = i * sides + (j + 1) % sides
    b0, b1 = a0 + sides, a1 + sides
    idx = np.stack([a0, b0, b1, a0, b1, a1], -1).reshape(-1)
    if caps:
        extra_v, extra_i = [], []
        for end, sign in ((0, -1.0), (n - 1, 1.0)):
            base = len(verts) + len(extra_v)
            nrm = t[end] * sign
            extra_v.append(np.r_[p[end], nrm, col[end], s[end]])
            for k in range(sides):
                extra_v.append(np.r_[pos[end, k], nrm, col[end], s[end]])
            for k in range(sides):
                a_, b_ = base + 1 + k, base + 1 + (k + 1) % sides
                extra_i += [base, a_, b_] if sign > 0 else [base, b_, a_]
        verts = np.concatenate([verts, np.asarray(extra_v)], 0)
        idx = np.r_[idx, np.asarray(extra_i)]
    return {"v": verts.astype("f4"), "i": idx.astype("i4")}


def merge_meshes(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    vs, is_, off = [], [], 0
    for m in parts:
        vs.append(m["v"])
        is_.append(m["i"] + off)
        off += len(m["v"])
    return {"v": np.concatenate(vs), "i": np.concatenate(is_)}


# ----------------------------------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------------------------------

class Layer:
    visible = True

    def setup(self, r: "GLRenderer") -> None:  # pragma: no cover - interface
        pass

    def draw_opaque(self, r: "GLRenderer") -> None:
        pass

    def draw_translucent(self, r: "GLRenderer") -> None:
        pass

    def depth_key(self, r: "GLRenderer") -> float:
        """Sorting key between translucent layers (larger = farther = drawn first)."""
        return 0.0


class _Instanced(Layer):
    """Shared state handling of the instanced layers: numpy state arrays, uploaded when dirty,
    translucent instances sorted back-to-front."""

    FMT = ""
    ATTRS: tuple = ()
    CULL = False  # closed convex glyphs: cull back faces when translucent

    def _pack(self) -> np.ndarray:  # (N, k) float32
        raise NotImplementedError

    def _centers(self) -> np.ndarray:
        raise NotImplementedError

    def _setup_instanced(self, r: "GLRenderer", prog, geom_vbo, geom_fmt, geom_attrs, ebo) -> None:
        self.prog = prog
        n = max(1, len(self._centers()))
        self.ibuf = r.ctx.buffer(reserve=n * self.STRIDE * 4, dynamic=True)
        self.tbuf = r.ctx.buffer(reserve=n * self.STRIDE * 4, dynamic=True)
        self.vao = r.ctx.vertex_array(prog, [(geom_vbo, geom_fmt, *geom_attrs), (self.ibuf, self.FMT, *self.ATTRS)],
                                      index_buffer=ebo)
        self.tvao = r.ctx.vertex_array(prog, [(geom_vbo, geom_fmt, *geom_attrs), (self.tbuf, self.FMT, *self.ATTRS)],
                                       index_buffer=ebo)
        self.dirty = True

    def _upload(self) -> None:
        if self.dirty:
            self._packed = np.ascontiguousarray(self._pack(), dtype="f4")
            self.ibuf.write(self._packed.tobytes())
            self.dirty = False

    def _uniforms(self, r: "GLRenderer") -> None:
        r.set_common(self.prog)

    def draw_opaque(self, r: "GLRenderer") -> None:
        if not self.visible or len(self._centers()) == 0:
            return
        self._upload()
        if not (self._op() >= 0.995).any():
            return
        self._uniforms(r)
        self.prog["pass_mode"].value = 0
        self.vao.render(moderngl.TRIANGLES, instances=len(self._packed))

    def draw_translucent(self, r: "GLRenderer") -> None:
        if not self.visible or len(self._centers()) == 0:
            return
        self._upload()
        op = self._op()
        sel = np.flatnonzero((op > 0.004) & (op < 0.995))
        if len(sel) == 0:
            return
        depth = (self._centers()[sel] - r.camera.eye) @ r.camera.forward()
        order = sel[np.argsort(-depth)]
        self.tbuf.write(self._packed[order].tobytes())
        self._uniforms(r)
        self.prog["pass_mode"].value = 1
        if self.CULL:
            r.ctx.enable(moderngl.CULL_FACE)
        self.tvao.render(moderngl.TRIANGLES, instances=len(order))
        r.ctx.disable(moderngl.CULL_FACE)

    def depth_key(self, r: "GLRenderer") -> float:
        c = self._centers()
        if len(c) == 0:
            return 0.0
        return float((c.mean(0) - r.camera.eye) @ r.camera.forward())


class Voxels(_Instanced):
    """Instanced cubes. *centers* (N, 3) mm; *pitch* the voxel side (mm) at size 1; the drawn side
    is ``pitch * fill`` (hairline gaps). State arrays ``rgb`` (N, 3), ``op`` (N,), ``size`` (N,);
    call :meth:`set` to change them."""

    FMT = "3f 3f 1f 1f/i"
    ATTRS = ("i_center", "i_rgb", "i_op", "i_size")
    STRIDE = 8
    CULL = True

    def __init__(self, centers: np.ndarray, pitch: float, fill: float = 0.9, color=(0.58, 0.64, 0.74),
                 edge_dark: float = 0.22) -> None:
        self.centers = np.asarray(centers, "f4").reshape(-1, 3)
        n = len(self.centers)
        self.pitch, self.fill, self.edge_dark = float(pitch), float(fill), float(edge_dark)
        self.rgb = np.tile(np.asarray(color, "f4"), (n, 1))
        self.op = np.zeros(n, "f4")
        self.size = np.ones(n, "f4")

    def __len__(self) -> int:
        return len(self.centers)

    def set(self, rgb=None, op=None, size=None) -> None:
        if rgb is not None:
            self.rgb = np.broadcast_to(np.asarray(rgb, "f4"), (len(self), 3))
        if op is not None:
            self.op = np.broadcast_to(np.asarray(op, "f4"), (len(self),))
        if size is not None:
            self.size = np.broadcast_to(np.asarray(size, "f4"), (len(self),))
        self.dirty = True

    def _centers(self):
        return self.centers

    def _op(self):
        return self.op

    def _pack(self):
        return np.c_[self.centers, self.rgb, self.op, self.size]

    def setup(self, r: "GLRenderer") -> None:
        prog = r.program("cube", _CUBE_VS, _CUBE_FS)
        self._setup_instanced(r, prog, r.cube_vbo, "3f 3f", ("in_pos", "in_norm"), r.cube_ebo)

    def _uniforms(self, r):
        r.set_common(self.prog)
        self.prog["side"].value = self.pitch * self.fill
        self.prog["edge_dark"].value = self.edge_dark


class Spheres(_Instanced):
    FMT = "3f 1f 3f 1f/i"
    ATTRS = ("i_center", "i_radius", "i_rgb", "i_op")
    STRIDE = 8
    CULL = True

    def __init__(self, centers: np.ndarray, radius=1.0, color=(0.2, 0.4, 0.8), subdiv: int = 2) -> None:
        self.centers = np.asarray(centers, "f4").reshape(-1, 3)
        n = len(self.centers)
        self.radius = np.broadcast_to(np.asarray(radius, "f4"), (n,)).copy()
        self.rgb = np.tile(np.asarray(color, "f4"), (n, 1))
        self.op = np.ones(n, "f4")
        self.subdiv = subdiv

    def __len__(self) -> int:
        return len(self.centers)

    def set(self, centers=None, radius=None, rgb=None, op=None) -> None:
        if centers is not None:
            self.centers = np.asarray(centers, "f4").reshape(-1, 3)
        n = len(self.centers)
        if radius is not None:
            self.radius = np.broadcast_to(np.asarray(radius, "f4"), (n,))
        if rgb is not None:
            self.rgb = np.broadcast_to(np.asarray(rgb, "f4"), (n, 3))
        if op is not None:
            self.op = np.broadcast_to(np.asarray(op, "f4"), (n,))
        self.dirty = True

    def _centers(self):
        return self.centers

    def _op(self):
        return self.op

    def _pack(self):
        return np.c_[self.centers, self.radius, self.rgb, self.op]

    def setup(self, r: "GLRenderer") -> None:
        prog = r.program("sphere", _SPHERE_VS, _SPHERE_FS)
        vbo, ebo = r.sphere_geometry(self.subdiv)
        self._setup_instanced(r, prog, vbo, "3f", ("in_pos",), ebo)

    def _upload(self) -> None:  # the instance count may change (particles)
        if self.dirty:
            need = max(1, len(self.centers)) * self.STRIDE * 4
            if need > self.ibuf.size:
                self.ibuf.orphan(need)
                self.tbuf.orphan(need)
        super()._upload()


class Arrows(_Instanced):
    """Instanced arrows: *origins* (N, 3), *vectors* (N, 3) in mm (the arrow's full length),
    *width* the shaft diameter (mm)."""

    FMT = "3f 3f 1f 3f 1f/i"
    ATTRS = ("i_origin", "i_vec", "i_width", "i_rgb", "i_op")
    STRIDE = 11

    def __init__(self, origins: np.ndarray, vectors: np.ndarray, width=0.3, color=(0.2, 0.4, 0.8)) -> None:
        self.origins = np.asarray(origins, "f4").reshape(-1, 3)
        n = len(self.origins)
        self.vectors = np.asarray(vectors, "f4").reshape(-1, 3)
        self.width = np.broadcast_to(np.asarray(width, "f4"), (n,)).copy()
        self.rgb = np.tile(np.asarray(color, "f4"), (n, 1))
        self.op = np.ones(n, "f4")

    def set(self, vectors=None, width=None, rgb=None, op=None) -> None:
        n = len(self.origins)
        if vectors is not None:
            self.vectors = np.asarray(vectors, "f4").reshape(-1, 3)
        if width is not None:
            self.width = np.broadcast_to(np.asarray(width, "f4"), (n,))
        if rgb is not None:
            self.rgb = np.broadcast_to(np.asarray(rgb, "f4"), (n, 3))
        if op is not None:
            self.op = np.broadcast_to(np.asarray(op, "f4"), (n,))
        self.dirty = True

    def _centers(self):
        return self.origins + 0.5 * self.vectors

    def _op(self):
        return self.op

    def _pack(self):
        return np.c_[self.origins, self.vectors, self.width, self.rgb, self.op]

    def setup(self, r: "GLRenderer") -> None:
        prog = r.program("arrow", _ARROW_VS, _SPHERE_FS)
        v, i = _arrow_geometry()
        vbo, ebo = r.ctx.buffer(v.tobytes()), r.ctx.buffer(i.tobytes())
        self._setup_instanced(r, prog, vbo, "3f 3f", ("in_pos", "in_norm"), ebo)


class Mesh(Layer):
    """A lit triangle mesh: ``{"v": (n, 11) [pos 3, normal 3, rgba 4, s 1], "i": (m,)}`` (see
    :func:`tube_mesh`), or build with :meth:`from_triangles`. ``opacity`` < 1 draws it translucent
    with its triangles sorted back-to-front; ``rim`` > 0 makes a translucent shell denser at its
    silhouette. ``model`` is an optional 4 × 4 world transform."""

    def __init__(self, mesh: dict[str, np.ndarray], opacity: float = 1.0, amb: float = 0.5, spec: float = 0.18,
                 rim: float = 0.0, two_sided: bool = True) -> None:
        self.v = np.ascontiguousarray(mesh["v"], "f4")
        self.i = np.ascontiguousarray(mesh["i"], "i4").reshape(-1, 3)
        self.opacity, self.amb, self.spec, self.rim = float(opacity), float(amb), float(spec), float(rim)
        self.progress = 1.0
        self.model = np.eye(4)
        self.two_sided = two_sided
        self._cent = self.v[self.i, :3].mean(1) if len(self.i) else np.zeros((0, 3))
        self.dirty_v = False

    @staticmethod
    def from_triangles(verts: np.ndarray, faces: np.ndarray, color=(0.8, 0.3, 0.3), normals=None,
                       colors: np.ndarray | None = None) -> dict[str, np.ndarray]:
        verts = np.asarray(verts, float)
        faces = np.asarray(faces, int)
        if normals is None:
            fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
            normals = np.zeros_like(verts)
            for k in range(3):
                np.add.at(normals, faces[:, k], fn)
            normals = _normalize(normals)
        if colors is None:
            colors = np.tile(np.r_[np.asarray(color, float), 1.0], (len(verts), 1))
        elif colors.shape[1] == 3:
            colors = np.c_[colors, np.ones(len(colors))]
        s = np.zeros((len(verts), 1))
        return {"v": np.c_[verts, normals, colors, s].astype("f4"), "i": faces.ravel().astype("i4")}

    def set_colors(self, rgba: np.ndarray) -> None:
        rgba = np.asarray(rgba, "f4")
        if rgba.shape[1] == 3:
            rgba = np.c_[rgba, self.v[:, 9]]
        self.v[:, 6:10] = rgba
        self.dirty_v = True

    def setup(self, r: "GLRenderer") -> None:
        self.prog = r.program("mesh", _MESH_VS, _MESH_FS)
        self.vbo = r.ctx.buffer(self.v.tobytes(), dynamic=True)
        self.ebo = r.ctx.buffer(self.i.tobytes(), dynamic=True)
        self.vao = r.ctx.vertex_array(self.prog, [(self.vbo, "3f 3f 4f 1f", "in_pos", "in_norm", "in_rgba", "in_s")],
                                      index_buffer=self.ebo)

    def _draw(self, r: "GLRenderer", opacity: float) -> None:
        if self.dirty_v:
            self.vbo.write(self.v.tobytes())
            self.dirty_v = False
        r.set_common(self.prog)
        self.prog["model"].write(_mat(self.model))
        self.prog["opacity"].value = opacity
        self.prog["progress"].value = self.progress + 1e-6
        self.prog["amb"].value = self.amb
        self.prog["spec"].value = self.spec
        self.prog["rim"].value = self.rim
        self.vao.render(moderngl.TRIANGLES)

    def _solid(self) -> bool:
        return self.opacity >= 0.995 and self.rim == 0.0 and float(self.v[:, 9].min(initial=1.0)) >= 0.995

    def draw_opaque(self, r: "GLRenderer") -> None:
        if not self.visible or len(self.i) == 0 or self.opacity < 0.004:
            return
        if self._solid():
            if not self.two_sided:
                r.ctx.enable(moderngl.CULL_FACE)
            self._draw(r, 1.0)
            r.ctx.disable(moderngl.CULL_FACE)

    def draw_translucent(self, r: "GLRenderer") -> None:
        if not self.visible or len(self.i) == 0 or self.opacity < 0.004 or self._solid():
            return
        cent = (self.model[:3, :3] @ self._cent.T).T + self.model[:3, 3]
        depth = (cent - r.camera.eye) @ r.camera.forward()
        self.ebo.write(self.i[np.argsort(-depth)].tobytes())
        self._draw(r, self.opacity)

    def depth_key(self, r):
        if len(self._cent) == 0:
            return 0.0
        c = self.model[:3, :3] @ self._cent.mean(0) + self.model[:3, 3]
        return float((c - r.camera.eye) @ r.camera.forward())


class Tubes(Mesh):
    """Polylines as tubes (one mesh). ``progress`` ∈ [0, 1] reveals each tube along its own arc
    length. *colors*: one RGB per polyline, or per point arrays."""

    def __init__(self, polylines: list[np.ndarray], radius=0.5, colors=None, sides: int = 10, opacity: float = 1.0,
                 amb: float = 0.45, spec: float = 0.3) -> None:
        parts = []
        for k, pl in enumerate(polylines):
            pl = np.asarray(pl, float)
            if len(pl) < 2:
                continue
            rad = radius[k] if isinstance(radius, (list, tuple)) else radius
            col = None
            if colors is not None:
                c = np.asarray(colors[k] if isinstance(colors, (list, tuple)) or np.ndim(colors) > 1 else colors,
                               float)
                col = np.tile(c, (len(pl), 1)) if c.ndim == 1 else c
            parts.append(tube_mesh(pl, rad, sides=sides, colors=col))
        mesh = merge_meshes(parts) if parts else {"v": np.zeros((0, 11), "f4"), "i": np.zeros(0, "i4")}
        super().__init__(mesh, opacity=opacity, amb=amb, spec=spec)


class Lines(Layer):
    """Line segments (1 px, antialiased by the MSAA buffer): *starts*, *ends* (n, 3); *colors*
    (n, 3|4) or one RGB. ``progress`` ∈ [0, 1] shows the first part of the segments (in their
    order: sort them for a sweep). Translucent, unsorted (thin lines blend fine)."""

    def __init__(self, starts: np.ndarray, ends: np.ndarray, colors=(0.2, 0.2, 0.2), opacity: float = 1.0,
                 normals: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        a, b = np.asarray(starts, "f4").reshape(-1, 3), np.asarray(ends, "f4").reshape(-1, 3)
        n = len(a)
        c = np.asarray(colors, "f4")
        c = np.tile(c, (n, 1)) if c.ndim == 1 else c
        if c.shape[1] == 3:
            c = np.c_[c, np.ones(n, "f4")]
        self.n = n
        self.cull = normals is not None  # with surface normals at both ends: hide the back of a surface
        na, nb = normals if normals is not None else (np.zeros((n, 3), "f4"), np.zeros((n, 3), "f4"))
        self.v = np.ascontiguousarray(np.stack([np.c_[a, c, na], np.c_[b, c, nb]], 1).reshape(-1, 10), "f4")
        self.opacity = float(opacity)
        self.progress = 1.0

    def setup(self, r: "GLRenderer") -> None:
        self.prog = r.program("lines", _LINE_VS, _LINE_FS)
        self.vbo = r.ctx.buffer(self.v.tobytes())
        self.vao = r.ctx.vertex_array(self.prog, [(self.vbo, "3f 4f 3f", "in_pos", "in_rgba", "in_norm")])

    def draw_translucent(self, r: "GLRenderer") -> None:
        k = int(round(self.n * float(np.clip(self.progress, 0, 1))))
        if not self.visible or self.opacity < 0.004 or k == 0:
            return
        r.set_common(self.prog)
        self.prog["opacity"].value = self.opacity
        self.prog["cull"].value = int(self.cull)
        self.vao.render(moderngl.LINES, vertices=2 * k)

    def depth_key(self, r):
        return float((self.v[:, :3].mean(0) - r.camera.eye) @ r.camera.forward()) if self.n else 0.0


class Points(Layer):
    """Round point sprites at *positions* (n, 3): ``size_px`` on screen at the camera distance it
    was created for (``ref_dist``), scaled with depth. ``progress`` shows the first part."""

    def __init__(self, positions: np.ndarray, colors=(0.2, 0.2, 0.2), size_px: float = 3.0, opacity: float = 1.0,
                 ref_dist: float = 300.0, normals: np.ndarray | None = None) -> None:
        p = np.asarray(positions, "f4").reshape(-1, 3)
        c = np.asarray(colors, "f4")
        c = np.tile(c, (len(p), 1)) if c.ndim == 1 else c
        if c.shape[1] == 3:
            c = np.c_[c, np.ones(len(p), "f4")]
        self.n = len(p)
        self.cull = normals is not None
        nrm = np.zeros((len(p), 3), "f4") if normals is None else np.asarray(normals, "f4")
        self.v = np.ascontiguousarray(np.c_[p, c, nrm], "f4")
        self.size_px, self.opacity, self.ref_dist = float(size_px), float(opacity), float(ref_dist)
        self.progress = 1.0

    def setup(self, r: "GLRenderer") -> None:
        self.prog = r.program("points", _POINT_VS, _POINT_FS)
        self.vbo = r.ctx.buffer(self.v.tobytes())
        self.vao = r.ctx.vertex_array(self.prog, [(self.vbo, "3f 4f 3f", "in_pos", "in_rgba", "in_norm")])

    def draw_translucent(self, r: "GLRenderer") -> None:
        k = int(round(self.n * float(np.clip(self.progress, 0, 1))))
        if not self.visible or self.opacity < 0.004 or k == 0:
            return
        r.set_common(self.prog)
        self.prog["opacity"].value = self.opacity
        self.prog["size_px"].value = self.size_px * r.height / 1080.0
        self.prog["ref_dist"].value = self.ref_dist
        self.prog["cull"].value = int(self.cull)
        r.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.vao.render(moderngl.POINTS, vertices=k)

    def depth_key(self, r):
        return float((self.v[:, :3].mean(0) - r.camera.eye) @ r.camera.forward()) if self.n else 0.0


class Quad(Layer):
    """A textured rectangle: *corners* (4, 3) world mm in order (0,0) (1,0) (1,1) (0,1) of the
    texture's (u, v); *rgba* (h, w, 4) uint8, row 0 = v 0."""

    def __init__(self, corners: np.ndarray, rgba: np.ndarray, opacity: float = 1.0) -> None:
        self.corners = np.asarray(corners, "f4")
        self.rgba = np.ascontiguousarray(rgba, np.uint8)
        self.opacity = float(opacity)
        self._new_tex = None

    def set_image(self, rgba: np.ndarray) -> None:
        self._new_tex = np.ascontiguousarray(rgba, np.uint8)

    def set_corners(self, corners: np.ndarray) -> None:
        self.corners = np.asarray(corners, "f4")
        self.vbo.write(self._verts())

    def _verts(self) -> bytes:
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], "f4")
        v = np.c_[self.corners, uv][[0, 1, 2, 0, 2, 3]]
        return np.ascontiguousarray(v, "f4").tobytes()

    def setup(self, r: "GLRenderer") -> None:
        self.ctx = r.ctx
        self.prog = r.program("quad", _QUAD_VS, _QUAD_FS)
        self.vbo = r.ctx.buffer(self._verts(), dynamic=True)
        self.vao = r.ctx.vertex_array(self.prog, [(self.vbo, "3f 2f", "in_pos", "in_uv")])
        self._make_tex(self.rgba)

    def _make_tex(self, rgba: np.ndarray) -> None:
        h, w = rgba.shape[:2]
        if getattr(self, "tex", None) is not None and self.tex.size == (w, h):
            self.tex.write(rgba.tobytes())
        else:
            self.tex = self.ctx.texture((w, h), 4, rgba.tobytes())
            self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.tex.repeat_x = self.tex.repeat_y = False

    def draw_translucent(self, r: "GLRenderer") -> None:
        if not self.visible or self.opacity < 0.004:
            return
        if self._new_tex is not None:
            self._make_tex(self._new_tex)
            self._new_tex = None
        r.set_common(self.prog)
        self.tex.use(0)
        self.prog["tex"].value = 0
        self.prog["opacity"].value = self.opacity
        self.vao.render(moderngl.TRIANGLES)

    def depth_key(self, r):
        return float((self.corners.mean(0) - r.camera.eye) @ r.camera.forward())


class VolumeMIP(Layer):
    """Ray-marched maximum-intensity projection of *volume* (3D, values in [0, 1] or uint8) placed
    by *affine* (voxel index -> world mm). Drawn as ``color`` with alpha
    ``opacity · ((max − lo)/(hi − lo))^gamma``: dark vessels on the white page, like an
    angiogram film. ``clip_lo``/``clip_hi`` crop it in normalised texture units."""

    def __init__(self, volume: np.ndarray, affine: np.ndarray, color=(0.11, 0.12, 0.14), opacity: float = 0.9,
                 lo: float = 0.15, hi: float = 0.85, gamma: float = 1.4, steps: int = 384) -> None:
        v = np.asarray(volume)
        if v.dtype != np.uint8:
            v = np.clip(np.rint(np.asarray(v, float) * 255.0), 0, 255).astype(np.uint8)
        self.vol = v
        self.affine = np.asarray(affine, float)
        self.color, self.opacity = np.asarray(color, float), float(opacity)
        self.lo, self.hi, self.gamma, self.steps = float(lo), float(hi), float(gamma), int(steps)
        self.clip_lo = np.zeros(3)
        self.clip_hi = np.ones(3)
        nx, ny, nz = v.shape
        # unit cube -> voxel index space (texel centres at (i + 0.5)/n) -> world
        s = np.eye(4)
        s[:3, :3] = np.diag([nx, ny, nz])
        s[:3, 3] = -0.5
        self.model = self.affine @ s

    def setup(self, r: "GLRenderer") -> None:
        self.prog = r.program("mip", _MIP_VS, _MIP_FS)
        nx, ny, nz = self.vol.shape
        self.tex = r.ctx.texture3d((nx, ny, nz), 1, np.ascontiguousarray(self.vol.transpose(2, 1, 0)).tobytes())
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.repeat_x = self.tex.repeat_y = self.tex.repeat_z = False
        cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], "f4")
        faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                          [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], "i4")
        self.vbo = r.ctx.buffer(cube.tobytes())
        self.ebo = r.ctx.buffer(faces.tobytes())
        self.vao = r.ctx.vertex_array(self.prog, [(self.vbo, "3f", "in_pos")], index_buffer=self.ebo)

    def draw_translucent(self, r: "GLRenderer") -> None:
        if not self.visible or self.opacity < 0.004:
            return
        r.set_common(self.prog)
        self.prog["model"].write(_mat(self.model))
        self.prog["inv_model"].write(_mat(np.linalg.inv(self.model)))
        self.prog["eye"].value = tuple(r.camera.eye)
        for k in ("opacity", "lo", "hi", "gamma"):
            self.prog[k].value = getattr(self, k)
        self.prog["color"].value = tuple(self.color)
        self.prog["steps"].value = self.steps
        self.prog["clip_lo"].value = tuple(self.clip_lo)
        self.prog["clip_hi"].value = tuple(self.clip_hi)
        self.tex.use(0)
        self.prog["vol"].value = 0
        # every pixel of the box exactly once, from inside or outside: draw its back faces
        r.ctx.enable(moderngl.CULL_FACE)
        r.ctx.cull_face = "front"
        self.vao.render(moderngl.TRIANGLES)
        r.ctx.cull_face = "back"
        r.ctx.disable(moderngl.CULL_FACE)

    def depth_key(self, r):
        return 1e9  # a backdrop: drawn before the other translucent layers


# ----------------------------------------------------------------------------------------------
# renderer
# ----------------------------------------------------------------------------------------------

class GLRenderer:
    """Headless renderer: ``render()`` -> (height, width, 4) uint8, top row first."""

    def __init__(self, width: int, height: int, samples: int = 8, background=BG) -> None:
        if moderngl is None:
            raise ImportError("moderngl is required for the GPU renderer")
        self.width, self.height = int(width), int(height)
        self.ctx = moderngl.create_standalone_context(backend="egl", require=330)
        samples = min(samples, self.ctx.max_samples)
        size = (self.width, self.height)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(size, 4, samples=samples)],
                                        depth_attachment=self.ctx.depth_renderbuffer(size, samples=samples))
        self.out = self.ctx.framebuffer(color_attachments=[self.ctx.renderbuffer(size, 4)])
        self.background = tuple(background)
        self.camera = OrbitCamera()
        self.layers: list[Layer] = []
        self._programs: dict[str, object] = {}
        self._spheres: dict[int, tuple] = {}
        v, i = _cube_geometry()
        self.cube_vbo, self.cube_ebo = self.ctx.buffer(v.tobytes()), self.ctx.buffer(i.tobytes())

    def program(self, name: str, vs: str, fs: str):
        if name not in self._programs:
            self._programs[name] = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        return self._programs[name]

    def sphere_geometry(self, subdiv: int):
        if subdiv not in self._spheres:
            v, i = icosphere(subdiv)
            self._spheres[subdiv] = (self.ctx.buffer(v.tobytes()), self.ctx.buffer(i.tobytes()))
        return self._spheres[subdiv]

    def add(self, *layers: Layer) -> None:
        for layer in layers:
            if layer not in self.layers:
                layer.setup(self)
                self.layers.append(layer)

    def remove(self, *layers: Layer) -> None:
        for layer in layers:
            if layer in self.layers:
                self.layers.remove(layer)

    def set_common(self, prog) -> None:
        self._set(prog, "view", self._view)
        self._set(prog, "proj", self._proj)
        if "light_dir" in prog:
            prog["light_dir"].value = tuple(self._light)

    @staticmethod
    def _set(prog, name: str, m: np.ndarray) -> None:
        if name in prog:
            prog[name].write(_mat(m))

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World points (n, 3) -> NDC (n, 2) and depth sign (in front of the camera)."""
        p = np.c_[np.atleast_2d(np.asarray(pts, float)), np.ones(len(np.atleast_2d(pts)))]
        c = (self.camera.proj(self.width / self.height) @ self.camera.view() @ p.T).T
        return c[:, :2] / c[:, 3:4]

    def render(self) -> np.ndarray:
        cam = self.camera
        self._view = cam.view()
        self._proj = cam.proj(self.width / self.height)
        light = np.array([0.35, 0.55, 0.76])
        self._light = light / np.linalg.norm(light)
        ctx = self.ctx
        self.fbo.use()
        self.fbo.depth_mask = True
        ctx.clear(*self.background, 1.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        for layer in self.layers:
            if layer.visible:
                layer.draw_opaque(self)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.fbo.depth_mask = False
        for layer in sorted([l for l in self.layers if l.visible], key=lambda l: -l.depth_key(self)):
            layer.draw_translucent(self)
        self.fbo.depth_mask = True
        ctx.copy_framebuffer(self.out, self.fbo)
        data = np.frombuffer(self.out.read(components=4), np.uint8).reshape(self.height, self.width, 4)
        img = data[::-1].copy()
        img[..., 3] = 255
        return img


__all__ = ["Arrows", "GLRenderer", "Layer", "Lines", "Mesh", "OrbitCamera", "Points", "Quad", "Spheres", "Tubes", "VolumeMIP",
           "Voxels", "icosphere", "merge_meshes", "tube_mesh"]
