"""Mock-world renderer that draws real triangular prisms (the stock mock draws triangles as boxes), so shape/yaw
perception can be tested without Isaac. Same geometry as scene/export.py: apex along +y at yaw 0, vertices 30 mm
from the centroid (altitude 45 mm), faces 15 mm from it."""
from __future__ import annotations

import numpy as np

from aaltonomous.baselines.classical.mock_world import Renderer, _yaw_R


class ShapeRenderer(Renderer):
    def _prism(self, rgb, zb, center, yaw, alt, hz, col):
        r_in = alt / 3.0
        normals = [np.array([np.cos(a), np.sin(a), 0.0]) for a in yaw + np.radians([-90.0, 30.0, 150.0])]
        planes = [(n, r_in) for n in normals] + [(np.array([0, 0, 1.0]), hz), (np.array([0, 0, -1.0]), hz)]
        win = self._pixels(center, _yaw_R(yaw), np.array([alt * 2 / 3, alt * 2 / 3, hz]))
        if win is None:
            return
        d = self.rays[win]
        o = self.o - center
        tn = np.zeros(d.shape[:2])
        tf = np.full(d.shape[:2], np.inf)
        nz = np.zeros(d.shape[:2])
        miss = np.zeros(d.shape[:2], bool)
        for n, off in planes:
            den = d @ n
            num = off - o @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = num / den
            ent = den < -1e-12
            ext = den > 1e-12
            upd = ent & (t > tn)
            nz = np.where(upd, abs(n[2]), nz)
            tn = np.where(upd, t, tn)
            tf = np.where(ext, np.minimum(tf, t), tf)
            miss |= (~ent & ~ext) & (num < 0)
        hit = ~miss & (tn <= tf) & (tn > 0)
        self._write(rgb, zb, win, hit, tn, nz, col)

    def render(self, objects, rng=None, depth_noise_mm=0.0, hole_frac=0.0):
        tris = [o for o in objects if o.get("shape") == "triangle"]
        rest = [o for o in objects if o.get("shape") != "triangle"]
        rgb, depth = super().render(rest, rng, 0.0, 0.0)
        rgb = rgb.astype(float)
        zb = np.where(depth > 0, depth / 1000.0, np.inf)
        for ob in tris:
            self._prism(rgb, zb, np.asarray(ob["pos"], float), float(ob.get("yaw", 0.0)), ob["size"],
                        ob.get("height", ob["size"]) / 2, self.colors[ob["color"]])
        depth = np.where(np.isfinite(zb), zb * 1000.0, 0.0)
        if rng is not None and depth_noise_mm > 0:
            depth = depth + rng.normal(0, depth_noise_mm, depth.shape)
        if rng is not None and hole_frac > 0:
            depth[rng.random(depth.shape) < hole_frac] = 0
        return np.clip(rgb, 0, 255).astype(np.uint8), np.clip(depth, 0, 65535).astype(np.uint16)


def make_shape_mock_env(seed=0, **kw):
    """make_mock_env with triangles drawn as prisms."""
    from aaltonomous.baselines.classical.mock_world import make_mock_env
    from aaltonomous.interface.calibration import Calibration
    env = make_mock_env(seed=seed, **kw)
    env.backend.renderer = ShapeRenderer(env.scene, Calibration.load())
    return env
