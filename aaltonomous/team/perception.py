"""Head RGB-D perception = the baseline's HSV + depth detector, plus shape and yaw per object (team/shape.py).

Detections carry .shape ('cube' | 'cylinder' | 'triangle' | None) and .yaw, which the oracle planner's face-aligned
grasp (sim/oracle.py _OraclePlanner) uses to pinch across two flat faces instead of across a corner. With
shape-based instructions ("cubes go left") the shape also picks the bin.
"""
from __future__ import annotations

import numpy as np

from aaltonomous.baselines.classical.perception import Detection, Perception

from .shape import classify, triangle_centroid


class ShapePerception(Perception):
    def _from_depth(self, color, u, v, d, area):
        det = super()._from_depth(color, u, v, d, area)
        if det is None:
            return None
        P = self.cal.pixel_to_pelvis(self.cam, u, v, d)
        z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
        top = P[P[:, 2] >= z_top - self.cfg["top_band"]]
        shape, yaw, info = classify(top[:, :2])
        det.shape, det.yaw, det.shape_info = shape, yaw, info
        if shape == "triangle":       # pixel weighting biases a triangle's centre: use the enclosing triangle
            c = triangle_centroid(top[:, :2])
            if c is not None and np.linalg.norm(c - (det.top[:2] - self.xy_offset)) < 0.012:
                det.top = np.r_[c + self.xy_offset, det.top[2]]
        return det

    def _from_plane(self, color, uv, area, n_depth):
        det = super()._from_plane(color, uv, area, n_depth)
        det.shape, det.shape_info = None, {}
        return det


def describe(d: Detection):
    s = getattr(d, "shape", None)
    y = getattr(d, "yaw", None)
    return f"{d.color} {s or '?'} at ({d.top[0]:+.3f},{d.top[1]:+.3f}) yaw={'-' if y is None else f'{np.degrees(y):.0f}'}"
