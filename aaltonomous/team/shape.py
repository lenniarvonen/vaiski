"""Shape + yaw of an object from its top-face points (pelvis frame), so the planner can grasp across flat faces.

Top faces seen from above: cube = 45 mm square, cylinder = 45 mm disc, triangle = equilateral, altitude 45 mm.
Scale-free features of the convex hull H of the top-face xy points:
    rect = area(H) / area(min-area rectangle)   square ~1.0, disc ~0.79, triangle ~0.5
    tri  = area(H) / area(min enclosing triangle)  square ~0.5, disc ~0.60, triangle ~1.0
Nearest prototype wins; too far from all of them -> shape None (the planner then uses its generic grasp).

Yaw convention = the scene/oracle one (scene/export.py meshes): cube faces at yaw + k*90 deg, triangle apex at
yaw + 90 deg (faces at yaw - 90, +30, +150). Returned folded: cube to [-45, 45) deg, triangle to [-60, 60) deg.
"""
from __future__ import annotations

import numpy as np

PROTOTYPES = {"cube": (0.97, 0.52), "cylinder": (0.79, 0.62), "triangle": (0.52, 0.95)}
MAX_DIST = 0.16


def features(xy_m):
    import cv2
    p = (np.asarray(xy_m, float) * 1000.0).astype(np.float32).reshape(-1, 1, 2)
    if len(p) < 12:
        return None
    hull = cv2.convexHull(p)
    a = float(cv2.contourArea(hull))
    if a < 150.0:                       # < 1.5 cm^2: not a top face
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(hull)
    tri_area, tri = cv2.minEnclosingTriangle(hull)
    return {"area": a, "rect": a / max(w * h, 1e-6), "tri": a / max(float(tri_area), 1e-6),
            "rect_angle": np.radians(ang), "triangle": np.asarray(tri, float).reshape(3, 2) / 1000.0,
            "hull": hull.reshape(-1, 2) / 1000.0}


def classify(xy_m):
    """-> (shape | None, yaw | None, info dict)."""
    f = features(xy_m)
    if f is None:
        return None, None, {}
    d = {k: float(np.hypot(f["rect"] - r, f["tri"] - t)) for k, (r, t) in PROTOTYPES.items()}
    shape = min(d, key=d.get)
    info = {"rect": round(f["rect"], 3), "tri": round(f["tri"], 3), "area_cm2": round(f["area"] / 100, 2),
            "dist": round(d[shape], 3)}
    if d[shape] > MAX_DIST:
        return None, None, info
    if shape == "cube":
        yaw = (f["rect_angle"] + np.pi / 4) % (np.pi / 2) - np.pi / 4
    elif shape == "triangle":
        tri = f["triangle"]
        c = f["hull"].mean(0)
        v = tri[np.argmax(np.linalg.norm(tri - c, axis=1))] - c      # any vertex works (120 deg symmetry)
        a = np.arctan2(v[1], v[0]) - np.pi / 2
        yaw = (a + np.pi / 3) % (2 * np.pi / 3) - np.pi / 3
    else:
        yaw = None
    return shape, (None if yaw is None else float(yaw)), info


def triangle_centroid(xy_m):
    """Centroid of the enclosing triangle (robust to a missing corner in the point cloud)."""
    f = features(xy_m)
    return None if f is None else f["triangle"].mean(0)
