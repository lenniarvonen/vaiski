"""TeamPolicy: the organisers' sim oracle (sim/oracle.py) with its privileged inputs replaced by what a submission
is allowed to see.

    oracle input (privileged)                 -> here (obs only)
    scene-service object poses + shape + yaw  -> head RGB-D: HSV + depth (baseline) + shape/yaw (team/shape.py)
    object names from the sim                 -> a small position tracker (ids survive across looks and relays)
    objects already in each bin (sim state)   -> our own count of what we dropped where
    template-table instruction lookup         -> team/instruction.py (unseen phrasings, shapes, negations)

Kept from the oracle (non-privileged planning): face-aligned pinch for cubes/triangles, joint-space carry shortcut,
relay of cross-body objects via the centre line, drop points spread inside a bin, prefetch planning of the next
object while the current one is carried, all plans pre-checked against the safety filter (0 clips).
Added: one extra look before declaring the counter empty.
"""
from __future__ import annotations

import itertools

import numpy as np

from aaltonomous.interface import tasks
from aaltonomous.sim.oracle import DROP_OFFSETS, RELAY_POINTS, OraclePolicy

from .instruction import parse
from .perception import ShapePerception, describe


class _Tracker:
    """Gives detections stable names: same colour within `radius` of a known object -> same name."""

    def __init__(self, radius=0.04):
        self.radius = radius
        self.reset()

    def reset(self):
        self.known = {}            # name -> (colour, xy)
        self.expect = {}           # name -> (colour, xy) where an object will be after a relay
        self._ids = itertools.count()

    def assign(self, dets):
        free = dict(self.known)
        for d in sorted(dets, key=lambda d: -d.n_px):
            xy = d.top[:2]
            best = None
            for pool in (free, self.expect):
                cands = [(np.linalg.norm(xy - p), n) for n, (c, p) in pool.items()
                         if c == d.color and np.linalg.norm(xy - p) < self.radius]
                if cands:
                    best = min(cands)[1]
                    pool.pop(best)
                    break
            d.name = best if best is not None else f"{d.color}{next(self._ids)}"
            self.known[d.name] = (d.color, xy.copy())
        return dets


class _ObsPerception:
    """Perception.detect() without arguments (the oracle's call style) on the latest head frame."""

    def __init__(self, perception, tracker):
        self.p, self.tracker, self.obs = perception, tracker, None

    def detect(self, rgb=None, depth=None, debug=False):
        if rgb is None:
            if self.obs is None:
                return ([], []) if debug else []
            rgb, depth = self.obs["head_rgb"], self.obs["head_depth"]
        if debug:
            dets, rej = self.p.detect(rgb, depth, debug=True)
            return self.tracker.assign(dets), rej
        return self.tracker.assign(self.p.detect(rgb, depth))

    def masks(self, rgb):
        return self.p.masks(rgb)


class TeamPolicy(OraclePolicy):
    def __init__(self, overrides=None, threaded=True, log=print, prefetch=True, extra_looks=1, **kw):
        self.tracker = _Tracker()
        self.extra_looks = extra_looks
        self.placed = {}
        self.parse = None
        super().__init__(client=None, overrides=overrides, threaded=threaded, log=log, prefetch=prefetch, **kw)
        self.perception = _ObsPerception(ShapePerception(self.cfg, self.scene, self.cal), self.tracker)

    # ------------------------------------------------------------------ episode
    def reset(self, info=None):
        if isinstance(info, str):
            info = {"instruction": info}
        info = dict(info or {})
        text = info.get("instruction") or tasks.FIXED_INSTRUCTION
        if not info.get("mapping"):
            self.parse = parse(text, task=info.get("task"))
            info["mapping"] = dict(self.parse.mapping) or dict(tasks.FIXED_MAPPING)
        super().reset(info)
        self.mapping_source = "team-parser"
        self.tracker.reset()
        self.placed = {b: 0 for b in self.scene.bin_names}
        self._empty = 0
        if self.parse is not None:
            self.log(f"[team] instruction {text!r} -> {self.mapping} ok={self.parse.ok} {self.parse.notes}")

    # ------------------------------------------------------------------ privileged bits replaced
    def _look(self, obs):
        self.perception.obs = obs
        return super()._look(obs)

    def _update_drop_offsets(self, exclude=None):
        self.planner.drop_offsets = {b: DROP_OFFSETS[k % len(DROP_OFFSETS)] for b, k in self.placed.items()}

    def _start(self, plan, t):
        tb = str(plan.target_bin or "")
        if tb in self.placed:
            self.placed[tb] += 1
        return super()._start(plan, t)

    def _commit(self, p):
        super()._commit(p)
        if getattr(p, "relay", None) is not None and getattr(p, "obj", None):
            c = self.tracker.known.get(p.obj, (p.color, None))[0]
            self.tracker.expect[p.obj] = (c, np.asarray(RELAY_POINTS[p.relay], float))

    def _plan_next(self, arm, hand, exclude=None, commit=True):
        res = super()._plan_next(arm, hand, exclude=exclude, commit=commit)
        if commit and self.last_detections:
            self.log("[team] saw " + "; ".join(describe(d) for d in self.last_detections))
        return res

    def _after_look(self, res, t):
        if res[0] == "none" and self._empty < self.extra_looks and t < self.time_limit - 5:
            self._empty += 1          # one more look (an arm in view, a blob split by a reflection, ...)
            self.state = "settle"
            return self._hold()
        if res[0] == "plan":
            self._empty = 0
        return super()._after_look(res, t)
