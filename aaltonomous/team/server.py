"""Our submission: TeamPolicy behind the policy-server protocol on :8000.

    python -m aaltonomous.team.server                     # what the Docker image runs
    docker build -f aaltonomous/docker/team.Dockerfile -t team/policy:v1 .

CPU only. Threaded like the reference baseline: act() answers at once (hold chunks while looking/planning), motion
comes back in chunks, so the command rate stays ~30 Hz.
Env: AALTO_TIME_LIMIT (s, default 120), AALTO_ARM_VEL (rad/s override), AALTO_PROFILE (config profile, default sim),
TEAM_PREFETCH=0 to plan only at looks.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from aaltonomous.deploy.policy_server import Policy, serve

from .policy import TeamPolicy

log = logging.getLogger("aaltonomous.team")


class TeamServerPolicy(Policy):
    name = "team-vision-oracle"

    def __init__(self, **kw):
        tl = float(os.environ.get("AALTO_TIME_LIMIT", 120.0))
        ov = {}
        if os.environ.get("AALTO_ARM_VEL"):
            ov["motion.arm_max_vel"] = float(os.environ["AALTO_ARM_VEL"])
        self.pol = TeamPolicy(overrides=ov, time_limit=tl, prefetch=os.environ.get("TEAM_PREFETCH", "1") != "0",
                              log=lambda m: log.info(m), **kw)
        self.action_horizon = int(self.pol.cfg["motion"]["chunk"])

    def reset(self, instruction, task=None, seed=None):
        job, t0 = self.pol._job, time.monotonic()        # let a look from the last episode finish (not thread-safe)
        while job is not None and job["result"] is None and time.monotonic() - t0 < 30.0:
            time.sleep(0.02)
        self.pol.reset({"instruction": instruction, "task": task})
        log.info("reset task=%s seed=%s", task, seed)

    def act(self, obs):
        a = self.pol.act(obs)
        return {"arm_q_target": np.asarray(a["arm_q_target"], np.float64),
                "hand_q_target": np.asarray(a["hand_q_target"], np.float64), "done": bool(a.get("done", False))}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    pol = TeamServerPolicy()
    log.info("team policy loaded in %.1f s", time.monotonic() - t0)
    serve(pol, port=int(os.environ.get("PORT", 8000)))
