"""In-process episodes on the rendered mock world (no GPU): plumbing, parsing, tracking, relays, safety clips.

    python -m aaltonomous.team.mock_eval --task sort_language --seeds 0 1 2 3 4 [--policy baseline]

Grasps in the mock are kinematic (an object attaches when the hand closes near it), so this does NOT measure grasp
quality: use the Isaac sim for that (aalto eval).
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def run_episode(policy_name, task, seed, time_limit=120.0, verbose=False):
    from aaltonomous.team.mock_shapes import make_shape_mock_env
    env = make_shape_mock_env(seed=seed)
    env.task = task
    env.time_limit = time_limit
    obs = env.reset()
    log = print if verbose else (lambda *a: None)
    if policy_name == "baseline":
        from aaltonomous.baselines.classical.policy import ClassicalPolicy
        pol = ClassicalPolicy(threaded=False, safety=env.filter.cfg, time_limit=time_limit, log=log)
    else:
        from aaltonomous.team.policy import TeamPolicy
        pol = TeamPolicy(threaded=False, prefetch=False, safety=env.filter.cfg, time_limit=time_limit, log=log)
    pol.reset({"instruction": obs["instruction"], "task": task})
    clips, info = 0, {}
    while True:
        obs, info = env.step(pol.act(obs))
        clips += int(bool(info["clipped"]))
        if info["episode_over"]:
            break
    mapping = env.mapping or {"red": "left", "yellow": "center", "blue": "right"}
    out = env.backend.outcomes(mapping)
    score = env.scene.score(task, out)
    return {"seed": seed, "instruction": obs["instruction"], "n": len(out), "correct": out.count("correct"),
            "wrong": out.count("wrong"), "score": score, "clips": clips, "t": round(info["t"], 1),
            "end": info["end_reason"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sort_language")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--policy", default="team", choices=["team", "baseline"])
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    rows = []
    for s in a.seeds:
        t0 = time.time()
        r = run_episode(a.policy, a.task, s, verbose=a.v)
        rows.append(r)
        print(f"seed {s}: {r['correct']}/{r['n']} correct, {r['wrong']} wrong, score {r['score']}, clips {r['clips']}, "
              f"sim t {r['t']} s ({r['end']}), wall {time.time() - t0:.0f} s | {r['instruction']}")
    tot = sum(r["n"] for r in rows)
    print(f"TOTAL {a.policy} {a.task}: {sum(r['correct'] for r in rows)}/{tot} correct, "
          f"{sum(r['wrong'] for r in rows)} wrong, mean score {np.mean([r['score'] for r in rows]):.2f}, "
          f"clips {sum(r['clips'] for r in rows)}")


if __name__ == "__main__":
    main()
