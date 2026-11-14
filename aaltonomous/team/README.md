# G1 sorter: strategy and how to run it on Verda

## Strategy

Rung 2 is worth double (+2 per object), and the finals weight an unseen instruction ×1.5. The reference baseline
ignores the instruction, so on rung 2 it scores negative. So we keep the classical route the organisers have
already tuned and make it read instructions and see shapes:

| Piece | What it does | Status |
|---|---|---|
| `team/instruction.py` | Rule-based parser: synonyms, either word order, "respectively", "the rest / remaining bin", negations ("not the left bin"), "everything except red", shape rules ("cubes go left", "red cubes …"), elimination of the missing colour. No network, < 1 ms. Outputs the same mapping format the official scorer uses (`scene.resolve_bin`). | All 288 training phrasings, 500 random `make_instruction` draws and 37 hand-written unseen paraphrases correct (`test_instruction.py`) |
| `team/shape.py`, `team/perception.py` | Baseline HSV + depth detector, plus **shape and yaw** from the top-face point cloud (hull vs. min rectangle / min triangle) | Rendered mock world: 131/131 shapes right, cube yaw ≤ 1°, triangle yaw ≤ 9° |
| `team/policy.py` | The organisers' **sim oracle** planner (face-aligned pinch for cubes/triangles, relays for cross-body objects, spread drop points, planning the next pick while carrying), with every privileged input swapped for camera data: poses come from perception, object ids from a position tracker, bin counts from our own drop log. Adds one extra look before declaring "done". | Mock world: 68/68 objects across 16 episodes, 0 safety clips. Baseline on the same rung-2 seeds: 7/43, score −2.2 |
| `team/server.py`, `docker/team.Dockerfile` | Submission server (threaded, hold chunks while planning). CPU-only, same base image as the baseline | Protocol dry run: pass, 0 errors, 30 Hz. Image not built yet |

The mock world attaches objects kinematically, so **grasp quality is untested**. Isaac on an RTX GPU is the real test.
The baseline's misses are mostly triangles slipping, and the face-aligned grasp is aimed at those.

## Run on the Verda instance (RTX, sim container running per the Quickstart)

```bash
# 1. on the host: get our repo (starter kit + team code)
git clone git@github.com:lenniarvonen/vaiski.git && cd vaiski

# 2. copy it into the sim container and run the unit tests
docker cp aaltonomous/team aalto:/opt/aalto/aaltonomous/
docker exec -it aalto bash -lc 'cd /opt/aalto && aalto py -m unittest aaltonomous/team/test_instruction.py'

# 3. start our policy server inside the container (tmux pane 1)
docker exec -it aalto bash -lc 'cd /opt/aalto && aalto py -m aaltonomous.team.server'

# 4. score it (tmux pane 2). Compare with `aalto baseline` (≈ 76–82 % on rung 1)
docker exec -it aalto bash -lc 'cd /opt/aalto && aalto eval --policy ws://localhost:8000 --team me --task sort_fixed --trials 10'
docker exec -it aalto bash -lc 'cd /opt/aalto && aalto eval --policy ws://localhost:8000 --team me --task sort_language --trials 10'
# then the same with domain randomisation: restart the sim with AALTO_DR=train aalto sim

# 5. build and gate the submission image (host, needs the URDF: bash aaltonomous/cloud/setup_instance.sh or copy
#    /opt/aalto from the container as in Quickstart §2)
docker build -f aaltonomous/docker/team.Dockerfile -t team/policy:v1 .
python -m aaltonomous.deploy.replay --synthetic 100 --container team/policy:v1
```

What to send back after step 4: the score table and `runs/eval/*/eval.json`, plus the server log lines
starting with `[team] saw` (they show the detected shapes). If triangles still slip, the knob is `AALTO_TRI_PHI`
(finger angle against the apex, default 105°).

## Next steps, in order of expected payoff

1. Isaac numbers for both rungs, with DR on and off. Fix whatever fails first: grasp, perception or time.
2. Tune perception on **Drop 1** real frames: HSV ranges under venue light, depth holes, and `xy_offset` for head
   extrinsics error. Replay drop frames through `ShapePerception` offline.
3. Speed: 5 objects take ~55 s in the mock. Using both arms at once would leave margin under the 120 s limit.
4. Optional: a small local LLM as a fallback when `parse(...).ok` is False (no network at the finals).
