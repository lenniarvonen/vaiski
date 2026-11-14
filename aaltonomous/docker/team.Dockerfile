# Our submission: TeamPolicy (aaltonomous/team) = camera perception (HSV + depth + shape/yaw) + the oracle's planner
# + the rung-2 instruction parser. CPU only, same base and wheels as the reference baseline image.
#
#   docker build -f aaltonomous/docker/team.Dockerfile -t team/policy:v1 .          # from the repo root
#   docker run --rm -p 8000:8000 team/policy:v1
#   python -m aaltonomous.deploy.replay --synthetic 100 --container team/policy:v1 --gpus none
#
# Calibration + config are baked in: rebuild after a calibration update from the organisers.
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy==2.2.6 scipy==1.15.2 opencv-python-headless==4.10.0.84 pyyaml \
        msgpack "websockets>=12" pin==3.3.1

WORKDIR /app
COPY third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf third_party/xr_teleoperate/assets/g1/
COPY aaltonomous/__init__.py aaltonomous/
COPY aaltonomous/deploy/__init__.py aaltonomous/deploy/protocol.py aaltonomous/deploy/policy_server.py aaltonomous/deploy/
COPY aaltonomous/interface/__init__.py aaltonomous/interface/safety.py aaltonomous/interface/calibration.py \
     aaltonomous/interface/tasks.py aaltonomous/interface/joint_spec.yaml aaltonomous/interface/calibration.yaml \
     aaltonomous/interface/
COPY aaltonomous/scene/__init__.py aaltonomous/scene/scene.py aaltonomous/scene/scene.yaml aaltonomous/scene/
COPY aaltonomous/baselines/__init__.py aaltonomous/baselines/
COPY aaltonomous/baselines/classical/*.py aaltonomous/baselines/classical/config.yaml aaltonomous/baselines/classical/
COPY aaltonomous/sim/oracle.py aaltonomous/sim/
COPY aaltonomous/sim/aalto_task/__init__.py aaltonomous/sim/aalto_task/layout.py aaltonomous/sim/aalto_task/
COPY aaltonomous/team/__init__.py aaltonomous/team/instruction.py aaltonomous/team/shape.py \
     aaltonomous/team/perception.py aaltonomous/team/policy.py aaltonomous/team/server.py aaltonomous/team/
# Load-time check at build: config, URDF, calibration and the parser work (fails the build, not the run).
RUN python3 -c "from aaltonomous.team.server import TeamServerPolicy; TeamServerPolicy(); \
from aaltonomous.team.instruction import parse; assert parse('Sort the objects: red into right, blue into left, yellow into center.').mapping == {'red':'right','blue':'left','yellow':'center'}"

EXPOSE 8000
CMD ["python3", "-m", "aaltonomous.team.server"]
