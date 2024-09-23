# 1. Test setup:
# docker run -it --rm --gpus all pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build -f  Dockerfile -t img . && \
# docker run -it --rm --gpus all -v $PWD:/workspace img \
#   sh xvfb_run.sh python3 dreamer.py \
#   --configs dmc_vision --task dmc_walker_walk \
#   --logdir "./logdir/dmc_walker_walk"
#
# 3. See results:
# tensorboard --logdir ~/logdir
#
# 4. To set up Atari or Minecraft environments, please check the scripts located in "env/setup_scripts".

# System
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
    vim libgl1-mesa-glx libosmesa6 \
    wget unrar cmake g++ libgl1-mesa-dev \
    libx11-6 openjdk-8-jdk x11-xserver-utils xvfb \
    && apt-get clean
RUN pip3 install --upgrade pip

# Envs
ENV NUMBA_CACHE_DIR=/tmp

WORKDIR /workspace
COPY requirements.txt .

# Install requiremqnts
RUN pip3 install -r requirements.txt