#!/bin/sh

# Install Java 8 before running this script by either of the following methods.

# 1. Use docker
# $ apt-get update
# $ apt-get install -y openjdk-8-jdk
# 2. Use conda
# $ conda install -c conda-forge openjdk=8

pip3 install https://github.com/NM512/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
# Downgrade to install old gym
pip3 install setuptools==60.0.0
pip3 install pip==22.0
pip3 install gym==0.19.0
pip3 install cloudpickle==2.2.1
