#!/bin/sh

# Run this script to install Atari
pip3 install atari-py==0.2.9
pip3 install opencv-python==4.7.0.72
mkdir roms && cd roms
wget -L -nv http://www.atarimania.com/roms/Roms.rar
unrar x -o+ Roms.rar
python3 -m atari_py.import_roms ROMS
cd .. && rm -rf roms
