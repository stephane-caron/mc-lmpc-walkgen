#!/bin/zsh

dt=3e-2  # [s], same value as used for simulation timestep
FOLDER="camera"
FRAMERATE=$(echo "print int(1. / ${dt})" | python)
EXT=mp4

echo $FRAMERATE

avconv -r ${FRAMERATE} -qscale 1 -i ${FOLDER}/%05d.png ./${FOLDER}.${EXT}
