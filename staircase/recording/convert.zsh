#!/bin/zsh

if [[ $# == 0 ]] then
    echo "Usage: $0 <camera_folder>"
    exit
fi

dt=3e-2  # [s], same value as used for simulation timestep
CAMERA=$1
FRAMERATE=$(echo "print int(1. / ${dt})" | python)
EXT=mp4

echo $FRAMERATE

avconv -r ${FRAMERATE} -qscale 1 -i ${CAMERA}/%05d.png ./${CAMERA}.${EXT}
