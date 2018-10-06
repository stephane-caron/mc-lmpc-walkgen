# Multi-contact Walking by Linearized Model Predictive Control

Source code for https://hal.archives-ouvertes.fr/hal-01349880

## Installation

On Ubuntu 14.04, once you have [installed
OpenRAVE](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html),
do:

```bash
sudo apt-get install cython python python-dev python-pip python-scipy python-shapely
sudo pip install pycddlib quadprog pyclipper
```

Then, clone the repository and its submodule via:

```bash
git clone --recursive https://github.com/stephane-caron/3d-walking-lmpc.git
```

If you already have [pymanoid](https://github.com/stephane-caron/pymanoid)
installed on your system, be sure that its version matches that of the
submodule.

## Usage

The ``walk.sh`` script from the top-level directory opens the staircase
simulation.

For a more detailed analysis, there are three subfolders in this repository,
corresponding to different Sections of the paper:

- *Section IV*: [cones/](cones/) contains scripts to display and play with the CoM
  acceleration cones
- *Section VI*: [staircase/](staircase/) walks the humanoid model around a
  circular staircase with tilted stepping stones
- *Appendix*: [sep/](sep/) compares four algorithms calculating the static-equilibrium
  polygon

Due to the copyright problem, we cannot release the COLLADA model ``HRP4R.dae``
used to produce the accompanying video and paper illustrations. We have
replaced it with
[JVRC-1](https://github.com/stephane-caron/openrave_models/tree/master/JVRC-1),
which has the same kinematic chain.
