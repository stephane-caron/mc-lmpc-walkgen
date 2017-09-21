# 3D Walking by Linearized Model Predictive Control

Source code for https://hal.archives-ouvertes.fr/hal-01349880

## Abstract

We present a multi-contact walking pattern generator based on preview-control
of the 3D acceleration of the center of mass (CoM). A key point in the design
of our algorithm is the calculation of contact-stability constraints. Thanks to
a mathematical observation on the algebraic nature of the frictional wrench
cone, we show that the 3D volume of feasible CoM accelerations is a always a
downward-pointing cone. We reduce its computation to a convex hull of (dual) 2D
points, for which optimal O(n log n) algorithms are readily available. This
reformulation brings a significant speedup compared to previous methods, which
allows us to compute time-varying contact-stability criteria fast enough for
the control loop. Next, we propose a conservative trajectory-wide
contact-stability criterion, which can be derived from CoM-acceleration volumes
at marginal cost and directly applied in a model-predictive controller. We
finally implement this pipeline and exemplify it with the HRP-4 humanoid model
in multi-contact dynamically walking scenarios.

<img src="https://scaron.info/images/humanoids-2016.png" height="350" />

Authors:
[Stéphane Caron](https://scaron.info) and
[Abderrahmane Kheddar](http://www.lirmm.fr/lirmm_eng/users/utilisateurs-lirmm/equipes/idh/abderrahmane-kheddar)

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
git clone --recursive https://github.com/stephane-caron/3d-com-lmpc.git
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

## Questions?

Feel free to post your questions or comments in the issue tracker.
