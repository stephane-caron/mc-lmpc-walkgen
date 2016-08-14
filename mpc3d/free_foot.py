#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of 3d-mpc <https://github.com/stephane-caron/3d-mpc>.
#
# 3d-mpc is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# 3d-mpc is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# 3d-mpc. If not, see <http://www.gnu.org/licenses/>.

from numpy import hstack
from pymanoid import Contact
from pymanoid.rotations import quat_slerp
from pymanoid.rotations import rotation_matrix_from_quat


def pose_interp(pose0, pose1, t):
    """Linear pose interpolation."""
    pos = pose0[4:] + t * (pose1[4:] - pose0[4:])
    quat = quat_slerp(pose0[:4], pose1[:4], t)
    return hstack([quat, pos])


class FreeFoot(Contact):

    def __init__(self, **kwargs):
        super(FreeFoot, self).__init__(X=0.2, Y=0.1, **kwargs)
        self.end_pose = None
        self.mid_pose = None
        self.start_pose = None

    def reset(self, start_pose, end_pose):
        mid_pose = pose_interp(start_pose, end_pose, .5)
        mid_n = rotation_matrix_from_quat(mid_pose[:4])[0:3, 2]
        mid_pose[4:] += 0.2 * mid_n
        self.set_pose(start_pose)
        self.start_pose = start_pose
        self.mid_pose = mid_pose
        self.end_pose = end_pose

    def update_pose(self, x):
        """Update pose for x \in [0, 1]."""
        if x >= 1.:
            return
        elif x <= .5:
            pose0 = self.start_pose
            pose1 = self.mid_pose
            y = 2. * x
        else:  # .5 < x < 1
            pose0 = self.mid_pose
            pose1 = self.end_pose
            y = 2. * x - 1.
        pos = (1. - y) * pose0[4:] + y * pose1[4:]
        quat = quat_slerp(pose0[:4], pose1[:4], y)
        self.set_pose(hstack([quat, pos]))
