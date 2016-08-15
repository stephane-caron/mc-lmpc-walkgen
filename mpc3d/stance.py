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

# from cwc import compute_cwc_pyparma
from pymanoid import ContactSet

try:
    from hrp4_pymanoid import HRP4 as RobotModel
except ImportError:
    from pymanoid.robots import JVRC1 as RobotModel


class Stance(ContactSet):

    def __init__(self, phase, left_foot=None, right_foot=None,
                 ref_velocity=0.4):
        """
        Create a new stance.

        INPUT:

        - ``phase`` -- corresponding phase in the FSM
        - ``left_foot`` -- (optional) left foot contact
        - ``right_foot`` -- (optional) right foot contact
        - ``ref_velocity`` -- (default: 0.4 m/s) target forward COM velocity
        """
        contacts = {}
        if left_foot:
            contacts['left_foot'] = left_foot
        if right_foot:
            contacts['right_foot'] = right_foot
        target_foot = left_foot if phase[-1] == 'L' else right_foot
        self.com = target_foot.p + [0., 0., RobotModel.leg_length]
        self.comd = ref_velocity * target_foot.t
        self.is_double_support = phase.startswith('DS')
        self.is_single_support = phase.startswith('SS')
        self.left_foot = left_foot
        self.right_foot = right_foot
        self.phase = phase
        super(Stance, self).__init__(contacts)
        self.compute_stability_criteria()

    def compute_stability_criteria(self):
        self.cwc = self.compute_wrench_cone([0, 0, 0])  # calls cdd
        # self.cwc = compute_cwc_pyparma(self, [0, 0, 0])
        m = RobotModel.mass  # however, the SEP does not depend on this
        self.sep = self.compute_static_equilibrium_area(m)
