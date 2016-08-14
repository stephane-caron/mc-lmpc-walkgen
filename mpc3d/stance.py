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

from cwc import compute_cwc_pyparma
from pymanoid import ContactSet

try:
    from hrp4_pymanoid import HRP4 as RobotModel
except ImportError:
    from pymanoid.robots import JVRC1 as RobotModel


class Stance(ContactSet):

    def __init__(self, state, left_foot=None, right_foot=None):
        contacts = {}
        if left_foot:
            contacts['left_foot'] = left_foot
        if right_foot:
            contacts['right_foot'] = right_foot
        target_foot = left_foot if state[-1] == 'L' else right_foot
        self._cwc = None
        self._sep = None
        self.com = target_foot.p + [0., 0., RobotModel.leg_length]
        self.left_foot = left_foot
        self.right_foot = right_foot
        self.state = state
        self.target_foot = target_foot
        super(Stance, self).__init__(contacts)

    @property
    def is_double_support(self):
        return self.state.startswith('DS')

    @property
    def is_single_support(self):
        return self.state.startswith('SS')

    @property
    def comd(self):
        return 0.4 * self.target_foot.t

    @property
    def cwc(self):
        """Contact Wrench Cone at world origin"""
        if self._cwc is None:
            self._cwc = self.compute_wrench_cone([0, 0, 0])  # cdd
        elif self._cwc is None:
            self._cwc = compute_cwc_pyparma(self, [0, 0, 0])
        return self._cwc

    @property
    def sep(self):
        """Static-equilibrium polygon as list of vertices"""
        if self._sep is None:
            m = RobotModel.mass  # however, the SEP does not depend on this
            self._sep = self.compute_static_equilibrium_area(m)
        return self._sep
