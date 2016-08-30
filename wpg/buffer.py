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

from numpy import zeros
from simulation import Process
from threading import Lock


class PreviewBuffer(Process):

    """
    These buffers store COM accelerations output by the preview controller and
    execute them until the next update.
    """

    def __init__(self, com):
        self.com = com
        self.preview_index = 0
        self.preview_lock = Lock()
        self.preview = None
        self.rem_time = 0.

    def update_preview(self, preview):
        with self.preview_lock:
            self.preview_index = 0
            self.preview = preview

    def get_next_preview_window(self):
        """
        Returns the next pair ``(comdd, dT)`` in the preview window, where
        acceleration ``comdd`` is executed during ``dT``.
        """
        with self.preview_lock:
            if self.preview is None:
                return (zeros(3), 0.)
            j = 3 * self.preview_index
            comdd = self.preview.U[j:j + 3]
            if comdd.shape[0] == 0:
                comdd = zeros(3)
                self.preview = None
            self.preview_index += 1
            return (comdd, self.preview.timestep)

    @property
    def preview_was_updated(self):
        """Returns True when preview was updated since last read."""
        return self.preview_index == 0

    def on_tick(self, sim):
        if self.rem_time < sim.dt:
            (self.comdd, self.rem_time) = self.get_next_preview_window()
        self.com.integrate_acceleration(self.comdd, sim.dt)
        self.rem_time -= sim.dt
