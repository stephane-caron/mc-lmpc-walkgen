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

import threading
import time


class Simulation(object):

    def __init__(self, dt, slowdown=None):
        """
        Create a new simulation object.

        INPUT:

        - ``dt`` -- time interval between two ticks in simulation time
        - ``slowdown`` -- ratio from simulation time to real time
        """
        self.__sleep_dt = slowdown * dt if slowdown else dt
        self.dt = dt
        self.event = threading.Event()
        self.loop_start = {}
        self.slowdown = slowdown
        self.start_time = time.time()

    @property
    def is_started(self):
        return self.event.isSet()

    def pause(self):
        self.event.clear()

    def resume(self):
        self.event.set()

    def sleep(self, dT=None):
        """
        Delay execution for a duration ``dT`` in simulation time.

        INPUT:

        - ``dT`` -- sleep duration in simulation time
        """
        if dT is None:
            return time.sleep(self.__sleep_dt)
        elif self.slowdown:
            return time.sleep(self.slowdown * dT)
        return time.sleep(dT)

    def start(self):
        self.event.set()

    def step(self, nb_steps=1):
        self.event.set()
        self.event.clear()
        while nb_steps > 1:
            nb_steps -= 1
            self.sleep(self.dt)
            self.event.set()
            self.event.clear()

    def stop(self):
        self.event.clear()

    def sync_loop(self, name):
        """
        Check that the loop of the process identified by ``name`` does not
        execute faster than ``self.dt``.

        INPUT:

        - ``name`` -- identifier of caller thread
        """
        self.event.wait()
        cur_time = self.time()
        if name in self.loop_start:
            sim_elapsed = cur_time - self.loop_start[name]
            if sim_elapsed < self.dt:
                self.sleep(self.dt - sim_elapsed)
        self.loop_start[name] = cur_time

    def time(self):
        if self.slowdown:
            return self.slowdown * (time.time() - self.start_time)
        return time.time() - self.start_time
