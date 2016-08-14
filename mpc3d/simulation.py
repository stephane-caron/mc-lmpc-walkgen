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

import time

from threading import Lock, Thread


class Simulation(object):

    def __init__(self, dt):
        """
        Create a new simulation object.

        INPUT:

        - ``dt`` -- time interval between two ticks in simulation time
        """
        self.callbacks = []
        self.dt = dt
        self.is_running = False
        self.tick_time = 0
        self.state_lock = Lock()

    def __del__(self):
        """Close thread at shutdown."""
        self.stop()

    def add_callback(self, callback):
        """Add a function called after each tick (insertion order matters)."""
        self.callbacks.append(callback)

    def run_thread(self):
        """Run simulation thread."""
        while self.is_running:
            self.step()

    def start(self):
        """Start simulation thread. """
        self.is_running = True
        self.thread = Thread(target=self.run_thread, args=())
        self.thread.start()

    def step(self, n=1):
        """Perform one simulation step."""
        for _ in xrange(n):
            t0 = time.time()
            with self.state_lock:
                for callback in self.callbacks:
                    callback(self)
            rem_time = self.dt - (time.time() - t0)
            if rem_time > 1e-4:
                time.sleep(rem_time)
            elif rem_time < -1e-4:
                print "Time budget exhausted by %.1f ms" % (-1000. * rem_time)
            self.tick_time += 1

    def stop(self):
        self.is_running = False
