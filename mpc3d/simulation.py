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

from stats import AvgStdEstimator
from threading import Thread


class Process(object):

    def on_tick(self, sim):
        """Function called by the Simulation parent after each clock tick."""
        pass


class Simulation(object):

    def __init__(self, dt):
        """
        Create a new simulation object.

        INPUT:

        - ``dt`` -- time interval between two ticks in simulation time
        - ``profile`` -- when True, reports computation times for each Process
        """
        self.comp_times = {}
        self.dt = dt
        self.extras = []
        self.is_running = False
        self.processes = []
        self.tick_time = 0

    def __del__(self):
        """Close thread at shutdown."""
        self.stop()

    def run_thread(self):
        """Run simulation thread."""
        while self.is_running:
            self.step()

    def schedule(self, process):
        """Add a Process to the schedule list (insertion order matters)."""
        self.processes.append(process)

    def schedule_extra(self, process):
        """Schedule a Process not counted in the computation time budget."""
        self.extras.append(process)

    def start(self):
        """Start simulation thread. """
        self.is_running = True
        self.thread = Thread(target=self.run_thread, args=())
        self.thread.daemon = True
        self.thread.start()

    def step(self, n=1):
        """Perform one simulation step."""
        for _ in xrange(n):
            t0 = time.time()
            for process in self.processes:
                process.on_tick(self)
            rem_time = self.dt - (time.time() - t0)
            if rem_time < -1e-4:
                print "Time time exhausted by %.1f ms" % (-1000. * rem_time)
            if self.extras:
                for process in self.extras:
                    process.on_tick(self)
                rem_time = self.dt - (time.time() - t0)
            if rem_time > 1e-4:
                time.sleep(rem_time)
            self.tick_time += 1

    def stop(self):
        self.is_running = False

    def report_comp_times(self, d):
        for (key, value) in d.iteritems():
            if key not in self.comp_times:
                self.comp_times[key] = AvgStdEstimator()
            self.comp_times[key].add(value)

    def print_comp_times(self):
        total_avg, total_std = 0., 0.
        for (key, estimator) in self.comp_times.iteritems():
            avg, std, n = estimator.get_all()
            avg *= 1000  # [ms]
            std *= 1000  # [ms]
            print "%20s: %.1f ms +/- %.1f ms over %5d items" % (
                key, avg, std, n)
            total_avg += avg
            total_std += std
        print "%20s  ----------------------------------" % ''
        print "%20s: %.1f ms +/- %.1f ms" % ("total", total_avg, total_std)
