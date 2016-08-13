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

from numpy import zeros
from pymanoid import draw_line
from threading import Lock, Thread


class COMAccelBuffer(object):

    """
    These buffers store COM acceleration trajectories output by the preview
    controller and execute them until the next update.
    """

    def __init__(self, com, fsm):
        self.com = com
        self.com_traj = []
        self.com_traj_handles = []
        self.comd = zeros(3)
        self.comdd = zeros(3)
        self.comdd_index = 0
        self.comdd_lock = Lock()
        self.comdd_traj = []
        self.comdd_vector = None
        self.fsm = fsm
        self.thread = None
        self.thread_lock = Lock()
        self.timestep = 0.

    @property
    def cur_height(self):
        return self.com.z

    def update_control(self, mpc):
        with self.comdd_lock:
            self.comdd_index = 0
            self.comdd_vector = mpc.U
            self.timestep = mpc.timestep

    def get_next_acceleration(self):
        with self.comdd_lock:
            if self.comdd_vector is None:
                comdd = zeros(3)
            else:
                j = 3 * self.comdd_index
                comdd = self.comdd_vector[j:j + 3]
                if comdd.shape[0] == 0:
                    comdd = zeros(3)
                    self.comdd_vector = None
            self.comdd_index += 1
            return (comdd, self.timestep)

    def start_thread(self, callback):
        self.thread = Thread(
            target=self.run_thread, args=(callback,))
        self.thread.daemon = True
        self.thread.start()

    def pause_thread(self):
        self.thread_lock.acquire()

    def resume_thread(self):
        self.thread_lock.release()

    def stop_thread(self):
        self.thread_lock = None

    def run_thread(self, callback):
        comdd = zeros(3)
        while self.thread_lock:
            with self.thread_lock:
                t0 = time.time()
                (comdd, dT) = self.get_next_acceleration()
                prev_com = self.com.p
                self.com.set_pos(
                    prev_com + self.comd * dT + comdd * .5 * dT ** 2)
                self.comd += comdd * dT
                self.com_traj.append(self.com.p)
                self.com_traj_handles.append(
                    draw_line(prev_com, self.com.p, color='b', linewidth=3))
                self.comdd = comdd
                self.comdd_traj.append(comdd)
                callback(comdd)
                time.sleep(dT - time.time() + t0)
