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

import pylab
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

    def plot_com_traj(self):
        pylab.ion()
        pylab.clf()
        x_traj, y_traj, z_traj = zip(*self.com_traj)
        pylab.plot(x_traj, color='r')
        pylab.plot(y_traj, color='g')
        pylab.plot(z_traj, color='b')
        pylab.legend(('$\ddot{x}_G$', '$\ddot{y}_G$', '$\ddot{z}_G$'))

    def plot_comdd_traj(self):
        pylab.ion()
        pylab.clf()
        pylab.plot(self.comdd_traj)

    def plot_last_profiles(self, com, comd):
        """
        Plot preview positions.

        com -- current COM position
        comd -- current COM velocity
        """
        x_list, xd_list, xdd_list = [], [], []
        y_list, yd_list, ydd_list = [], [], []
        z_list, zd_list, zdd_list = [], [], []
        nb_steps = len(self.comdd_vector) / 3
        for i in xrange(nb_steps):
            comdd = self.comdd_vector[3 * i:3 * (i + 1)]
            # com_list.append(com)
            x_list.append(com[0])
            y_list.append(com[1])
            z_list.append(com[2])
            # comd_list.append(comd)
            xd_list.append(comd[0])
            yd_list.append(comd[1])
            zd_list.append(comd[2])
            # comdd_list.append(comdd)
            xdd_list.append(comdd[0])
            ydd_list.append(comdd[1])
            zdd_list.append(comdd[2])
            com += comd * self.timestep + .5 * comdd * self.timestep ** 2
            comd += comdd * self.timestep

        pylab.ion()
        pylab.clf()
        pylab.subplot(331)
        pylab.plot(xdd_list, 'r-', label='$\ddot{x}_G$')
        pylab.legend()
        pylab.subplot(332)
        pylab.plot(ydd_list, 'g-', label='$\ddot{y}_G$')
        pylab.legend()
        pylab.subplot(333)
        pylab.plot(zdd_list, 'b-', label='$\ddot{z}_G$')
        pylab.legend()
        pylab.subplot(334)
        pylab.plot(xd_list, 'r-', label='$\dot{x}_G$')
        pylab.legend()
        pylab.subplot(335)
        pylab.plot(yd_list, 'g-', label='$\dot{y}_G$')
        pylab.legend()
        pylab.subplot(336)
        pylab.plot(zd_list, 'b-', label='$\dot{z}_G$')
        pylab.legend()
        pylab.subplot(337)
        pylab.plot(x_list, 'r-', label='$x_G$')
        pylab.legend()
        pylab.subplot(338)
        pylab.plot(y_list, 'g-', label='$y_G$')
        pylab.legend()
        pylab.subplot(339)
        pylab.plot(z_list, 'b-', label='$z_G$')
        pylab.legend()
