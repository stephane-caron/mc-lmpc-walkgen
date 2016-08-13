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
from pymanoid import draw_line
from threading import Lock, Thread


class COMAccelBuffer(object):

    """
    These buffers store COM accelerations output by the preview controller and
    execute them until the next update.
    """

    def __init__(self, com, fsm, show_past=False, show_preview=True):
        self.com = com
        self.comd = zeros(3)
        self.comdd = zeros(3)
        self.preview_index = 0
        self.preview_lock = Lock()
        self.comdd_vector = None
        self.fsm = fsm
        self.preview = None
        self.past_handles = [] if show_past else None
        self.preview_handles = [] if show_preview else None
        self.thread = None
        self.thread_lock = Lock()

    def update_preview(self, preview):
        with self.preview_lock:
            self.preview_index = 0
            self.preview = preview
            if self.preview_handles is not None:
                self.draw_preview()

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

    def start_thread(self, sim):
        self.thread = Thread(target=self.run_thread, args=(sim,))
        self.thread.daemon = True
        self.thread.start()

    def pause_thread(self):
        self.thread_lock.acquire()

    def resume_thread(self):
        self.thread_lock.release()

    def stop_thread(self):
        self.thread_lock = None

    def run_thread(self, sim):
        comdd = zeros(3)
        while self.thread_lock:
            with self.thread_lock:
                sim.sync_loop('com_buffer')
                (comdd, dT) = self.get_next_preview_window()
                self.euler_integrate(comdd, dT)
                sim.sleep(dT)

    def euler_integrate(self, comdd, dt):
        com0 = self.com.p
        self.com.set_pos(com0 + self.comd * dt + comdd * .5 * dt ** 2)
        self.comd += comdd * dt
        self.comdd = comdd
        if self.past_handles is not None:
            self.past_handles.append(
                draw_line(com0, self.com.p, color='b', linewidth=3))

    def clear_preview(self):
        self.preview_handles = []

    def draw_preview(self):
        com = self.com.p
        comd = self.comd
        dT = self.preview.timestep
        for preview_index in xrange(len(self.preview.U) / 3):
            com0 = com
            j = 3 * preview_index
            comdd = self.preview.U[j:j + 3]
            com = com + comd * dT + comdd * .5 * dT ** 2
            comd += comdd * dT
            color = 'b' if preview_index <= self.preview.switch_step else 'r'
            self.preview_handles.append(
                draw_line(com0, com, color=color, linewidth=3))
