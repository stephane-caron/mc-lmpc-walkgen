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

import pymanoid

from numpy import arange, hstack
from pymanoid import draw_line
from pymanoid.rotations import quat_slerp
from pymanoid.rotations import rotation_matrix_from_quat
from stance import Stance
from threading import Lock, Thread


def pose_interp(pose0, pose1, t):
    """Linear pose interpolation."""
    pos = pose0[4:] + t * (pose1[4:] - pose0[4:])
    quat = quat_slerp(pose0[:4], pose1[:4], t)
    return hstack([quat, pos])


class FreeLimb(pymanoid.Contact):

    def __init__(self, **kwargs):
        super(FreeLimb, self).__init__(X=0.2, Y=0.1, **kwargs)
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


class StanceFSM(object):

    transitions = {
        'DS-L': 'SS-L',
        'SS-L': 'DS-R',
        'DS-R': 'SS-R',
        'SS-R': 'DS-L'
    }

    def __init__(self, contacts, com, init_phase, ss_duration, ds_duration,
                 init_com_offset=None, cyclic=False):
        """
        Create a new finite state machine.

        INPUT:

        - ``contacts`` -- sequence of contacts
        - ``com`` -- PointMass object giving the current position of the COM
        - ``init_phase`` -- string giving the initial FSM state
        - ``ss_duration`` -- duration of single-support phases
        - ``ds_duration`` -- duration of double-support phases

        .. NOTE::

            This function updates the position of ``com`` as a side effect.
        """
        assert init_phase in ['DS-L', 'DS-R']  # kron
        first_stance = Stance(init_phase, contacts[0], contacts[1])
        if init_com_offset is not None:
            first_stance.com += init_com_offset
        com.set_pos(first_stance.com)
        self._next_stance = None
        self.com = com
        self.contacts = contacts
        self.cur_phase = init_phase
        self.cur_stance = first_stance
        self.cyclic = cyclic
        self.ds_duration = ds_duration
        self.free_foot = FreeLimb(visible=False, color='c')
        self.left_foot_traj_handles = []
        self.nb_contacts = len(contacts)
        self.next_contact_id = 2 if init_phase == 'DS-R' else 3  # kroooon
        self.rem_time = 0.
        self.right_foot_traj_handles = []
        self.ss_duration = ss_duration
        self.state_time = 0.
        self.thread = None
        self.thread_lock = None

    def start_thread(self, sim, callback):
        """
        Start FSM thread.

        INPUT:

        - ``sim`` -- a Simulation object
        - ``callback`` -- function called after each phase transition
        """
        self.thread_lock = Lock()
        self.thread = Thread(
            target=self.run_thread, args=(sim, callback,))
        self.thread.daemon = True
        self.thread.start()

    def pause_thread(self):
        self.thread_lock.acquire()

    def resume_thread(self):
        self.thread_lock.release()

    def stop_thread(self):
        self.thread_lock = None

    def run_thread(self, sim, callback):
        """
        Run the FSM thread.

        INPUT:

        - ``sim`` -- a Simulation object
        - ``callback`` -- function called after each phase transition
        """
        record_foot_traj = True
        while self.thread_lock:
            with self.thread_lock:
                phase_duration = \
                    self.ds_duration if self.cur_stance.is_double_support \
                    else self.ss_duration
                self.rem_time = phase_duration
                for t in arange(0., phase_duration, sim.dt):
                    self.state_time = t
                    if self.cur_stance.is_single_support:
                        progress = self.state_time / phase_duration  # in [0, 1]
                        prev_pos = self.free_foot.p
                        self.free_foot.update_pose(progress)
                        if record_foot_traj:
                            if self.cur_stance.left_foot:
                                self.right_foot_traj_handles.append(
                                    draw_line(prev_pos, self.free_foot.p,
                                              color='r', linewidth=3))
                            else:
                                self.left_foot_traj_handles.append(
                                    draw_line(prev_pos, self.free_foot.p,
                                              color='g', linewidth=3))
                    sim.sleep()
                    self.rem_time -= sim.dt
                if self.cur_stance.is_double_support:
                    next_stance = self.next_stance

                    def is_inside_next_com_polygon(p):
                        return next_stance.is_inside_static_equ_polygon(p, 39.)

                    while not is_inside_next_com_polygon(self.com.p):
                        sim.sleep()
                self.step()
                callback()
                sim.sync()

    @property
    def next_contact(self):
        return self.contacts[self.next_contact_id]

    @property
    def next_duration(self):
        if self.next_phase.startswith('SS'):
            return self.ss_duration
        return self.ds_duration

    @property
    def next_stance(self):
        if self._next_stance is not None:
            # internal save to avoid useless CWC/SEP recomputations
            return self._next_stance
        if self.next_phase == 'SS-L':
            left_foot = self.cur_stance.left_foot
            right_foot = None
        elif self.next_phase == 'DS-R':
            left_foot = self.cur_stance.left_foot
            right_foot = self.next_contact
        elif self.next_phase == 'SS-R':
            left_foot = None
            right_foot = self.cur_stance.right_foot
        elif self.next_phase == 'DS-L':
            left_foot = self.next_contact
            right_foot = self.cur_stance.right_foot
        else:  # should not happen
            assert False, "Unknown state: %s" % self.next_phase
        self._next_stance = Stance(self.next_phase, left_foot, right_foot)
        return self._next_stance

    # @property
    # def next_ss_stance(self):
    #     assert self.cur_stance.is_single_support
    #     t = self.transitions
    #     if self.cur_stance.left_foot is None:
    #         return Stance(t[t[self.cur_phase]], self.next_contact, None)
    #     else:  # self.cur_stance.right_foot is None
    #         return Stance(t[t[self.cur_phase]], None, self.next_contact)

    @property
    def next_phase(self):
        return self.transitions[self.cur_phase]

    def get_time_to_transition(self):
        return self.rem_time

    def step(self):
        next_stance = self.next_stance
        next_phase = self.next_phase
        if next_phase.startswith('DS'):
            self.next_contact_id += 1
            if self.next_contact_id >= self.nb_contacts:
                if self.cyclic:
                    self.next_contact_id -= self.nb_contacts
                elif self.thread_lock:  # thread is running
                    self.stop_thread()
        self._next_stance = None
        self.cur_phase = next_phase
        self.cur_stance = next_stance

    def get_preview_targets(self):
        time_to_transition = self.rem_time
        if self.cur_stance.is_single_support \
                and time_to_transition < 0.5 * self.ss_duration:
            horizon = time_to_transition \
                + self.ds_duration \
                + 0.5 * self.ss_duration
            com_target = self.next_stance.com
        elif self.cur_stance.is_double_support:
            horizon = time_to_transition + 0.5 * self.ss_duration
            com_target = self.cur_stance.com
        else:  # single support with plenty of time ahead
            horizon = time_to_transition
            com_target = self.cur_stance.com
        return (time_to_transition, horizon, com_target)
