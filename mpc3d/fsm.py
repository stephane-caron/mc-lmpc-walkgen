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


from free_foot import FreeFoot
from pymanoid import draw_line
from stance import Stance


class StateMachine(object):

    transitions = {
        'DS-L': 'SS-L',
        'SS-L': 'DS-R',
        'DS-R': 'SS-R',
        'SS-R': 'DS-L'
    }

    def __init__(self, contacts, com, init_phase, ss_duration, ds_duration,
                 init_com_offset=None, cyclic=False, callback=None):
        """
        Create a new finite state machine.

        INPUT:

        - ``contacts`` -- sequence of contacts
        - ``com`` -- PointMass object giving the current position of the COM
        - ``init_phase`` -- string giving the initial FSM state
        - ``ss_duration`` -- duration of single-support phases
        - ``ds_duration`` -- duration of double-support phases
        - ``init_com_offset`` -- used for initialization only
        - ``cyclic`` -- if True, contact sequence is looped over
        - ``callback`` -- (optional) function called after phase transitions

        .. NOTE::

            This function updates the position of ``com`` as a side effect.

        .. NOTE::

            Assumes that the motion starts at the end of a DS phase.
        """
        assert init_phase in ['DS-L', 'DS-R']  # kron
        first_stance = Stance(init_phase, contacts[0], contacts[1])
        if init_com_offset is not None:
            first_stance.com += init_com_offset
        com.set_pos(first_stance.com)
        self._next_stance = None
        self.callback = callback
        self.com = com
        self.contacts = contacts
        self.cur_duration = ds_duration  # start on DS for now
        self.cur_phase = init_phase
        self.cur_stance = first_stance
        self.cyclic = cyclic
        self.ds_duration = ds_duration
        self.free_foot = FreeFoot(visible=False, color='c')
        self.is_not_over = True
        self.left_foot_traj_handles = []
        self.nb_contacts = len(contacts)
        self.next_contact_id = 2 if init_phase == 'DS-R' else 3  # kroooon
        self.rem_time = 0.  # initial state is assumed at end of DS phase
        self.right_foot_traj_handles = []
        self.ss_duration = ss_duration
        self.thread = None
        self.thread_lock = None

    @property
    def next_contact(self):
        return self.contacts[self.next_contact_id]

    @property
    def next_phase(self):
        return self.transitions[self.cur_phase]

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

    def on_tick(self, sim):
        """
        Update the FSM state after a tick of the control loop.

        INPUT:

        - ``sim`` -- instance of current simulation
        """
        def can_switch_to_ss():
            m, next_stance = 39., self.next_stance
            return next_stance.is_inside_static_equ_polygon(self.com.p, m)

        if self.rem_time > 0.:
            if self.cur_stance.is_single_support:
                self.update_free_foot()
            self.rem_time -= sim.dt
        elif self.cur_stance.is_double_support and not can_switch_to_ss():
            print "COM is not ready for next SS yet"
        else:
            self.step()

    def step(self):
        next_stance = self.next_stance
        next_phase = self.next_phase
        if next_phase.startswith('DS'):
            self.next_contact_id += 1
            if self.next_contact_id >= self.nb_contacts:
                if self.cyclic:
                    self.next_contact_id -= self.nb_contacts
                else:
                    self.is_not_over = False
        else:  # next_phase.startswith('SS')
            next_pose = self.next_contact.pose
            if next_stance.left_foot is None:
                self.free_foot.reset(self.cur_stance.left_foot.pose, next_pose)
            else:  # next_stance.right_foot is None
                self.free_foot.reset(self.cur_stance.right_foot.pose, next_pose)
        self._next_stance = None
        self.cur_phase = next_phase
        self.cur_stance = next_stance
        self.cur_duration = \
            self.ds_duration if self.cur_stance.is_double_support \
            else self.ss_duration
        self.rem_time = self.cur_duration
        if self.callback is not None:
            self.callback()

    def update_free_foot(self):
        progress = 1. - self.rem_time / self.cur_duration
        prev_pos = self.free_foot.p
        self.free_foot.update_pose(progress)
        if self.cur_stance.left_foot:
            self.right_foot_traj_handles.append(
                draw_line(prev_pos, self.free_foot.p, color='r', linewidth=3))
        else:
            self.left_foot_traj_handles.append(
                draw_line(prev_pos, self.free_foot.p, color='g', linewidth=3))

    def get_preview_targets(self):
        if self.cur_stance.is_single_support \
                and self.rem_time < 0.5 * self.ss_duration:
            horizon = self.rem_time \
                + self.ds_duration \
                + 0.5 * self.ss_duration
            com_target = self.next_stance.com
        elif self.cur_stance.is_double_support:
            horizon = self.rem_time + 0.5 * self.ss_duration
            com_target = self.cur_stance.com
        else:  # single support with plenty of time ahead
            horizon = self.rem_time
            com_target = self.cur_stance.com
        return (self.rem_time, horizon, com_target)
