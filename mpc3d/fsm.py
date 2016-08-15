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
from simulation import Process
from stance import Stance


class StateMachine(Process):

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
        self.nb_contacts = len(contacts)
        self.next_contact_id = 2 if init_phase == 'DS-R' else 3  # kroooon
        self.next_stance = None  # initialized below
        self.phase_id = -1  # 0 will be the first SS phase
        self.rem_time = 0.  # initial state is assumed at end of DS phase
        self.ss_duration = ss_duration
        self.thread = None
        self.thread_lock = None
        #
        self.next_stance = self.compute_next_stance()

    @property
    def next_contact(self):
        return self.contacts[self.next_contact_id]

    @property
    def next_phase(self):
        return self.transitions[self.cur_phase]

    def compute_next_stance(self):
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
        return Stance(self.next_phase, left_foot, right_foot)

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
                progress = 1. - self.rem_time / self.cur_duration
                self.free_foot.update_pose(progress)
            self.rem_time -= sim.dt
        elif (self.cur_stance.is_double_support and
              self.next_stance is not None and not can_switch_to_ss()):
            print "\n\nCOM is not ready for next SS yet\n\n"
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
        self.cur_phase = next_phase
        self.cur_stance = next_stance
        self.next_stance = self.compute_next_stance()
        self.cur_duration = \
            self.ds_duration if self.cur_stance.is_double_support \
            else self.ss_duration
        self.rem_time = self.cur_duration
        self.phase_id += 1
        if self.callback is not None:
            self.callback()

    def get_preview_targets(self):
        if self.cur_stance.is_single_support \
                and self.rem_time < 0.5 * self.ss_duration:
            horizon = self.rem_time \
                + self.ds_duration \
                + 0.5 * self.ss_duration
            target_com = self.next_stance.com
            target_comd = (self.next_stance.com - self.cur_stance.com) / horizon
            # target_comd = self.next_stance.comd
        elif self.cur_stance.is_double_support:
            horizon = self.rem_time + 0.5 * self.ss_duration
            target_com = self.cur_stance.com
            target_comd = self.next_stance.comd
        else:  # single support with plenty of time ahead
            horizon = self.rem_time
            target_com = self.cur_stance.com
            target_comd = self.cur_stance.comd
        return (self.rem_time, horizon, target_com, target_comd)
