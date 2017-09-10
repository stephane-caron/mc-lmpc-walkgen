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

import os
import sys
import time

from logging import warning
from numpy import array, bmat, dot, eye, hstack, sqrt, vstack, zeros
from scipy.linalg import block_diag

try:  # use local pymanoid submodule
    script_path = os.path.realpath(__file__)
    sys.path = [os.path.dirname(script_path) + '/../pymanoid'] + sys.path
    from pymanoid import PointMass, solve_qp
except:  # this is to avoid warning E402 from Pylint
    pass

from simulation import Process
from tube import COMTube, TubeError


def norm(v):
    return sqrt(dot(v, v))


class PreviewControl(object):

    """
    Preview control for a system with linear dynamics:

        x_{k+1} = A * x_k + B * u_k

    where x is assumed to be the state of a configuration variable p, i.e.,

        x_k = [  p_k ]
              [ pd_k ]

    subject to constraints:

        x_0 = x_init                    -- initial state
        for all k,   C(k) * u_k <= d(k) -- control constraints
        for all k,   E(k) * p_k <= f(k) -- position constraints

    The output control law will minimize, by decreasing priority:

        1)  |x_{nb_steps} - x_goal|^2
        2)  sum_k |u_k|^2

    Note that this is a weighted (not prioritized) minimization.
    """

    def __init__(self, A, B, G, h, x_init, x_goal, nb_steps, E=None, f=None):
        """
        Instantiate a new controller.

        INPUT:

        - ``A`` -- state linear dynamics matrix
        - ``B`` -- control linear dynamics matrix
        - ``G`` -- matrix for control inequality constraints
        - ``h`` -- vector for control inequality constraints
        - ``x_init`` -- initial state
        - ``x_goal`` -- goal state
        - ``nb_steps`` -- number of discretized time steps
        - ``E`` -- (optional) matrix for state inequality constraints
        - ``f`` -- (optional) vector for state inequality constraints
        """
        u_dim = B.shape[1]
        x_dim = A.shape[1]
        self.A = A
        self.B = B
        self.E = E
        self.G = G
        self.U_dim = u_dim * nb_steps
        self.X_dim = x_dim * nb_steps  # not used but meh
        self.f = f
        self.h = h
        self.nb_steps = nb_steps
        self.phi_last = None
        self.psi_last = None
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.x_goal = x_goal
        self.x_init = x_init

    def compute_dynamics(self):
        """
        Blah:

            x_1 =     A' * x_0 +       B'  * u_0
            x_2 = (A'^2) * x_0 + (A' * B') * u_0 + B' * u_1
            ...

        Second, rewrite future sxstem dxnamics as:

            X = Phi * x_0 + Psi * U

            U = [u_0 ... u_{N-1}]
            X = [x_0 ... x_{N-1}]

            x_k = phi[k] * x_0 + psi[k] * U
            x_N = phi_last * x_0 + psi_last * U

        """
        phi = eye(self.x_dim)
        psi = zeros((self.x_dim, self.U_dim))
        G_list, h_list = [], []
        for k in xrange(self.nb_steps):
            # x_k = phi * x_init + psi * U
            # p_k = phi[:3] * x_init + psi[:3] * U
            # E * p_k <= f
            # (E * psi[:3]) * U <= f - (E * phi[:3]) * x_init
            if self.E is not None:
                G_list.append(dot(self.E, psi[:3]))
                h_list.append(self.f - dot(dot(self.E, phi[:3]), self.x_init))
            phi = dot(self.A, phi)
            psi = dot(self.A, psi)
            psi[:, self.u_dim * k:self.u_dim * (k + 1)] = self.B
        self.G_state = G_list
        self.h_state = h_list
        self.phi_last = phi
        self.psi_last = psi

    def compute_control(self):
        assert self.psi_last is not None, "Call compute_dynamics() first"

        # Cost 1: sum_k u_k^2
        P1 = eye(self.U_dim)
        q1 = zeros(self.U_dim)
        w1 = 1.

        # Cost 2: |x_N - x_goal|^2 = |A * x - b|^2
        A = self.psi_last
        b = self.x_goal - dot(self.phi_last, self.x_init)
        P2 = dot(A.T, A)
        q2 = -dot(b.T, A)
        w2 = 1000.

        # Weighted combination of all costs
        P = w1 * P1 + w2 * P2
        q = w1 * q1 + w2 * q2

        # Inequality constraints
        t1 = time.time()
        if self.E is not None:
            G = vstack([self.G] + self.G_state)
            h = hstack([self.h] + self.h_state)
            self.U = solve_qp(P, q, G, h)
        else:
            self.U = solve_qp(P, q, self.G, self.h)
        self.comp_times = [t1 - self.t0, time.time() - t1]


class COMPreviewControl(PreviewControl):

    def __init__(self, com_init, comd_init, com_goal, comd_goal, tube,
                 duration, switch_time, nb_steps, state_constraints=False):
        self.t0 = time.time()
        dT = duration / nb_steps
        I = eye(3)
        A = array(bmat([[I, dT * I], [zeros((3, 3)), I]]))
        B = array(bmat([[.5 * dT ** 2 * I], [dT * I]]))
        x_init = hstack([com_init, comd_init])
        x_goal = hstack([com_goal, comd_goal])
        switch_step = int(switch_time / dT)
        G_list = []
        h_list = []
        C1, d1 = tube.dual_hrep[0]
        E, f = None, None
        if state_constraints:
            E, f = tube.full_hrep
        if 0 <= switch_step < nb_steps - 1:
            C2, d2 = tube.dual_hrep[1]
        for k in xrange(nb_steps):
            if k <= switch_step:
                G_list.append(C1)
                h_list.append(d1)
            else:  # k > switch_step
                G_list.append(C2)
                h_list.append(d2)
        G = block_diag(*G_list)
        h = hstack(h_list)
        super(COMPreviewControl, self).__init__(
            A, B, G, h, x_init, x_goal, nb_steps, E, f)
        self.switch_step = switch_step
        self.timestep = dT


class TubePreviewControl(Process):

    def __init__(self, com, fsm, preview_buffer, nb_mpc_steps, tube_radius):
        """
        Create a new feedback controller that continuously runs the preview
        controller and sends outputs to a COMAccelBuffer.

        INPUT:

        - ``com`` -- PointMass containing current COM state
        - ``fsm`` -- instance of finite state machine
        - ``preview_buffer`` -- PreviewBuffer to send MPC outputs to
        - ``nb_mpc_steps`` -- discretization step of the preview window
        - ``tube_radius`` -- tube radius (in L1 norm)
        """
        self.com = com
        self.fsm = fsm
        self.last_phase_id = -1
        self.nb_mpc_steps = nb_mpc_steps
        self.preview_buffer = preview_buffer
        self.target_box = PointMass(fsm.cur_stance.com, 30., color='g')
        self.thread = None
        self.thread_lock = None
        self.tube = None
        self.tube_radius = tube_radius
        self.verbose = False

    def on_tick(self, sim):
        cur_com = self.com.p
        cur_comd = self.com.pd
        cur_stance = self.fsm.cur_stance
        next_stance = self.fsm.next_stance
        preview_targets = self.fsm.get_preview_targets()
        switch_time, horizon, target_com, target_comd = preview_targets
        if self.verbose:
            print "\nVelocities:"
            print "- |cur_comd| =", norm(cur_comd)
            print "- |target_comd| =", norm(target_comd)
            print "\nTime:"
            print "- horizon =", horizon
            print "- switch_time =", switch_time
            print "- timestep = ", horizon / self.nb_mpc_steps
            print""
        self.target_box.set_pos(target_com)
        try:
            self.tube = COMTube(
                cur_com, target_com, cur_stance, next_stance, self.tube_radius)
        except TubeError as e:
            warning("Tube error: %s" % str(e))
            return
        preview_control = COMPreviewControl(
            cur_com, cur_comd, target_com, target_comd, self.tube, horizon,
            switch_time, self.nb_mpc_steps)
        preview_control.compute_dynamics()
        try:
            preview_control.compute_control()
            self.preview_buffer.update_preview(preview_control)
        except ValueError as e:
            warning("MPC couldn't solve QP, constraints may be inconsistent")
            return
        sim.report_comp_times({
            'tube_primal_vrep': self.tube.comp_times[0],
            'tube_primal_hrep': self.tube.comp_times[1],
            'tube_dual_vrep': self.tube.comp_times[2],
            'tube_dual_hrep': self.tube.comp_times[3],
            'qp_build': preview_control.comp_times[0],
            'qp_solve': preview_control.comp_times[1]
        })
