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

from tube import COMTube, TubeError
from numpy import array, bmat, dot, eye, hstack, sqrt, vstack, zeros
from pymanoid import PointMass, solve_qp
from scipy.linalg import block_diag
from simulation import Process
from warnings import warn


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

        1)  |p_{nb_steps} - p_goal|^2
        1)  |pd_{nb_steps} - pd_goal|^2
        2)  sum_k |u_k|^2

    Note that this is a weighted (not prioritized) minimization.
    """

    def __init__(self, A, B, C, d, E, f, x_init, x_goal, nb_steps):
        """
        Instantiate a new controller.

        INPUT:

        - ``A`` -- state linear dynamics matrix
        - ``B`` -- control linear dynamics matrix
        - ``C`` -- map from k to H-rep matrix of control constraints
        - ``d`` -- map from k to H-rep vector of control constraints
        - ``E`` -- map from k to H-rep matrix of state constraints
        - ``f`` -- map from k to H-rep vector of state constraints
        - ``x_init`` -- initial state
        - ``x_goal`` -- goal state
        - ``nb_steps`` -- number of discretized time steps
        """
        u_dim = C(0).shape[1]
        x_dim = x_init.shape[0]
        self.A = A
        self.B = B
        self.C = C
        self.E = E
        self.G_state = None
        self.U_dim = u_dim * nb_steps
        self.X_dim = x_dim * nb_steps
        self.d = d
        self.f = f
        self.h_state = None
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
        N = self.nb_steps
        phi = eye(self.x_dim)
        psi = zeros((self.x_dim, self.U_dim))
        G, h = [], []  # list of matrices for inequalities G * x <= h
        for k in xrange(N):
            # x_k = phi * x_init + psi * U
            # p_k = phi[:3] * x_init + psi[:3] * U
            # E * p_k <= f
            # (E * psi[:3]) * U <= f - (E * phi[:3]) * x_init
            G.append(dot(self.E(k), psi[:3]))
            h.append(self.f(k) - dot(dot(self.E(k), phi[:3]), self.x_init))

            # Now we update phi and psi for iteration k + 1
            phi = dot(self.A, phi)
            psi = dot(self.A, psi)
            psi[:, self.u_dim * k:self.u_dim * (k + 1)] = self.B
        self.G_state = vstack(G)
        self.h_state = hstack(h)
        self.phi_last = phi
        self.psi_last = psi

    def compute_control(self):
        if self.psi_last is None:
            self.compute_dynamics()

        A = self.psi_last
        b = self.x_goal - dot(self.phi_last, self.x_init)

        # Cost 1: sum_k u_k^2
        P1 = eye(self.U_dim)
        q1 = zeros(self.U_dim)
        w1 = 1.

        # Cost 2: |p_N - p_goal|^2 = |A[:3] x - b[:3]|^2
        P2 = dot(A[:3].T, A[:3])
        q2 = -dot(b[:3].T, A[:3])
        w2 = 1000.

        # Cost 3: |pd_N - pd_goal|^3 = |A[3:] x - b[3:]|^3
        P3 = dot(A[3:].T, A[3:])
        q3 = -dot(b[3:].T, A[3:])
        w3 = 1000.

        # Weighted combination of all costs
        P = w1 * P1 + w2 * P2 + w3 * P3
        q = w1 * q1 + w2 * q2 + w3 * q3

        # Inequality constraints
        G_control = block_diag(*[self.C(k) for k in xrange(self.nb_steps)])
        h_control = hstack([self.d(k) for k in xrange(self.nb_steps)])
        G = vstack([G_control, self.G_state])
        h = hstack([h_control, self.h_state])
        G = G_control
        h = h_control
        # G = self.G_state
        # h = self.h_state

        t0 = time.time()
        self.U = solve_qp(P, q, G, h)
        print "Solved QP in %.2f ms" % (1000. * (time.time() - t0))
        print "End error (position):", norm(
            dot(self.phi_last[:3], self.x_init) +
            dot(self.psi_last[:3], self.U) -
            self.x_goal[:3])
        print "End error (velocity):", norm(
            dot(self.phi_last[3:], self.x_init) +
            dot(self.psi_last[3:], self.U) -
            self.x_goal[3:])


class COMPreviewControl(PreviewControl):

    def __init__(self, com_init, comd_init, com_goal, comd_goal, tube,
                 duration, switch_time, nb_steps):
        dT = duration / nb_steps
        I = eye(3)
        A = array(bmat([[I, dT * I], [zeros((3, 3)), I]]))
        B = array(bmat([[.5 * dT ** 2 * I], [dT * I]]))
        x_init = hstack([com_init, comd_init])
        x_goal = hstack([com_goal, comd_goal])
        switch_step = int(switch_time / dT)
        C, d, E, f = self.compute_inequalities(tube, switch_step, nb_steps)
        super(COMPreviewControl, self).__init__(
            A, B, C, d, E, f, x_init, x_goal, nb_steps)
        self.duration = duration
        self.switch_step = switch_step
        self.timestep = dT

    def compute_inequalities(self, tube, switch_step, nb_steps):
        def multiplex_matrices(M1, M2, switch_step):
            def M(k):
                return M1 if k <= switch_step else M2
            return M

        def wrap_matrix(M1):
            def M(k):
                return M1
            return M

        C1, d1 = tube.compute_dual_hrep(stance_id=0)
        E1, f1 = tube.compute_primal_hrep(stance_id=0)
        if switch_step >= nb_steps - 1:
            C = wrap_matrix(C1)
            d = wrap_matrix(d1)
            E = wrap_matrix(E1)
            f = wrap_matrix(f1)
        else:
            C2, d2 = tube.compute_dual_hrep(stance_id=1)
            E2, f2 = tube.compute_primal_hrep(stance_id=1)
            C = multiplex_matrices(C1, C2, switch_step)
            d = multiplex_matrices(d1, d2, switch_step)
            E = multiplex_matrices(E1, E2, switch_step)
            f = multiplex_matrices(f1, f2, switch_step)
        return (C, d, E, f)


class TubePreviewControl(Process):

    def __init__(self, com, fsm, preview_buffer, nb_mpc_steps, tube_shape,
                 tube_radius):
        """
        Create a new feedback controller that continuously runs the preview
        controller and sends outputs to a COMAccelBuffer.

        INPUT:

        - ``com`` -- PointMass containing current COM state
        - ``fsm`` -- instance of finite state machine
        - ``preview_buffer`` -- PreviewBuffer to send MPC outputs to
        - ``nb_mpc_steps`` -- discretization step of the preview window
        - ``tube_shape`` -- number of vertices of the COM trajectory tube
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
        self.tube_shape = tube_shape

    def on_tick(self, sim):
        cur_com = self.com.p
        cur_comd = self.com.pd
        cur_stance = self.fsm.cur_stance
        next_stance = self.fsm.next_stance
        preview_targets = self.fsm.get_preview_targets()
        switch_time, horizon, target_com, target_comd = preview_targets

        target_moved = norm(target_com - self.target_box.p) > 1e-3
        phase_switched = self.fsm.phase_id > self.last_phase_id
        self.target_box.set_pos(target_com)
        self.last_phase_id = self.fsm.phase_id
        if True or not self.tube or target_moved or phase_switched or \
                not self.tube.contains(cur_com):
            print "Recomputing tube..."
            self.tube = COMTube(
                cur_com, target_com, cur_stance, next_stance, self.tube_shape,
                self.tube_radius)
        else:
            print "Keeping current tube"
        if True:
            print "\nVelocities:"
            print "- |cur_comd| =", norm(cur_comd)
            print "- |target_comd| =", norm(target_comd)
            print "\nTime:"
            print "- horizon =", horizon
            print "- switch_time =", switch_time
            print "- timestep = ", horizon / self.nb_mpc_steps
            print""
        try:
            preview_control = COMPreviewControl(
                cur_com, cur_comd, target_com, target_comd, self.tube, horizon,
                switch_time, self.nb_mpc_steps)
        except TubeError as e:
            print "Tube error: %s" % str(e)
            return
        preview_control.compute_dynamics()
        try:
            preview_control.compute_control()
            self.preview_buffer.update_preview(preview_control)
        except ValueError as e:
            warn("MPC: couldn't solve QP, maybe inconsistent constraints?")
            print "Exception:", e
            sim.stop()
