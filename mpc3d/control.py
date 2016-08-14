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

from tube import COMTube, TubeError
from numpy import array, bmat, dot, eye, hstack, sqrt, vstack, zeros
from pymanoid import PointMass, solve_qp
from scipy.linalg import block_diag
from warnings import warn


def norm(v):
    return sqrt(dot(v, v))


class PreviewControl(object):

    """
    Preview control for a system with linear dynamics:

        x_{k+1} = A * x_k + B * u_k

    subject to constraints:

        x_0 = x_init
        for all k,   C(k) * u_k <= d(k)
        for all k,   E(k) * x_k <= f(k)

    The output control law will minimize, by decreasing priority:

        1)  |x_{nb_steps} - x_goal|^2
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
            # Here: x_k = phi * x_init + psi * U
            # State inequality: E * x_k <= f
            # that is, (E * psi) * U <= f - (E * phi) * x_init
            G.append(dot(self.E(k), psi))
            h.append(self.f(k) - dot(dot(self.E(k), phi), self.x_init))

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

        # Cost 1: sum_k u_k^2
        P1 = eye(self.U_dim)
        q1 = zeros(self.U_dim)

        # Cost2: |x_N - x_goal|^2 = |A x - b|^2
        A = self.psi_last
        b = self.x_goal - dot(self.phi_last, self.x_init)
        P2 = dot(A.T, A)
        q2 = -dot(b.T, A)

        # Weighted combination of all costs
        w1 = 1.
        w2 = 1000.
        P = w1 * P1 + w2 * P2
        q = w1 * q1 + w2 * q2

        # Inequality constraints
        G_control = block_diag(*[self.C(k) for k in xrange(self.nb_steps)])
        h_control = hstack([self.d(k) for k in xrange(self.nb_steps)])
        G = vstack([G_control, self.G_state])
        h = hstack([h_control, self.h_state])
        G = G_control
        h = h_control
        # G = self.G_state
        # h = self.h_state

        self.U = solve_qp(P, q, G, h)
        import pylab
        print "End error:", (
            dot(self.U, dot(P2, self.U)) + 2 * dot(q2, self.U) +
            dot(b.T, b))
        print "End error:", pylab.norm(
            dot(self.phi_last, self.x_init) + dot(self.psi_last, self.U) -
            self.x_goal) ** 2


class COMAccelPreviewControl(PreviewControl):

    def __init__(self, com_init, comd_init, com_goal, comd_goal, tube,
                 duration, switch_time, nb_steps):
        dT = duration / nb_steps
        I, Z = eye(3), zeros((3, 3))
        A = array(bmat([
            [I, dT * I],
            [Z, I]]))
        B = array(bmat([
            [.5 * dT ** 2 * I],
            [dT * I]]))
        x_init = hstack([com_init, comd_init])
        x_goal = hstack([com_goal, comd_goal])
        switch_step = int(switch_time / dT)
        C, d, E, f = self.compute_inequalities(tube, switch_step, nb_steps)
        super(COMAccelPreviewControl, self).__init__(
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
        E1_pos, f1 = tube.compute_primal_hrep(stance_id=0)
        E1 = hstack([E1_pos, zeros(E1_pos.shape)])
        if switch_step >= nb_steps - 1:
            C = wrap_matrix(C1)
            d = wrap_matrix(d1)
            E = wrap_matrix(E1)
            f = wrap_matrix(f1)
        else:
            C2, d2 = tube.compute_dual_hrep(stance_id=1)
            E2_pos, f2 = tube.compute_primal_hrep(stance_id=1)
            E2 = hstack([E2_pos, zeros(E2_pos.shape)])
            C = multiplex_matrices(C1, C2, switch_step)
            d = multiplex_matrices(d1, d2, switch_step)
            E = multiplex_matrices(E1, E2, switch_step)
            f = multiplex_matrices(f1, f2, switch_step)
        return (C, d, E, f)


class FeedbackPreviewController(object):

    def __init__(self, com, fsm, preview_buffer, nb_mpc_steps, tube_shape,
                 tube_radius, draw_cone=True, draw_tube=True):
        """
        Create a new feedback controller that continuously runs the preview
        controller and sends outputs to a COMAccelBuffer.

        INPUT:

        - ``com`` -- PointMass containing current COM state
        - ``fsm`` -- instance of finite state machine
        - ``preview_buffer`` -- COMAccelBuffer to send MPC outputs to
        - ``nb_mpc_steps`` -- discretization step of the preview window
        - ``tube_shape`` -- number of vertices of the COM trajectory tube
        - ``tube_radius`` -- tube radius (in L1 norm)
        """
        self.draw_cone = draw_cone
        self.draw_tube = draw_tube
        self.com = com
        self.fsm = fsm
        self.nb_mpc_steps = nb_mpc_steps
        self.preview_buffer = preview_buffer
        self.target_box = PointMass(fsm.cur_stance.com, 30., color='g')
        self.thread = None
        self.thread_lock = None
        self.tube = None
        self.tube_radius = tube_radius
        self.tube_shape = tube_shape

    def show_cone(self):
        self.draw_cone = True

    def hide_cone(self):
        self.draw_cones = False
        self.cone_handle = None

    def on_tick(self, sim):
        cur_com = self.com.p
        cur_comd = self.com.pd
        cur_stance = self.fsm.cur_stance
        next_stance = self.fsm.next_stance
        preview_targets = self.fsm.get_preview_targets()
        switch_time, horizon, target_com = preview_targets
        target_comd = zeros(3)
        self.target_box.set_pos(target_com)
        self.tube = COMTube(
            cur_com, target_com, cur_stance, next_stance, self.tube_shape,
            self.tube_radius)
        if self.draw_tube:
            self.tube_handle = self.tube.draw_primal_polytopes()
        if True:
            print "\ncur_com =", repr(cur_com)
            print "cur_comd =", repr(cur_comd)
            print "target_com =", repr(target_com)
            print "switch_time =", switch_time
            print "horizon =", horizon
            print ""
            print "cur_stance.is_single_support =", \
                cur_stance.is_single_support
            print "cur_stance.is_single_support =", \
                self.fsm.cur_stance.is_single_support
        try:
            preview_control = COMAccelPreviewControl(
                cur_com, cur_comd, target_com, target_comd, self.tube,
                horizon, switch_time, self.nb_mpc_steps)
        except TubeError as e:
            print "Tube error: %s" % str(e)
            self.cone_handle = None
            sim.stop()
            return
        preview_control.compute_dynamics()
        try:
            preview_control.compute_control()
            self.preview_buffer.update_preview(preview_control)
        except ValueError as e:
            warn("MPC: couldn't solve QP, maybe inconsistent constraints?")
            print "Exception:", e
            sim.stop()
        try:
            if self.draw_cone:
                self.cone_handle = self.tube.draw_dual_cones()
        except TubeError as e:
            print "Tube error (drawing): %s" % str(e)
