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

from tube import TrajectoryTube
from numpy import array, bmat, dot, eye, hstack, ones, sqrt, vstack, zeros
from pymanoid import PointMass, solve_qp
from scipy.linalg import block_diag
from threading import Lock, Thread
from warnings import warn


def norm(v):
    return sqrt(dot(v, v))


COMPARE_QP_SOLVERS = False

if COMPARE_QP_SOLVERS:
    # https://github.com/stephane-caron/oqp
    from oqp import solve_qp_cvxopt
    from oqp import solve_qp_qpoases
    from oqp import solve_qp_quadprog
    from numpy import average
    import time
    l0 = []
    l1 = []
    l2 = []
    t_origin = time.time()


class PreviewControl(object):

    def __init__(self, A, B, C, d, x_init, x_goal, nb_steps):
        """
        Preview control for a system with linear dynamics:

            x_{k+1} = A * x_k + B * u_k

        subject to constraints:

            C * u <= d
            x(0) = x_init
            x(duration) = x_goal

        """
        self.A = A
        self.B = B
        self.C = C
        self.d = d
        self.nb_steps = nb_steps
        self.u_dim = C.shape[1]
        self.U_dim = self.u_dim * self.nb_steps
        self.x_dim = x_init.shape[0]
        self.X_dim = self.x_dim * self.nb_steps
        self.x_goal = x_goal
        self.x_init = x_init

        self.compute_dynamics()
        self.compute_control()

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

            x_N = phi_last * x_0 + psi_last * U

        """
        N = self.nb_steps
        phi = eye(self.x_dim)
        psi = zeros((self.x_dim, self.U_dim))
        for i in xrange(N):
            phi = dot(self.A, phi)
            psi = dot(self.A, psi)
            psi[:, self.u_dim * i:self.u_dim * (i + 1)] = self.B
        self.phi_last = phi
        self.psi_last = psi

    def compute_control(self):
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
        if True:
            G = block_diag(*[self.C for _ in xrange(self.nb_steps)])
            h = hstack([self.d for _ in xrange(self.nb_steps)])
        else:
            G = vstack([+eye(self.U_dim), -eye(self.U_dim)])
            h = hstack([ones(self.U_dim), ones(self.U_dim)])

        if COMPARE_QP_SOLVERS:
            t0 = time.time()
            U0 = solve_qp_cvxopt(P, q, G, h)
            t1 = time.time()
            l0.append(t1 - t0)
            U1 = solve_qp_qpoases(P, q, G, h)
            t2 = time.time()
            l1.append(t2 - t1)
            U2 = solve_qp_quadprog(P, q, G, h)
            l2.append(time.time() - t2)
            if len(l0) % 100 == 0 and False:
                print "Over %.2f s (%d iterations)" % (t2 - t_origin, len(l0))
                print "- avg. CVXOPT   = %.2f ms" % (1000. * average(l0))
                print "- avg. qpOASES  = %.2f ms" % (1000. * average(l1))
                print "- avg. quadprog = %.2f ms" % (1000. * average(l2))
                print "------------------------"
            if norm(U0 - U1) > 1e-2 or norm(U2 - U1) > 1e-2:
                warn("|U0 - U1| = %.1e and |U2 - U1| = %.1e" % (
                    norm(U0 - U1), norm(U2 - U1)))

        self.U = solve_qp(P, q, G, h)
        e = dot(A, self.U) - b
        self.end_cost = dot(e, e)


class COMAccelPreviewControl(PreviewControl):

    def __init__(self, com_init, comd_init, com_goal, comd_goal, comdd_face,
                 duration, nb_steps):
        assert com_init.shape[0] == 3
        assert com_goal.shape[0] == 3
        assert comd_init.shape[0] == 3
        assert comd_goal.shape[0] == 3
        dT = duration / nb_steps
        I, Z = eye(3), zeros((3, 3))
        A = array(bmat([
            [I, dT * I],
            [Z, I]]))
        B = array(bmat([
            [.5 * dT ** 2 * I],
            [dT * I]]))
        C, d = comdd_face
        assert C.shape[1] == 3, "C.shape = %s" % str(C.shape)
        x_init = hstack([com_init, comd_init])
        x_goal = hstack([com_goal, comd_goal])
        super(COMAccelPreviewControl, self).__init__(
            A, B, C, d, x_init, x_goal, nb_steps)
        self.duration = duration
        self.timestep = dT


class FeedbackPreviewController(object):

    def __init__(self, fsm, com_buffer, nb_mpc_steps, tube_shape,
                 draw_cone=True, draw_tube=True):
        """
        Create a new feedback controller that continuously runs the preview
        controller and sends outputs to a COMAccelBuffer.

        INPUT:

        - ``fsm`` -- instance of finite state machine
        - ``com_buffer`` -- COMAccelBuffer to send MPC outputs to
        - ``nb_mpc_steps`` -- discretization step of the preview window
        - ``tube_shape`` -- number of vertices of the COM trajectory tube
        """
        self.com_buffer = com_buffer
        self.draw_cone = draw_cone
        self.draw_tube = draw_tube
        self.fsm = fsm
        self.nb_mpc_steps = nb_mpc_steps
        self.target_box = PointMass(fsm.cur_stance.com, 30., color='g')
        self.thread = None
        self.thread_lock = None
        self.tube_shape = tube_shape

    def show_cone(self):
        self.draw_cone = True

    def hide_cone(self):
        self.draw_cone = False
        self.cone_handle = None

    def start_thread(self):
        self.thread_lock = Lock()
        self.thread = Thread(
            target=self.run_thread, args=())
        self.thread.daemon = True
        self.thread.start()

    def pause_thread(self):
        self.thread_lock.acquire()

    def resume_thread(self):
        self.thread_lock.release()

    def stop_thread(self):
        self.thread_lock = None

    def run_thread(self):
        target_comd = zeros(3)
        while self.thread_lock:
            cur_com = self.com_buffer.com.p
            cur_comd = self.com_buffer.comd
            cur_stance = self.fsm.cur_stance
            preview_horizon, target_com = self.fsm.get_preview_targets()
            self.target_box.set_pos(target_com)
            tube = TrajectoryTube(
                cur_com, target_com, cur_stance, self.tube_shape)
            if tube.nb_vertices < 2:
                continue
            comdd_face = tube.compute_dual_cone()
            if comdd_face is None:
                continue
            if self.draw_cone:
                self.cone_handle = tube.draw_dual_cone()
            if self.draw_tube:
                self.tube_handle = tube.draw()
            try:
                preview_control = COMAccelPreviewControl(
                    cur_com, cur_comd, target_com, target_comd, comdd_face,
                    preview_horizon, nb_steps=self.nb_mpc_steps)
                self.com_buffer.update_control(preview_control)
            except ValueError:
                warn("MPC failed: inconsistent constraints?")
