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

import IPython
import os.path
import sys
import thread
import time

from numpy import array, cross, dot, hstack, ones

try:
    import pymanoid
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../pymanoid')
    import pymanoid
    from pymanoid.tasks import COMTask

try:
    from wpg.cwc import compute_cwc_pyparma
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/..')
    from wpg.cwc import compute_cwc_pyparma

try:
    from polygon import compute_polygon_hull
    from polygon import draw_static_polygon
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../sep')
    from polygon import compute_polygon_hull
    from polygon import draw_static_polygon

if os.path.isfile('HRP4R.dae'):
    from pymanoid.robots import HRP4 as RobotModel
else:  # default to JVRC-1
    from pymanoid.robots import JVRC1 as RobotModel


dt = 3e-2  # [s]
gui_handles = {}
robot_mass = 39.  # [kg] later set from RobotModel
z_high = 1.5
z_mid = 0.75


def run_ik_thread():
    while True:
        robot.step_ik(dt)
        com.set_pos(robot.com)
        time.sleep(dt)


def run_forces_thread():
    handles = []
    while True:
        try:
            support = contacts.find_static_supporting_forces(
                outbox.p, robot_mass)
            handles = [pymanoid.draw_force(c, fc) for (c, fc) in support]
        except Exception as e:
            print "Force computation failed:", e
        time.sleep(dt)
    return handles


def compute_acceleration_set(contact_set, p_com, display_scale=0.05):
    gravity = pymanoid.get_gravity()
    # A_O = contacts.compute_wrench_cone([0, 0, 0])
    A_O = compute_cwc_pyparma(contact_set, [0, 0, 0])
    B = A_O[:, :3] + cross(A_O[:, 3:], p_com)
    c = dot(B, gravity)
    g = -gravity[2]
    assert g > 0
    # assert all(c > 0), "c > 0 assertion failed"
    # assert all(B[:, 2] < 0)
    if any(abs(c) < 1e-10):
        I = [i for i in xrange(len(c)) if abs(c[i]) > 1e-10]
        B, c = B[I], c[I]
    check = c / B[:, 2]
    assert max(check) - min(check) < 1e-10, "max - min failed"
    assert abs(check[0] - (-g)) < 1e-10, "check is not -g?"
    sigma = c / g
    B2 = hstack([
        (B[:, column] / sigma).reshape((B.shape[0], 1))
        for column in [0, 1]])
    vertices2d = compute_polygon_hull(B2, ones(len(c)))

    def vertices_at(z):
        v = [array([a * (g + z), b * (g + z)]) for (a, b) in vertices2d]
        return [array([x, y, z]) for (x, y) in v]

    return [array([0, 0, -g])] + vertices_at(z=+g)


def draw_cone_thread():
    while True:
        try:
            COMDD_K = 0.05
            vertices = compute_acceleration_set(contacts, com.p)
            vscale = [com_target.p + COMDD_K * array(v) for v in vertices]
            gui_handles['acc'] = pymanoid.draw_3d_cone(
                apex=vscale[0], axis=[0, 0, 1], section=vscale[1:],
                combined='r.-#', linewidth=1.)
        except Exception as e:
            print "draw_acceleration_thread:", e
            continue
        time.sleep(1e-2)


def recompute_static_polygon():
    global static_handle
    static_handle = draw_static_polygon(
        contacts, p=[0., 0., robot.com[2]], combined='g.-#')


if __name__ == "__main__":
    pymanoid.init()
    viewer = pymanoid.get_env().GetViewer()
    viewer.SetCamera(array([
        [0.60587192, -0.36596244,  0.70639274, -2.4904027],
        [-0.79126787, -0.36933163,  0.48732874, -1.6965636],
        [0.08254916, -0.85420468, -0.51334199,  2.79584694],
        [0.,  0.,  0.,  1.]]))
    robot = RobotModel(download_if_needed=True)
    robot.set_transparency(0.5)
    robot_mass = robot.mass

    fname = sys.argv[1] if len(sys.argv) > 1 else '../stances/double.json'
    contacts = pymanoid.ContactSet.from_json(fname)

    com = pymanoid.Cube(0.01, color='r')
    outbox = pymanoid.Cube(0.05, color='g')
    com_target = pymanoid.Cube(0.01, visible=False)
    if 'single.json' in fname:
        outbox.set_pos([0.,  0.,  z_high])
    elif 'double.json' in fname or 'triple.json' in fname:
        outbox.set_pos([0.3,  0.14,  z_high])
    com_target.set_x(outbox.x)
    com_target.set_y(outbox.y)
    com_target.set_z(outbox.z - z_high + z_mid)

    robot.set_dof_values(robot.q_halfsit)
    robot.set_active_dofs(
        robot.chest + robot.free + robot.legs + robot.arms)
    robot.init_ik()
    robot.generate_posture(contacts)
    robot.ik.add_task(COMTask(robot, com_target))

    # Uncomment the following to show the static-equilibrium
    # polygon as in Figure 2 of the paper:
    # recompute_static_polygon()

    thread.start_new_thread(run_ik_thread, ())
    thread.start_new_thread(run_forces_thread, ())
    thread.start_new_thread(draw_cone_thread, ())

    viewer.SetBkgndColor([1., 1., 1.])
    viewer.SetCamera([
        [0.7941174,  0.30773652, -0.52409521,  2.07556486],
        [0.60733387, -0.36935776,  0.70336365, -2.3524673],
        [0.02287205, -0.87685408, -0.48021224,  2.18942738],
        [0.,  0.,  0.,  1.]])

    IPython.embed()
