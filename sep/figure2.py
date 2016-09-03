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
import threading
import time

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

from numpy import array, cross, dot, hstack, ones
from polygon import compute_polygon_hull
from polygon import draw_static_polygon

try:
    from hrp4_pymanoid import HRP4 as RobotModel
except ImportError:
    from pymanoid.robots import JVRC1 as RobotModel


cyan = (0., 0.5, 0.5, 0.5)
dt = 3e-2  # [s]
robot_lock = threading.Lock()
green = (0., 0.5, 0., 0.5)
gui_handles = {}
handles = [None, None]
magenta = (0.5, 0., 0.5, 0.5)
orange = (0.5, 0.5, 0., 0.5)
saved_handles = []
screenshot = False
z_high = 1.5
z_mid = 0.75

# IK params
qd_lim = 10.
K_doflim = 5.
G_com = 1. / dt
G_link = 0.9 / dt
w_link = 100.
w_com = 005.
w_reg = 001.


def run_ik_thread():
    while True:
        robot_lock.acquire()
        robot.step_ik(dt)
        com.set_pos(robot.com)
        com_target.set_x(outbox.x)
        com_target.set_y(outbox.y)
        com_target.set_z(outbox.z - z_high + z_mid)
        robot_lock.release()
        time.sleep(dt)


def run_forces_thread():
    handles = []
    while True:
        robot_lock.acquire()
        try:
            support = contacts.find_static_supporting_forces(
                outbox.p, robot.mass)
            handles = [pymanoid.draw_force(c, fc) for (c, fc) in support]
        except Exception as e:
            print "Force computation failed:", e
        robot_lock.release()
        time.sleep(dt)
    return handles


def save_handles():
    global saved_handles
    saved_handles.append(handles[0])
    saved_handles.append(handles[1])


def check_cwc_dual():
    A_O = contacts.compute_wrench_cone([0, 0, 0])
    for i in xrange(A_O.shape[0]):
        a_O, a = A_O[i, :3], A_O[i, 3:]
        for contact in contacts.contacts:
            for point in contact.vertices:
                for s in contact.force_span:
                    assert (dot(a_O + cross(a, point), s) < 1e-10)
    print "OK"


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
        print "ici!!!!!"
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
            COMDD_K = 0.08
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
    static_handle = draw_static_polygon(contacts, p=[0., 0., robot.com[2]],
                                        combined='g.-#')


def prepare_screenshot(ambient=0., diffuse=0.9):
    global screenshot
    screenshot = True
    # outbox.set_visible(False)
    viewer.SetBkgndColor([1., 1., 1.])
    # with robot_lock:
    #     robot.set_transparency(0)
    #     for link in robot.rave.GetLinks():
    #         if len(link.GetGeometries()) > 0:
    #             geom = link.GetGeometries()[0]
    #             geom.SetAmbientColor([ambient] * 3)
    #             geom.SetDiffuseColor([diffuse] * 3)
    viewer.SetCamera([
        [0.7941174,  0.30773652, -0.52409521,  2.07556486],
        [0.60733387, -0.36935776,  0.70336365, -2.3524673],
        [0.02287205, -0.87685408, -0.48021224,  2.18942738],
        [0.,  0.,  0.,  1.]])


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

    # contacts = pymanoid.ContactSet({
    #     'left_foot': pymanoid.Contact(
    #         X=0.2,
    #         Y=0.1,
    #         pos=[0.40, 0.15, 0.1],
    #         rpy=[0, 0, 0],
    #         friction=0.5,
    #         visible=True),
    #     'right_foot': pymanoid.Contact(
    #         X=0.2,
    #         Y=0.1,
    #         pos=[0., -0.195, 0.],
    #         rpy=[0, 0, 0],
    #         friction=0.5,
    #         visible=True),
    #     'left_hand': pymanoid.Contact(
    #         X=0.1,
    #         Y=0.05,
    #         pos=[0.55, 0.46, 0.96],
    #         rpy=[0., -0.8, 0.],
    #         friction=0.5,
    #         visible=True)
    # })

    fname = sys.argv[1] if len(sys.argv) > 1 else 'stances/figure2-double.json'
    contacts = pymanoid.ContactSet.from_json(fname)

    com = pymanoid.Cube(0.01, color='r')
    outbox = pymanoid.Cube(0.02, color='b')
    com_target = pymanoid.Cube(0.01, visible=False)

    if 'figure2-single.json' in fname:
        outbox.set_pos([0.,  0.,  z_high])
    elif 'figure2-double.json' in fname or 'figure2-triple.json' in fname:
        outbox.set_pos([0.3,  0.04,  z_high])
    else:
        warn("Unknown contact set, you will have to set the COM position.")

    # robot.set_dof_values(
    #     [0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
    #      +0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
    #      +0.00000000e+00,   0.00000000e+00,  -9.22429319e-16,
    #      -3.48563058e-16,  -9.22429319e-16,  -3.48563058e-16,
    #      -9.22429319e-16,  -3.48563058e-16,  -9.22429319e-16,
    #      -3.48563058e-16,   6.03576484e-01,  -4.25114264e-01,
    #      -4.44119059e-01,   1.22293674e-02,   2.21026645e-01,
    #      -1.03735006e-01,   7.82306779e-01,   4.36332312e-01,
    #      -1.34390352e+00,   1.93795649e+00,  -8.06028081e-01,
    #      -1.65483336e-02,   3.91923914e-01,   2.77184351e-01,
    #      +0.00000000e+00,  -1.57009246e-16,  -1.54186877e+00,
    #      -8.63021181e-01,  -1.11825384e-01,  -8.44173746e-01,
    #      +0.00000000e+00,   1.10377247e-15,   9.27601065e-15,
    #      +1.27675648e-15,   0.00000000e+00,  -3.56455657e-01,
    #      +5.04355129e-01,   7.07895451e-01,  -1.97006496e-01,
    #      -1.58161387e+00,  -7.15584993e-01,   1.23044758e+00,
    #      +0.00000000e+00,   4.05153619e-16,  -1.16736736e-01,
    #      +9.89644142e-02,   6.95014625e-01,   1.42318089e-01,
    #      +1.91849964e-01,  -6.91626439e-01])

    with robot_lock:
        robot.set_dof_values(robot.q_halfsit)
        robot.set_active_dofs(
            robot.chest + robot.free + robot.legs + robot.arms)
        robot.init_ik()
        robot.generate_posture(contacts)
        robot.ik.add_task(COMTask(robot, com_target))

    recompute_static_polygon()

    thread.start_new_thread(run_ik_thread, ())
    # thread.start_new_thread(run_forces_thread, ())
    thread.start_new_thread(draw_cone_thread, ())

    prepare_screenshot()

    IPython.embed()
