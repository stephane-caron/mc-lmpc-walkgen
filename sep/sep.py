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
import numpy
import os
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

from polygon import compute_static_polygon_bretl
from polygon import compute_static_polygon_cdd_hull
from polygon import compute_static_polygon_pyparma_hull
from polygon import compute_static_polygon_cdd_only
from warnings import warn

if os.path.isfile('HRP4R.dae'):
    from pymanoid.robots import HRP4 as RobotModel
else:  # default to JVRC-1
    from pymanoid.robots import JVRC1 as RobotModel


black = (0., 0., 0., 0.5)
custom_mass = 39
cyan = (0., 0.5, 0.5, 0.5)
dt = 3e-2  # [s]
green = (0., 0.5, 0., 0.5)
gui_handles = {}
handles = [None, None]
magenta = (0.5, 0., 0.5, 0.5)
robot = None
robot_lock = threading.Lock()
robot_mass = 39  # [kg], updated once robot model is loaded
saved_handles = []
threads = []
yellow = (0.5, 0.5, 0., 0.5)
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
screenshot = False


def run_ik_thread():
    while True:
        with robot_lock:
            if robot is not None:
                robot.step_ik(dt)
            com_target.set_x(outbox.x)
            com_target.set_y(outbox.y)
            com_target.set_z(outbox.z - z_high + z_mid)
        time.sleep(dt)


def draw_com_polygon(vertices, color):
    return pymanoid.draw_polygon(
        [(x[0], x[1], com_target.z) for x in vertices],
        normal=[0, 0, 1], combined='m.-#', color=color, pointsize=0.02,
        linewidth=3.)


def draw_cdd_thread():
    while True:
        try:
            vertices = compute_static_polygon_cdd_hull(contacts)
            if vertices:
                gui_handles['cdd'] = draw_com_polygon(vertices, magenta)
        except Exception as e:
            print "draw_cdd_thread:", e
            continue
        time.sleep(1e-2)


def draw_pyparma_thread():
    while True:
        try:
            vertices = compute_static_polygon_pyparma_hull(contacts)
            if vertices:
                gui_handles['pyparma'] = draw_com_polygon(vertices, yellow)
        except Exception as e:
            print "draw_pyparma_thread:", e
            continue
        time.sleep(1e-2)


def draw_bretl_thread():
    while True:
        try:
            vertices = compute_static_polygon_bretl(contacts, solver='glpk')
            gui_handles['bretl'] = draw_com_polygon(vertices, cyan)
        except Exception as e:
            print "draw_bretl_thread:", e
            continue
        time.sleep(1e-2)


def draw_cdd_only_thread():
    while True:
        try:
            # you can vary ``custom_mass`` to check that it has no effect
            vertices = compute_static_polygon_cdd_only(contacts, custom_mass)
            gui_handles['cdd_only'] = draw_com_polygon(vertices, black)
        except Exception as e:
            print "draw_cdd_only_thread:", e
            continue
        time.sleep(1e-2)


if __name__ == "__main__":
    pymanoid.init()
    robot = RobotModel(download_if_needed=True)
    robot.set_transparency(0.2)
    robot_mass = robot.mass
    viewer = pymanoid.get_env().GetViewer()
    viewer.SetCamera(numpy.array([
        [0.60587192, -0.36596244,  0.70639274, -2.4904027],
        [-0.79126787, -0.36933163,  0.48732874, -1.6965636],
        [0.08254916, -0.85420468, -0.51334199,  2.79584694],
        [0.,  0.,  0.,  1.]]))
    viewer.SetBkgndColor([1, 1, 1])

    fname = sys.argv[1] if len(sys.argv) > 1 else '../stances/triple.json'
    contacts = pymanoid.ContactSet.from_json(fname)

    com_target = pymanoid.Cube(0.01, visible=True)
    outbox = pymanoid.Cube(0.02, color='b')

    if 'single.json' in fname:
        outbox.set_pos([0.,  0.,  z_high])
    elif 'double.json' in fname or 'triple.json' in fname:
        outbox.set_pos([0.3,  0.04,  z_high])
    else:
        warn("Unknown contact set, you will have to set the COM position.")

    with robot_lock:
        robot.set_dof_values(robot.q_halfsit)
        robot.set_active_dofs(
            robot.chest + robot.legs + robot.arms + robot.free)
        robot.init_ik()
        robot.generate_posture(contacts)
        robot.ik.add_task(COMTask(robot, com_target))

    print """
Static-equilibrium polygon computations
=======================================

Legend:
- Magenta area: computed using cdd + Qhull
- Yellow area: computed using Parma + Qhull
- Green area: computed using Bretl and Lall's method
- Black area: computed using cdd only"""

    threads.append(thread.start_new_thread(run_ik_thread, ()))
    threads.append(thread.start_new_thread(draw_cdd_thread, ()))
    threads.append(thread.start_new_thread(draw_bretl_thread, ()))
    threads.append(thread.start_new_thread(draw_pyparma_thread, ()))
    if contacts.nb_contacts < 3:
        # too slow for triple-contact, would freeze the whole process
        threads.append(thread.start_new_thread(draw_cdd_only_thread, ()))
    contacts.start_force_thread(com_target, robot_mass, dt=1e-2)

    if IPython.get_ipython() is None:
        IPython.embed()
