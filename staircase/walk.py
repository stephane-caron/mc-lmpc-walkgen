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

try:
    import pymanoid
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../pymanoid')
    import pymanoid

try:
    from mpc3d.com_buffer import COMAccelBuffer
    from mpc3d.control import FeedbackPreviewController
    from mpc3d.fsm import StanceFSM
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/..')
    from mpc3d.com_buffer import COMAccelBuffer
    from mpc3d.control import FeedbackPreviewController
    from mpc3d.fsm import StanceFSM

from numpy import arange, cos, hstack, pi, sin, zeros, array
from numpy.random import random, seed
from pymanoid import draw_force, draw_polygon, Contact, PointMass
from pymanoid.tasks import ContactTask, DOFTask, LinkPoseTask, MinCAMTask
from threading import Lock
from time import sleep as real_sleep
from time import time as real_time

try:
    from hrp4_pymanoid import HRP4 as RobotModel
except ImportError:
    from pymanoid.robots import JVRC1 as RobotModel


dt = 3e-2  # [s]
gui_handles = {}
robot_lock = Lock()
robot_mass = 39.  # [kg] updated after robot model is loaded
sim_timescale = 1.
start_time = None


def sim_sleep(duration):
    real_sleep(sim_timescale * duration)


def sim_time():
    return sim_timescale * real_time()


def generate_staircase(radius, angular_step, height, roughness, friction,
                       step_dim_x, step_dim_y):
    """
    Generate a new slanted staircase with tilted steps.

    INPUT:

    radius -- staircase radius (in [m])
    angular_step -- angular step between contacts (in [rad])
    height -- altitude variation (in [m])
    roughness -- amplitude of roll-pitch-yaw orientations at contacts (in [rad])
    friction -- friction coefficient between a robot foot and a step
    step_dim_x -- half-length of each step
    step_dim_y -- half-width of each step
    """
    steps = []
    for theta in arange(0., 2 * pi, angular_step):
        left_foot = Contact(
            X=step_dim_x,
            Y=step_dim_y,
            pos=[radius * cos(theta),
                 radius * sin(theta),
                 radius + .5 * height * sin(theta)],
            rpy=(roughness * (random(3) - 0.5) +
                 [0, 0, theta + .5 * pi]),
            friction=friction,
            visible=True)
        right_foot = Contact(
            X=step_dim_x,
            Y=step_dim_y,
            pos=[1.3 * radius * cos(theta + .5 * angular_step),
                 1.3 * radius * sin(theta + .5 * angular_step),
                 radius + .5 * height * sin(theta + .5 * angular_step)],
            rpy=(roughness * (random(3) - 0.5) +
                 [0, 0, theta + .5 * pi]),
            friction=friction,
            visible=True)
        steps.append(left_foot)
        steps.append(right_foot)
    return steps


def empty_gui_list(l):
    for i in xrange(len(l)):
        l[i] = None


def dash_graph_handles(handles):
    for i in xrange(len(handles)):
        if i % 2 == 0:
            handles[i] = None


def set_camera_0():
    pymanoid.get_viewer().SetCamera([
        [0.97248388,  0.01229851, -0.23264533,  2.34433222],
        [0.21414823, -0.44041135,  0.87188209, -2.02105641],
        [-0.0917368, -0.89771186, -0.43092664,  3.40723848],
        [0.,  0.,  0.,  1.]])


def set_camera_1():
    pymanoid.get_viewer().SetCamera([
        [1., 0.,  0., -7.74532557e-04],
        [0., 0.,  1., -4.99819374e+00],
        [0., -1., 0., 1.7],
        [0., 0.,  0., 1.]])


def prepare_screenshot(scrot_time=38.175):
    set_camera_1()
    if start_time is None:
        start()
    while real_time() < start_time + scrot_time:
        real_sleep(1e-2)
    stop()
    empty_gui_list(gui_handles['forces'])
    empty_gui_list(gui_handles['static'])
    # empty_gui_list(mpc.cone_handle)
    mpc.target_box.hide()
    # mpc.tube_handle.SetShow(False)
    com_buffer.com.set_visible(False)
    com_buffer.comdd_handle.Close()
    robot.set_transparency(0)
    viewer.SetBkgndColor([1, 1, 1])
    dash_graph_handles(fsm.left_foot_traj_handles)
    dash_graph_handles(fsm.right_foot_traj_handles)
    # dash_graph_handles(com_buffer.com_traj_handles)


def fsm_post_step_callback():
    # (1) Update static-equilibrium polygon
    vertices = fsm.cur_stance.compute_static_equilibrium_area(robot_mass)
    gui_handles['static'] = draw_polygon(
        [(x[0], x[1], com_buffer.cur_height) for x in vertices],
        normal=[0, 0, 1])
    # (2) Update IK tasks
    with robot_lock:
        robot.ik.remove_task(robot.left_foot.name)
        robot.ik.remove_task(robot.right_foot.name)
        if fsm.cur_stance.left_foot is not None:
            robot.ik.add_task(
                ContactTask(robot, robot.left_foot, fsm.cur_stance.left_foot))
        else:  # left_foot is free
            fsm.free_foot.reset(robot.left_foot.pose, fsm.next_contact.pose)
            robot.ik.add_task(
                LinkPoseTask(robot, robot.left_foot, fsm.free_foot))
        if fsm.cur_stance.right_foot is not None:
            robot.ik.add_task(
                ContactTask(robot, robot.right_foot, fsm.cur_stance.right_foot))
        else:  # right_foot is free
            fsm.free_foot.reset(robot.right_foot.pose, fsm.next_contact.pose)
            robot.ik.add_task(
                LinkPoseTask(robot, robot.right_foot, fsm.free_foot))


last_bkgnd_switch = None


def comdd_callback(comdd):
    global last_bkgnd_switch
    gravity = pymanoid.get_gravity()
    wrench = hstack([robot_mass * (comdd - gravity), zeros(3)])
    support = fsm.cur_stance.find_supporting_forces(
        wrench, com_buffer.com.p, 39., 10.)
    if not support:
        gui_handles['forces'] = []
        viewer.SetBkgndColor([.8, .4, .4])
        last_bkgnd_switch = real_time()
    else:
        gui_handles['forces'] = [draw_force(c, fc) for (c, fc) in support]
    if last_bkgnd_switch is not None \
            and real_time() - last_bkgnd_switch > 0.2:
        # let's keep epilepsy at bay
        viewer.SetBkgndColor([.6, .6, .8])
        last_bkgnd_switch = None


if __name__ == "__main__":
    seed(42)
    pymanoid.init()
    viewer = pymanoid.get_viewer()
    set_camera_0()
    robot = RobotModel(download_if_needed=True)
    robot_mass = robot.mass  # saved here to avoid taking robot_lock
    robot.set_transparency(0.3)

    staircase = generate_staircase(
        radius=1.4,
        angular_step=0.5,
        height=1.4,
        roughness=0.5,
        friction=0.7,
        step_dim_x=0.2,
        step_dim_y=0.1)

    com_pm = PointMass([0, 0, 0], robot_mass)
    fsm = StanceFSM(
        staircase, com_pm, 'DS-R',
        ss_duration=1.0,
        ds_duration=0.5,
        init_com_offset=array([0.05, 0., 0.]),
        cyclic=True)
    com_buffer = COMAccelBuffer(com_pm, fsm)
    mpc = FeedbackPreviewController(
        fsm, com_buffer, nb_mpc_steps=10, tube_shape=2)

    with robot_lock:
        robot.set_dof_values(robot.q_halfsit)
        active_dofs = robot.chest + robot.free
        active_dofs += robot.left_leg + robot.right_leg
        robot.set_active_dofs(active_dofs)
        robot.init_ik(
            gains={
                # NB: compared with pre-print values, pymanoid gains have been
                # updated by commit c7851e092a0876ae... (see log for details)
                'com': 1.,
                'contact': 0.9,
                'link_pose': 0.9,
                'posture': 0.005,
            },
            weights={
                # NB: due to an IK bug, the pre-print video was generated with
                # a free link gain of 100 instead of 5; see pymanoid commit
                # f2c24b95936aacb6b905f9adfb0cc07af3127b2d.
                'com': 5.,
                'contact': 100.,
                'link_pose': 100.,  # not 5.
                'posture': 0.1,
            })
        robot.set_dof_values([2.], [robot.TRANS_Z])  # start PG from above
        robot.generate_posture(fsm.cur_stance, max_it=200)
        robot.ik.tasks['com'].update_target(com_buffer.com)
        robot.ik.add_task(MinCAMTask(robot, weight=0.1))

        try:  # HRP-4
            robot.ik.add_task(
                DOFTask(robot, robot.CHEST_P, 0.2, gain=0.9, weight=0.05))
            robot.ik.add_task(
                DOFTask(robot, robot.CHEST_Y, 0., gain=0.9, weight=0.05))
            robot.ik.add_task(
                DOFTask(robot, robot.ROT_P, 0., gain=0.9, weight=0.05))
        except AttributeError:  # JVRC-1
            robot.ik.add_task(
                DOFTask(robot, robot.WAIST_P, 0.2, gain=0.9, weight=0.5))
            robot.ik.add_task(
                DOFTask(robot, robot.WAIST_Y, 0., gain=0.9, weight=0.5))
            robot.ik.add_task(
                DOFTask(robot, robot.WAIST_R, 0., gain=0.9, weight=0.5))
            robot.ik.add_task(
                DOFTask(robot, robot.ROT_P, 0., gain=0.9, weight=0.5))

    def start():
        global start_time
        fsm.start_thread(dt, fsm_post_step_callback, sim_sleep)
        com_buffer.start_thread(comdd_callback, sim_sleep)
        mpc.start_thread()
        robot.start_ik_thread(dt, sim_sleep)
        start_time = real_time()

    def stop():
        fsm.stop_thread()
        com_buffer.stop_thread()
        mpc.stop_thread()
        robot.stop_ik_thread()

    set_camera_1()
    print """

Multi-contact WPG based on Model Preview Control of 3D COM Accelerations
========================================================================

Ready to go! Type one of the following commands:

    start() -- start walking
    stop() -- freeze (cannot restart afterwards)
    prepare_screenshot() -- used to generate Figure 1 of the paper

You can access all state variables via this IPython shell.
Here is the list of global objects. Use <TAB> to see what's inside.

    com_buffer -- stores MPC output and feeds it to the IK
    fsm -- finite state machine
    mpc -- model-preview controller
    robot -- kinematic model of the robot (includes IK solver)

For example:

    com_buffer.comdd -- desired COM acceleration
    fsm.cur_stance -- current stance (contacts + target COM)
    mpc.hide_cone() -- hide drawing of preview COM acceleration cone
    robot.com -- robot COM position from kinematic model

Enjoy :)

"""
    if IPython.get_ipython() is None:
        IPython.embed()
