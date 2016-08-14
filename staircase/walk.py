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
import time
import threading

try:
    import pymanoid
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../pymanoid')
    import pymanoid

try:
    from mpc3d.buffer import PreviewBuffer  # whatever
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/..')
    from mpc3d.buffer import PreviewBuffer
    from mpc3d.control import FeedbackPreviewController
    from mpc3d.fsm import StateMachine
    from mpc3d.simulation import Simulation

from numpy import arange, cos, hstack, pi, sin, zeros, array
from numpy.random import random, seed
from pymanoid import draw_force, draw_polygon, Contact, PointMass
from pymanoid.tasks import ContactTask, DOFTask, LinkPoseTask, MinCAMTask

try:
    from hrp4_pymanoid import HRP4 as RobotModel
except ImportError:
    from pymanoid.robots import JVRC1 as RobotModel


gui_handles = {}
robot_lock = threading.Lock()
robot_mass = 39.  # [kg] updated after robot model is loaded


def generate_staircase(radius, angular_step, height, roughness, friction,
                       step_dim_x, step_dim_y):
    """
    Generate a new slanted staircase with tilted steps.

    INPUT:

    - ``radius`` -- staircase radius (in [m])
    - ``angular_step`` -- angular step between contacts (in [rad])
    - ``height`` -- altitude variation (in [m])
    - ``roughness`` -- amplitude of contact roll, pitch and yaw (in [rad])
    - ``friction`` -- friction coefficient between a robot foot and a step
    - ``step_dim_x`` -- half-length of each step
    - ``step_dim_y`` -- half-width of each step
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
    if not sim.is_started:
        sim.start()
    while sim.time() < scrot_time:
        sim.sleep(1e-2)
    sim.stop()
    # empty_gui_list(gui_handles['forces'])
    # empty_gui_list(gui_handles['static'])
    mpc.target_box.hide()
    preview_buffer.com.set_visible(False)
    robot.set_transparency(0)
    viewer.SetBkgndColor([1, 1, 1])
    dash_graph_handles(fsm.left_foot_traj_handles)
    dash_graph_handles(fsm.right_foot_traj_handles)


def update_sep():
    """UPdate static-equilibrium polygons"""
    if fsm.cur_stance.is_single_support:
        ss_stance = fsm.cur_stance
        ds_stance = fsm.next_stance
    else:  # fsm.cur_stance.is_double_support:
        ss_stance = fsm.next_stance
        ds_stance = fsm.cur_stance
    sep_height = preview_buffer.com.z - RobotModel.leg_length
    gui_handles['static-ss'] = draw_polygon(
        [(x[0], x[1], sep_height) for x in ss_stance.sep],
        normal=[0, 0, 1], color='c')
    gui_handles['static-ds'] = draw_polygon(
        [(x[0], x[1], sep_height) for x in ds_stance.sep],
        normal=[0, 0, 1], color='y')


def update_robot_ik():
    with robot_lock:
        robot.ik.remove_task(robot.left_foot.name)
        robot.ik.remove_task(robot.right_foot.name)
        if fsm.cur_stance.left_foot is not None:
            left_foot_task = ContactTask(
                robot, robot.left_foot, fsm.cur_stance.left_foot)
        else:  # left_foot is free
            left_foot_task = LinkPoseTask(
                robot, robot.left_foot, fsm.free_foot)
        if fsm.cur_stance.right_foot is not None:
            right_foot_task = ContactTask(
                robot, robot.right_foot, fsm.cur_stance.right_foot)
        else:  # right_foot is free
            right_foot_task = LinkPoseTask(
                robot, robot.right_foot, fsm.free_foot)
        robot.ik.add_task(left_foot_task)
        robot.ik.add_task(right_foot_task)


def fsm_callback():
    """Function called after each FSM phase transition."""
    update_sep()
    update_robot_ik()


class PausableThread(threading.Thread):

    def __init__(self):
        super(PausableThread, self).__init__()
        self.daemon = True
        self.lock = None

    def pause(self):
        self.lock.acquire()

    def resume(self):
        self.lock.release()

    def stop(self):
        self.lock = None

    def start(self):
        self.lock = threading.Lock()
        super(PausableThread, self).start()

    def run(self):
        while self.lock:
            with self.lock:
                self.step()


class ForcesThread(PausableThread):

    def __init__(self):
        super(ForcesThread, self).__init__(self)
        self.last_bkgnd_switch = None

    def step(self):
        """Find supporting contact forces at each COM acceleration update."""
        sim.sync('forces')
        comdd = preview_buffer.comdd
        gravity = pymanoid.get_gravity()
        wrench = hstack([robot_mass * (comdd - gravity), zeros(3)])
        support = fsm.cur_stance.find_supporting_forces(
            wrench, preview_buffer.com.p, robot_mass, 10.)
        if not support:
            gui_handles['forces'] = []
            viewer.SetBkgndColor([.8, .4, .4])
            self.last_bkgnd_switch = time.time()
        else:
            gui_handles['forces'] = [draw_force(c, fc) for (c, fc) in support]
        if self.last_bkgnd_switch is not None \
                and time.time() - self.last_bkgnd_switch > 0.2:
            # let's keep epilepsy at bay
            viewer.SetBkgndColor([.6, .6, .8])
            self.last_bkgnd_switch = None


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
        friction=0.9,
        step_dim_x=0.2,
        step_dim_y=0.1)

    com_pm = PointMass([0, 0, 0], robot_mass)
    fsm = StateMachine(
        staircase, com_pm, 'DS-R',
        ss_duration=1.0,
        ds_duration=0.5,
        init_com_offset=array([0.05, 0., 0.]),
        cyclic=True,
        callback=fsm_callback)
    preview_buffer = PreviewBuffer(
        com_pm,
        fsm)
    mpc = FeedbackPreviewController(
        com_pm,
        fsm,
        preview_buffer,
        nb_mpc_steps=10,
        tube_shape=8,
        tube_radius=0.02)

    with robot_lock:
        robot.set_dof_values(robot.q_halfsit)
        active_dofs = robot.chest + robot.free
        active_dofs += robot.left_leg + robot.right_leg
        robot.set_active_dofs(active_dofs)
        robot.init_ik(
            gains={
                'com': 1.,
                'contact': 1.,
                'link_pose': 1.,
                'posture': 1.,
            },
            weights={
                'com': 10.,
                'contact': 1000.,
                'link_pose': 100.,
                'posture': 1.,
            })
        robot.set_dof_values([2.], [robot.TRANS_Z])  # start PG from above
        robot.generate_posture(fsm.cur_stance, max_it=200)
        robot.ik.tasks['com'].update_target(preview_buffer.com)
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

    sim = Simulation(dt=3e-2)
    sim.add_callback(fsm.on_tick)
    sim.add_callback(preview_buffer.on_tick)
    sim.add_callback(mpc.on_tick)
    robot.start_ik_thread(3e-2)

    fsm_callback()  # show SE polygons at startup

    set_camera_1()
    if False:
        print """

Multi-contact WPG based on Model Preview Control of 3D COM Accelerations
========================================================================

Ready to go! You can control the simulation by:

    sim.start()     sim.pause()
    sim.stop()      sim.resume()

You can access all state variables via this IPython shell.
Here is the list of global objects. Use <TAB> to see what's inside.

    preview_buffer -- stores MPC output and feeds it to the IK
    fsm -- finite state machine
    mpc -- model-preview controller
    robot -- kinematic model of the robot (includes IK solver)

For example:

    preview_buffer.comdd -- desired COM acceleration
    fsm.cur_stance -- current stance (contacts + target COM)
    mpc.hide_cone() -- hide drawing of preview COM acceleration cone
    robot.com -- robot COM position from kinematic model

Enjoy :)

"""
    if IPython.get_ipython() is None:
        IPython.embed()
