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
import os
import re
import sys
import time
import threading

try:
    from pymanoid import init
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../pymanoid')
    from pymanoid import init, draw_line, draw_points, get_gravity, get_viewer
    from pymanoid import draw_force, draw_polygon, Contact, PointMass
    from pymanoid.tasks import ContactTask, DOFTask, LinkPoseTask, MinCAMTask
    from pymanoid import set_camera_top, draw_point
    from pymanoid.draw import draw_3d_cone, draw_polyhedron

try:
    from mpc3d.buffer import PreviewBuffer  # whatever
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/..')
    from mpc3d.buffer import PreviewBuffer
    from mpc3d.control import TubePreviewControl
    from mpc3d.fsm import StateMachine
    from mpc3d.simulation import Process, Simulation

from numpy import arange, cos, hstack, pi, sin, zeros, array
from numpy.random import random, seed

try:
    from hrp4_pymanoid import HRP4 as RobotModel
except ImportError:
    from pymanoid.robots import JVRC1 as RobotModel


# Settings
dt = 3e-2           # [s] simulation time step
ds_duration = 0.5   # [s] duration of double-support phases
ss_duration = 0.7   # [s] duration of single-support phases
tube_radius = 0.03  # [m]


# Global variables
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
    get_viewer().SetCamera([
        [0.97248388,  0.01229851, -0.23264533,  2.34433222],
        [0.21414823, -0.44041135,  0.87188209, -2.02105641],
        [-0.0917368, -0.89771186, -0.43092664,  3.40723848],
        [0.,  0.,  0.,  1.]])


def set_camera_1():
    get_viewer().SetCamera([
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
    mpc.target_box.hide()
    preview_buffer.com.set_visible(False)
    robot.set_transparency(0)
    viewer.SetBkgndColor([1, 1, 1])
    dash_graph_handles(fsm.left_foot_traj_handles)
    dash_graph_handles(fsm.right_foot_traj_handles)


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
    update_robot_ik()


class ForceDrawer(Process):

    def __init__(self):
        self.last_bkgnd_switch = None
        self.handles = []

    def on_tick(self, sim):
        """Find supporting contact forces at each COM acceleration update."""
        comdd = preview_buffer.comdd
        gravity = get_gravity()
        wrench = hstack([robot_mass * (comdd - gravity), zeros(3)])
        support = fsm.cur_stance.find_supporting_forces(
            wrench, preview_buffer.com.p, robot_mass, 10.)
        if not support:
            self.handles = []
            viewer.SetBkgndColor([.8, .4, .4])
            self.last_bkgnd_switch = time.time()
        else:
            self.handles = [draw_force(c, fc) for (c, fc) in support]
        if self.last_bkgnd_switch is not None \
                and time.time() - self.last_bkgnd_switch > 0.2:
            # let's keep epilepsy at bay
            viewer.SetBkgndColor([.6, .6, .8])
            self.last_bkgnd_switch = None


class PreviewDrawer(Process):

    def __init__(self):
        self.handles = []

    def on_tick(self, sim):
        com_pre, comd_pre = com.p, com.pd
        com_free, comd_free = com.p, com.pd
        dT = preview_buffer.preview.timestep
        self.handles = []
        self.handles.append(
            draw_point(com.p, color='m', pointsize=0.007))
        for preview_index in xrange(len(preview_buffer.preview.U) / 3):
            com_pre0 = com_pre
            j = 3 * preview_index
            comdd = preview_buffer.preview.U[j:j + 3]
            com_pre = com_pre + comd_pre * dT + comdd * .5 * dT ** 2
            comd_pre += comdd * dT
            color = \
                'b' if preview_index <= preview_buffer.preview.switch_step \
                else 'r'
            self.handles.append(
                draw_point(com_pre, color=color, pointsize=0.005))
            self.handles.append(
                draw_line(com_pre0, com_pre, color=color, linewidth=3))
            com_free0 = com_free
            com_free = com_free + comd_free * dT
            self.handles.append(
                draw_point(com_free, color='g', pointsize=0.005))
            self.handles.append(
                draw_line(com_free0, com_free, color='g', linewidth=3))


class ScreenshotTaker(Process):

    def __init__(self):
        print "Please click on the OpenRAVE window."
        line = os.popen('/usr/bin/xwininfo | grep "Window id:"').readlines()[0]
        window_id = "0x%s" % re.search('0x([0-9a-f]+)', line).group(1)
        self.frame_index = 0
        self.window_id = window_id

    def on_tick(self, sim):
        fname = './recording/camera/%05d.png' % (self.frame_index)
        os.system('import -window %s %s' % (self.window_id, fname))
        self.frame_index += 1


class SEPDrawer(Process):

    def __init__(self):
        self.ss_handles = None
        self.ds_handles = None

    def on_tick(self, sim):
        if fsm.cur_stance.is_single_support:
            ss_stance = fsm.cur_stance
            ds_stance = fsm.next_stance
        else:  # fsm.cur_stance.is_double_support:
            ss_stance = fsm.next_stance
            ds_stance = fsm.cur_stance
        sep_height = preview_buffer.com.z - RobotModel.leg_length
        self.ss_handles = draw_polygon(
            [(x[0], x[1], sep_height) for x in ss_stance.sep],
            normal=[0, 0, 1], color='c')
        self.ds_handles = draw_polygon(
            [(x[0], x[1], sep_height) for x in ds_stance.sep],
            normal=[0, 0, 1], color='y')


class TrajectoryDrawer(Process):

    def __init__(self, body, combined='b-', color=None, linewidth=3,
                 linestyle=None):
        color = color if color is not None else combined[0]
        linestyle = linestyle if linestyle is not None else combined[1]
        assert linestyle in ['-', '.']
        self.body = body
        self.color = color
        self.handles = []
        self.last_pos = body.p
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.parity = True

    def on_tick(self, sim):
        if self.linestyle == '-' or self.parity:
            self.handles.append(draw_line(
                self.last_pos, self.body.p, color=self.color,
                linewidth=self.linewidth))
        self.last_pos = self.body.p
        self.parity = not self.parity


class TubeDrawer(Process):

    def __init__(self):
        self.comdd_handle = []
        self.cone_handles = []
        self.poly_handles = []

    def on_tick(self, sim):
        scale = 0.05
        try:
            self.draw_primal(mpc.tube)
        except Exception as e:
            print "Drawing of polytopes failed: %s" % str(e)
        try:
            self.draw_dual(mpc.tube)
        except Exception as e:
            print "Drawing of dual cones failed: %s" % str(e)
        comdd = scale * preview_buffer.comdd
        self.comdd_handle = [
            draw_line([0, 0, 0], comdd, color='r', linewidth=3),
            draw_points([[0, 0, 0], comdd], color='r', pointsize=0.005)]

    def draw_primal(self, tube):
        self.poly_handles = []
        colors = [(0.5, 0.5, 0., 0.3), (0., 0.5, 0.5, 0.3)]
        if tube.start_stance.is_single_support:
            colors.reverse()
        for (i, vertices) in enumerate(tube.primal_vrep):
            color = colors[i]
            if len(vertices) == 1:
                self.poly_handles.append(
                    draw_point(vertices[0], color=color, pointsize=0.01))
            else:
                self.poly_handles.extend(
                    draw_polyhedron(vertices, '*.-#', color=color))

    def draw_dual(self, tube):
        self.cone_handles = []
        scale = 0.05
        apex = [0., 0., scale * 9.81]
        colors = [(0.5, 0.5, 0., 0.3), (0., 0.5, 0.5, 0.3)]
        if tube.start_stance.is_single_support:
            colors.reverse()
        for (stance_id, cone_vertices) in enumerate(tube.dual_vrep):
            color = colors[stance_id]
            vscale = [scale * array(v) for v in cone_vertices]
            self.cone_handles.extend(draw_3d_cone(
                # recall that cone_vertices[0] is [0, 0, +g]
                apex=apex, axis=[0, 0, 1], section=vscale[1:],
                combined='r-#', color=color))


def set_camera_cones():
    set_camera_top(x=0, y=0, z=0.1 * 9.81)

if __name__ == "__main__":
    seed(42)
    init()
    viewer = get_viewer()
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

    com = PointMass([0, 0, 0], robot_mass)
    preview_buffer = PreviewBuffer(com)
    fsm = StateMachine(
        staircase,
        com,
        'DS-R',
        ss_duration=ss_duration,
        ds_duration=ds_duration,
        init_com_offset=array([0.05, 0., 0.]),
        cyclic=True,
        callback=fsm_callback)
    mpc = TubePreviewControl(
        com,
        fsm,
        preview_buffer,
        nb_mpc_steps=10,
        tube_radius=tube_radius)

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
                'contact': 10000.,
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

    sim = Simulation(dt=dt)
    sim.schedule(fsm)
    sim.schedule(preview_buffer)
    sim.schedule(mpc)
    robot.start_ik_thread(dt)

    com_traj_drawer = TrajectoryDrawer(com, 'b-')
    forces_drawer = ForceDrawer()
    left_foot_traj_drawer = TrajectoryDrawer(robot.left_foot, 'g.')
    preview_drawer = PreviewDrawer()
    right_foot_traj_drawer = TrajectoryDrawer(robot.right_foot, 'r.')
    screenshots = None
    screenshots = ScreenshotTaker()
    sep_drawer = SEPDrawer()
    tube_drawer = TubeDrawer()
    sim.schedule_extra(com_traj_drawer)
    sim.schedule_extra(forces_drawer)
    sim.schedule_extra(left_foot_traj_drawer)
    sim.schedule_extra(preview_drawer)
    sim.schedule_extra(right_foot_traj_drawer)
    if screenshots:
        sim.schedule_extra(screenshots)
    sim.schedule_extra(sep_drawer)
    sim.schedule_extra(tube_drawer)

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
