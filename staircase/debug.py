#!/usr/bin/env python

import os
import sys
import threading

script_path = os.path.realpath(__file__)
# script_name = os.path.basename(script_path)
sys.path.append(os.path.dirname(script_path) + '/..')
sys.path.append(os.path.dirname(script_path) + '/../pymanoid')

import pymanoid
from numpy import array
from pymanoid import PointMass, draw_polygon
from mpc3d.tube import COMTube, TubeError
from mpc3d.fsm import StanceFSM
from walk import generate_staircase, set_camera_0


gui_handles = {}
mass = 20.  # [kg], used for drawing scale
sep_height = 0.
tube_shape = 8
tube_radius = 0.04


def draw_tube_thread():
    handles = []
    while True:
        try:
            tube = COMTube(
                start_com.p, end_com.p, fsm.cur_stance, fsm.next_stance,
                tube_shape, tube_radius)
            handles = [
                tube.draw_primal_polytopes(),
                tube.draw_dual_cones()]
        except TubeError as e:
            print "TubeError:", e
        start_com_sep.set_pos([start_com.x, start_com.y, sep_height])
        end_com_sep.set_pos([end_com.x, end_com.y, sep_height])
    return handles


if __name__ == "__main__":
    pymanoid.init()
    set_camera_0()
    staircase = generate_staircase(
        radius=1.4,
        angular_step=0.5,
        height=1.4,
        roughness=0.5,
        friction=0.7,
        step_dim_x=0.2,
        step_dim_y=0.1)
    for contact in staircase:
        contact.set_transparency(0.5)
    fsm = StanceFSM(
        staircase,
        PointMass([0, 0, 0], mass, visible=False), 'DS-R',
        ss_duration=1.0,
        ds_duration=0.5,
        init_com_offset=array([0.05, 0., 0.]),
        cyclic=True)
    start_com = PointMass([0, 0, 0], mass, color='b')
    end_com = PointMass([0, 0, 0], mass, color='g')
    start_com_sep = PointMass([0, 0, 0], mass / 2., color='b')
    end_com_sep = PointMass([0, 0, 0], mass / 2., color='g')

    def next_step():
        global cur_step, stance, sep_height
        if fsm.cur_stance.is_double_support:
            staircase[cur_step].set_transparency(0.5)
            staircase[cur_step].set_color('r')
            cur_step += 1
        else:
            staircase[(cur_step + 1) % len(staircase)].set_transparency(0)
            staircase[(cur_step + 1) % len(staircase)].set_color('g')
        fsm.step()
        _, _, com_target = fsm.get_preview_targets()
        if fsm.cur_stance.is_single_support:
            start_com.set_pos(fsm.cur_stance.com)
        end_com.set_pos(com_target)
        if fsm.cur_stance.is_single_support:
            ss_stance = fsm.cur_stance
            ds_stance = fsm.next_stance
        else:  # fsm.cur_stance.is_double_support:
            ss_stance = fsm.next_stance
            ds_stance = fsm.cur_stance
        sep_height = start_com.z - 0.2
        gui_handles['static-ss'] = draw_polygon(
            [(x[0], x[1], sep_height) for x in ss_stance.sep],
            normal=[0, 0, 1], color='c')
        gui_handles['static-ds'] = draw_polygon(
            [(x[0], x[1], sep_height) for x in ds_stance.sep],
            normal=[0, 0, 1], color='y')

    cur_step = 0
    staircase[cur_step].set_transparency(0)
    staircase[cur_step].set_color('g')
    staircase[cur_step + 1].set_transparency(0)  # starting in DS-R
    staircase[cur_step + 1].set_color('g')
    next_step()

    t = threading.Thread(target=draw_tube_thread, args=())
    t.daemon = True
    t.start()

    print ""
    print "=================================================================="
    print ""
    print "Only showing preview COM polytopes and their acceleration cones."
    print ""
    print "Call ``next_step()`` to cycle through the state machine."
    print ""
    print "=================================================================="
    print ""

    import IPython
    IPython.embed()
