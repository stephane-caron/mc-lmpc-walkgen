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
from mpc3d.tube import compute_com_tubes
from mpc3d.fsm import StanceFSM
from mpc3d.polygons import intersect_line_cylinder
from walk import generate_staircase, set_camera_0


gui_handles = {}
tube_shape = 8
tube_size = 0.04


def draw_tube_thread():
    handles = []
    mid_com_point = PointMass([0, 0, 0], 10)
    while True:
        ss_stance = \
            fsm.cur_stance if fsm.cur_stance.is_single_support \
            else fsm.next_stance
        mid_com = intersect_line_cylinder(start_com.p, end_com.p, ss_stance.sep)
        if mid_com is not None:
            mid_com_point.set_pos(mid_com)
            mid_com_point.set_visible(True)
        else:
            mid_com_point.set_visible(False)
        tube1, tube2 = compute_com_tubes(
            start_com.p, end_com.p, fsm.cur_stance, fsm.next_stance,
            tube_shape, tube_size)
        if tube1 == tube2:
            handles = [
                tube1.draw_primal_polytope(color='c'),
                tube1.draw_dual_cone()]
        else:
            handles = [
                tube1.draw_primal_polytope(color='c'),
                tube1.draw_dual_cone(),
                tube2.draw_primal_polytope(color='m'),
                tube2.draw_dual_cone()]
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
        PointMass([0, 0, 0], 39., visible=False), 'DS-R',
        ss_duration=1.0,
        ds_duration=0.5,
        init_com_offset=array([0.05, 0., 0.]),
        cyclic=True)
    start_com = PointMass([0, 0, 0], 39, color='b')
    end_com = PointMass([0, 0, 0], 39, color='g')

    def next_step():
        global cur_step, stance
        if fsm.cur_stance.is_double_support:
            staircase[cur_step].set_transparency(0.5)
            staircase[cur_step].set_color('r')
            cur_step += 1
        else:
            staircase[(cur_step + 1) % len(staircase)].set_transparency(0)
            staircase[(cur_step + 1) % len(staircase)].set_color('g')
        fsm.step()
        _, com_target = fsm.get_preview_targets()
        if fsm.cur_stance.is_single_support:
            start_com.set_pos(fsm.cur_stance.com)
        end_com.set_pos(com_target)
        ss_stance = \
            fsm.cur_stance if fsm.cur_stance.is_single_support \
            else fsm.next_stance
        vertices = ss_stance.sep
        gui_handles['static'] = draw_polygon(
            [(x[0], x[1], start_com.z) for x in vertices],
            normal=[0, 0, 1])

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
