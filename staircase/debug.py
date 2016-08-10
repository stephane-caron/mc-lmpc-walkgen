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
from pymanoid import PointMass
from mpc3d.tube import TrajectoryTube
from mpc3d.fsm import StanceFSM
from walk import generate_staircase, set_camera_0


def draw_tube_thread():
    handles = []
    while True:
        try:
            tube = TrajectoryTube(
                start_com.p, end_com.p, fsm.cur_stance, 6)
            handles = [
                tube.draw_primal_polytope(),
                tube.draw_dual_cone()]
        except Exception as e:
            print e
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
        if not fsm.cur_stance.is_double_support:
            start_com.set_pos(fsm.cur_stance.com)
            delta = fsm.next_ss_stance.com - start_com.p
            end_com.set_pos(start_com.p + 0.05 * delta)
        else:
            end_com.set_pos(fsm.next_stance.com)

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
