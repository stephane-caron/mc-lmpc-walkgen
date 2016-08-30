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

import cdd
import os.path
import sys

try:
    import pymanoid
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../pymanoid')
    import pymanoid

try:
    from wpg.cwc import compute_cwc_pyparma
    from wpg.polygons import compute_polygon_hull
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/..')
    from wpg.cwc import compute_cwc_pyparma
    from wpg.polygons import compute_polygon_hull

from numpy import array, dot, hstack, vstack, zeros
from projection import project_polytope_bretl


def compute_static_equilibrium_lp(contact_set):
    mass = 39  # [kg]
    p = [0, 0, 0]  # point where contact wrench is taken at
    G = contact_set.compute_grasp_matrix(p)
    F = contact_set.compute_stacked_wrench_cones()
    A = F
    b = zeros(A.shape[0])
    C = G[(0, 1, 2, 5), :]
    d = array([0, 0, mass * 9.81, 0])
    E = 1. / (mass * 9.81) * vstack([-G[4, :], +G[3, :]])
    f = array([p[0], p[1]])
    return (A, b, C, d, E, f)


def compute_static_polygon_bretl(contact_set):
    A, b, C, d, E, f = compute_static_equilibrium_lp(contact_set)
    vertices, _ = project_polytope_bretl(A, b, C, d, E, f)
    return vertices


def compute_static_polygon_cdd_hull(contact_set, p=[0, 0, 0]):
    A_O = contact_set.compute_wrench_cone(p)
    k, a_Oz, a_x, a_y = A_O.shape[0], A_O[:, 2], A_O[:, 3], A_O[:, 4]
    B, c = hstack([-a_y.reshape((k, 1)), +a_x.reshape((k, 1))]), -a_Oz
    return compute_polygon_hull(B, c)


def compute_static_polygon_pyparma_hull(contact_set, p=[0, 0, 0]):
    A_O = compute_cwc_pyparma(contact_set, p)
    k, a_Oz, a_x, a_y = A_O.shape[0], A_O[:, 2], A_O[:, 3], A_O[:, 4]
    B, c = hstack([-a_y.reshape((k, 1)), +a_x.reshape((k, 1))]), -a_Oz
    return compute_polygon_hull(B, c)


def compute_static_polygon_cdd_only(contact_set, mass):
    """
    Compute the static-equilibrium area of the center of mass.

    INPUT:

    - ``mass`` -- total mass of the robot

    .. NOTE::

        See http://dx.doi.org/10.1109/TRO.2008.2001360 for details. The
        implementation here uses the double-description method (cdd) rather
        than the equality-set projection algorithm, as described in
        http://arxiv.org/abs/1510.03232.
    """
    G = contact_set.compute_grasp_matrix([0, 0, 0])
    F = contact_set.compute_stacked_wrench_cones()
    b = zeros((F.shape[0], 1))
    # the input [b, -F] to cdd.Matrix represents (b - F x >= 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    M = cdd.Matrix(hstack([b, -F]), number_type='float')
    M.rep_type = cdd.RepType.INEQUALITY

    # Equalities:  C [GAW_1 GAW_2 ...] + d == 0
    C = G[(0, 1, 2, 5), :]
    d = array([0, 0, mass * 9.81, 0])
    # the input [d, -C] to cdd.Matrix.extend represents (d - C x == 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    M.extend(hstack([d.reshape((4, 1)), -C]), linear=True)

    # Convert from H- to V-representation
    # M.canonicalize()
    P = cdd.Polyhedron(M)
    V = array(P.get_generators())
    if V.shape[0] < 1:
        return [], []

    # COM position from GAW:  [pGx, pGy] = D * [GAW_1 GAW_2 ...]
    D = 1. / (mass * 9.81) * vstack([-G[4, :], +G[3, :]])
    vertices = []
    for i in xrange(V.shape[0]):
        # assert V[i, 0] == 1, "There should be no ray in this polygon"
        p = dot(D, V[i, 1:])
        vertices.append([p[0], p[1]])
    return vertices


def draw_static_polygon(contact_set, p=[0, 0, 0], color=None, combined='g-#'):
    color = color if color else (0., 0.5, 0., 0.5)
    vertices = compute_static_polygon_cdd_hull(contact_set, p)
    return pymanoid.draw_polygon(
        [(x[0] + p[0], x[1] + p[1], p[2]) for x in vertices],
        normal=[0, 0, 1], color=color, combined=combined)
