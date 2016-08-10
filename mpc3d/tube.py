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

import pymanoid

from numpy import array, cross, dot, float64, hstack, ones, sqrt, vstack
from polygons import compute_polygon_hull, intersect_line_cylinder
from pymanoid.draw import draw_3d_cone, draw_line, draw_polyhedron
from pymanoid.polyhedra import Polytope
from scipy.spatial.qhull import QhullError
from warnings import warn


def normalize(v):
    return v / sqrt(dot(v, v))


def reduce_polar_system(B, c):
    gravity = pymanoid.get_gravity()
    g = -gravity[2]
    assert g > 0
    # assert all(c > 0), "c > 0 assertion failed"
    # assert all(B[:, 2] < 0)
    assert all(abs(c) > 1e-10)
    check = c / B[:, 2]
    assert max(check) - min(check) < 1e-10, "max - min failed (%.1e)" % (
        (max(check) - min(check)))
    assert abs(check[0] - (-g)) < 1e-10, "check is not -g?"
    sigma = c / g
    B2 = hstack([
        (B[:, column] / sigma).reshape((B.shape[0], 1))
        for column in [0, 1]])

    try:
        vertices2d = compute_polygon_hull(B2, ones(len(c)))
    except QhullError:
        warn("QhullError: maybe output polygon was empty?")
        return []

    def vertices_at(z):
        v = [array([a * (g - z), b * (g - z)]) for (a, b) in vertices2d]
        return [array([x, y, z]) for (x, y) in v]

    return [array([0, 0, g])] + vertices_at(z=-g)


class COMTube(object):

    LINE = 2
    DIAMOND = 6
    PARALLELEPIPED = 8

    SHAPES = [LINE, DIAMOND, PARALLELEPIPED]

    def __init__(self, init_com, target_com, init_stance, shape, size,
                 start_margin=True, end_margin=True):
        """
        Create a new COM trajectory tube.

        INPUT:

        - ``init_com`` -- start position of the COM
        - ``target_com`` -- end position of the COM
        - ``init_stance`` -- stance used to compute the contact wrench cone
        - ``shape`` -- number of vertices of the tube (2, 6 or 8)
        - ``size`` -- side of the cross-section square (for ``shape`` > 2)
        - ``start_margin`` -- safety margin at start
        - ``end_margin`` -- safety margin at end

        .. NOTE::

            We are assuming that self.init_stance applies to all vertices of the
            tube. See the paper for a discussion of this technical choice.
        """
        assert shape in COMTube.SHAPES
        self._cone_vertices = None
        self._vertices = None
        self.delta = target_com - init_com
        self.end_margin = end_margin
        self.init_com = init_com
        self.init_stance = init_stance
        self.shape = shape
        self.size = size
        self.start_margin = start_margin
        self.target_com = target_com

    """
    Primal polytope
    ===============
    """

    @property
    def vertices(self):
        if self._vertices is None:
            self._vertices = self.compute_primal_vrep()
        return self._vertices

    @property
    def nb_vertices(self):
        return len(self.vertices)

    def compute_primal_vrep(self):
        if dot(self.delta, self.delta) < 1e-6:
            return [self.init_com]
        n = normalize(self.delta)
        t = array([0., 0., 1.])
        t -= dot(t, n) * n
        t = normalize(t)
        b = cross(n, t)
        square = [dx * t + dy * b for (dx, dy) in [
            (+self.size, +self.size),
            (+self.size, -self.size),
            (-self.size, +self.size),
            (-self.size, -self.size)]]
        tube_start = self.init_com
        if self.start_margin:
            tube_start -= 0.05 * self.delta
        tube_end = self.target_com
        if self.end_margin:
            tube_end += 0.05 * self.delta
        if self.shape == COMTube.PARALLELEPIPED:
            vertices = \
                [tube_start + s for s in square] + \
                [tube_end + s for s in square]
        elif self.shape == COMTube.DIAMOND:
            tube_mid = 0.5 * tube_start + 0.5 * tube_end
            vertices = [tube_start] + \
                [tube_mid + s for s in square] + [tube_end]
        else:  # default is COMTube.LINE
            vertices = [tube_start, tube_end]
        return vertices

    def compute_primal_hrep(self):
        """
        Compute the primal representation of the tube, i.e. the H-representation
        of its Euclidean polytope.

        OUTPUT:

        A tuple (A, b) such that the H-representation is A * x <= b.
        """
        return Polytope.hrep(self.vertices)

    def draw_primal_polytope(self, color='c'):
        if not self.vertices:
            return None
        elif len(self.vertices) == 2:
            return draw_line(
                self.vertices[0], self.vertices[1], color=[0., 0.5, 0.5],
                linewidth=5)
        return draw_polyhedron(self.vertices, '%c.-#' % color)

    """
    Dual cone
    =========
    """

    def compute_dual_vrep(self):
        A_O = self.init_stance.cwc  # CWC at world origin
        gravity = pymanoid.get_gravity()
        B_list, c_list = [], []
        for (i, v) in enumerate(self.vertices):
            B = A_O[:, :3] + cross(A_O[:, 3:], v)
            c = dot(B, gravity)
            B_list.append(B)
            c_list.append(c)
        B = vstack(B_list)
        c = hstack(c_list)
        return reduce_polar_system(B, c)

    def compute_dual_hrep(self):
        if not self._cone_vertices:
            self._cone_vertices = self.compute_dual_vrep()
        try:
            B_new, c_new = Polytope.hrep(self._cone_vertices)
            B, c = (B_new.astype(float64), c_new.astype(float64))
            return (B, c)
        except:
            warn("Polytope.hrep(cone_vertices) failed")
            return None

    def draw_dual_cone(self, scale=0.1):
        if not self._cone_vertices:
            self._cone_vertices = self.compute_dual_vrep()
        apex = self.target_com
        vscale = [apex + scale * array(v) for v in self._cone_vertices]
        return draw_3d_cone(  # recall that self.cone_vertices[0] is [0, 0, +g]
            apex=apex, axis=[0, 0, 1], section=vscale[1:])


def compute_com_tubes(cur_com, target_com, cur_stance, next_stance, shape,
                      size):
    ss_stance = cur_stance if cur_stance.is_single_support else next_stance
    mid_com = intersect_line_cylinder(cur_com, target_com, ss_stance.sep)
    if mid_com is None:
        # assuming cur_com is in the SEP, no intersection means that both COM
        # are inside the polygon
        tube = COMTube(cur_com, target_com, cur_stance, shape, size)
        return (tube, tube)
    u = normalize(target_com - cur_com)
    epsilon = (-1 if cur_stance.is_single_support else +1) * 1.5 * size
    mid_com += epsilon * u  # heuristic so that the SS tube lies inside the SEP
    tube1 = COMTube(cur_com, mid_com, cur_stance, shape, size, end_margin=False)
    tube2 = COMTube(mid_com, target_com, next_stance, shape, size,
                    start_margin=False)
    return (tube1, tube2)
