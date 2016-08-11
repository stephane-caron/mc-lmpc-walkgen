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

from numpy import array, cross, dot, float64, hstack, ones, sqrt, vstack, zeros
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
        vertices1d = compute_polygon_hull(B2, ones(len(c)))
    except QhullError:
        warn("QhullError: maybe output polygon was empty?")
        return []

    def vertices_at(z):
        v = [array([a * (g - z), b * (g - z)]) for (a, b) in vertices1d]
        return [array([x, y, z]) for (x, y) in v]

    return [array([0, 0, g])] + vertices_at(z=-g)


class COMTube(object):

    LINE = 2
    # DIAMOND = 6
    PARALLELEPIPED = 8

    SHAPES = [LINE, PARALLELEPIPED]

    def __init__(self, start_com, target_com, start_stance, target_stance,
                 shape, radius, start_margin=True, end_margin=True):
        """
        Create a new COM trajectory tube.

        INPUT:

        - ``start_com`` -- start position of the COM
        - ``target_com`` -- end position of the COM
        - ``start_stance`` -- stance used to compute the contact wrench cone
        - ``shape`` -- number of vertices of the tube (2, 6 or 8)
        - ``radius`` -- side of the cross-section square (for ``shape`` > 2)
        - ``start_margin`` -- safety margin at start
        - ``end_margin`` -- safety margin at end

        .. NOTE::

            We are assuming that self.start_stance applies to all vertices of
            the tube. See the paper for a discussion of this technical choice.
        """
        assert shape in COMTube.SHAPES
        self._cone_vertices = {}
        self._vertices = {}
        self.delta = target_com - start_com
        self.end_margin = end_margin
        self.radius = radius
        self.shape = shape
        self.single_polytope = False
        self.start_com = start_com
        self.start_margin = start_margin
        self.start_stance = start_stance
        self.target_com = target_com
        self.target_stance = target_stance

        self.compute_primal_vrep()

    """
    Primal polytopes
    ================
    """

    def compute_primal_vrep(self):
        if dot(self.delta, self.delta) < 1e-6:
            return [self.start_com]
        n = normalize(self.delta)
        t = array([0., 0., 1.])
        t -= dot(t, n) * n
        t = normalize(t)
        b = cross(n, t)
        if self.shape == COMTube.PARALLELEPIPED:
            cross_section = [dx * t + dy * b for (dx, dy) in [
                (+self.radius, +self.radius),
                (+self.radius, -self.radius),
                (-self.radius, +self.radius),
                (-self.radius, -self.radius)]]
        else:  # self.shape == COMTube.LINE:
            cross_section = [zeros(3)]
        tube_start = self.start_com
        safety_margin = 0.02  # [m]
        if self.start_margin:
            tube_start -= safety_margin * n
        tube_end = self.target_com
        if self.end_margin:
            tube_end += safety_margin * n
        if self.start_stance.is_single_support:
            sep = self.start_stance.sep
        else:  # self.target_stance.is_single_support:
            sep = self.target_stance.sep
        vertices0, vertices1 = [], []
        for s in cross_section:
            start_vertex = tube_start + s
            end_vertex = tube_end + s
            # NB: the order in the intersection matters, see function doc
            mid_vertex = intersect_line_cylinder(end_vertex, start_vertex, sep)
            if mid_vertex is None:
                # assuming that start_vertex is in the SEP, no intersection
                # means that both COM are inside the polygon
                vertices = (
                    [tube_start + s for s in cross_section] +
                    [tube_end + s for s in cross_section])
                self._single_polytope = True
                self._vertices = {0: vertices, 1: vertices}
                return
            if self.start_stance.is_single_support:
                mid_vertex = start_vertex + 0.95 * (mid_vertex - start_vertex)
            else:  # self.target_stance.is_single_support
                mid_vertex = end_vertex + 0.95 * (mid_vertex - end_vertex)
            vertices0.append(start_vertex)
            vertices0.append(mid_vertex)
            vertices1.append(mid_vertex)
            vertices1.append(end_vertex)
        self._single_polytope = False
        self._vertices = {0: vertices0, 1: vertices1}

    def compute_polytope_center(self, stance_id):
        V = array(self._vertices[stance_id])
        n = len(self._vertices[stance_id])
        return V.sum(axis=0) / n

    def compute_primal_hrep(self, stance_id):
        """
        Compute the primal representation of the tube, i.e. the H-representation
        of its Euclidean polytope.

        INPUT:

        - ``stance_id`` -- 0 for start stance_id, 1 for end stance_id

        OUTPUT:

        A tuple (A, b) such that the H-representation is A * x <= b.
        """
        if len(self._vertices) == 1:
            return Polytope.hrep(self._vertices[0])
        return Polytope.hrep(self._vertices[stance_id])

    def draw_primal_polytopes(self):
        """
        Draw polytopes for each stance.

        OUTPUT:

        GUI handles.
        """
        handles = []
        colors = ['c', 'y']
        for (stance_id, vlist) in self._vertices.iteritems():
            if stance_id > 0 and self._single_polytope:
                break
            elif len(vlist) == 2:
                handles.extend(draw_line(
                    vlist[0], vlist[1], color=[0., 0.5, 0.5], linewidth=5))
            else:  # should be a full polytope
                color = colors[stance_id]
                handles.extend(draw_polyhedron(vlist, '%c.-#' % color))
        return handles

    """
    Dual cone
    =========
    """

    def compute_dual_vrep(self, stance_id):
        """
        Compute vertices of dual COM acceleration cones.

        INPUT:

        - ``stance_id`` -- 0 for start stance_id, 1 for end stance_id

        OUTPUT:

        Dual cone in vertex representation. The first vertex is the apex [0, 0,
        +g], while the following ones give the polygon of the cross section at
        zdd = -g.
        """
        if stance_id in self._cone_vertices:
            return self._cone_vertices[stance_id]
        stance = self.start_stance if stance_id == 0 else self.target_stance
        A_O = stance.cwc  # CWC at world origin
        gravity = pymanoid.get_gravity()
        B_list, c_list = [], []
        for (i, v) in enumerate(self._vertices[stance_id]):
            B = A_O[:, :3] + cross(A_O[:, 3:], v)
            c = dot(B, gravity)
            B_list.append(B)
            c_list.append(c)
        B = vstack(B_list)
        c = hstack(c_list)
        self._cone_vertices[stance_id] = reduce_polar_system(B, c)
        return self._cone_vertices[stance_id]

    def compute_dual_hrep(self, stance_id):
        """
        Compute the halfspace representation of dual COM acceleration cones.

        INPUT:

        - ``stance_id`` -- 0 for start stance_id, 1 for end stance_id

        OUTPUT:

        Matrix of the dual cone halfspace representation.
        """
        cone_vertices = self.compute_dual_vrep(stance_id)
        try:
            B_new, c_new = Polytope.hrep(cone_vertices)
            B, c = (B_new.astype(float64), c_new.astype(float64))
            return (B, c)
        except:
            warn("Polytope.hrep(cone_vertices) failed")
            return None

    def draw_dual_cones(self, scale=0.1):
        handles = []
        for stance_id in [0, 1]:
            apex = self.compute_polytope_center(stance_id)
            cone_vertices = self.compute_dual_vrep(stance_id)
            vscale = [apex + scale * array(v) for v in cone_vertices]
            handles.extend(draw_3d_cone(
                # recall that cone_vertices[0] is [0, 0, +g]
                apex=apex, axis=[0, 0, 1], section=vscale[1:]))
        return handles
