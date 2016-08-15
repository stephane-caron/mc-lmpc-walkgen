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
import time

from numpy import array, cross, dot, float64, hstack, ones, sqrt, vstack
from polygons import compute_polygon_hull, intersect_line_cylinder
from pymanoid.polyhedra import Polytope
from scipy.spatial.qhull import QhullError


class TubeError(Exception):
    pass


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

    vertices2d = compute_polygon_hull(B2, ones(len(c)))

    def vertices_at(z):
        v = [array([a * (g - z), b * (g - z)]) for (a, b) in vertices2d]
        return [array([x, y, z]) for (x, y) in v]

    return [array([0, 0, g])] + vertices_at(z=-g)


class COMTube(object):

    def __init__(self, start_com, target_com, start_stance, next_stance, radius,
                 safety_margin=0.02):
        """
        Create a new COM trajectory tube.

        INPUT:

        - ``start_com`` -- start position of the COM
        - ``target_com`` -- end position of the COM
        - ``start_stance`` -- stance used to compute the contact wrench cone
        - ``radius`` -- side of the cross-section square (for ``shape`` > 2)
        - ``safety_margin`` -- safety margin (in [m]) around start and end COM
                               positions (default: 0.02)

        .. NOTE::

            We are assuming that self.start_stance applies to all vertices of
            the tube. See the paper for a discussion of this technical choice.
        """
        self.dual_hrep = []
        self.dual_vrep = []
        self.next_stance = next_stance
        self.primal_hrep = []
        self.primal_vrep = []
        self.radius = radius
        self.safety_margin = safety_margin
        self.start_com = start_com
        self.start_stance = start_stance
        self.target_com = target_com

        # all other computations depend on the primal V-rep:
        self.compute_primal_vrep()
        self.compute_primal_hrep()
        self.compute_dual_vrep()
        self.compute_dual_hrep()

    """
    Primal polytopes
    ================
    """

    def compute_primal_vrep(self):
        t0 = time.time()
        delta = self.target_com - self.start_com
        n = normalize(delta)
        t = array([0., 0., 1.])
        t -= dot(t, n) * n
        t = normalize(t)
        b = cross(n, t)
        cross_section = [dx * t + dy * b for (dx, dy) in [
            (+self.radius, +self.radius),
            (+self.radius, -self.radius),
            (-self.radius, +self.radius),
            (-self.radius, -self.radius)]]
        tube_start = self.start_com - self.safety_margin * n
        tube_end = self.target_com + self.safety_margin * n
        if self.start_stance.is_single_support:
            if False \
                    and all(abs(self.next_stance.com - self.target_com) < 1e-3):
                # we are at the end of an SS phase
                vertices = (
                    [tube_start + s for s in cross_section] +
                    [tube_end + s for s in cross_section])
                self.primal_vrep = [vertices]
                print "compute_primal_vrep(): %.1f ms" % (
                    1000. * (time.time() - t0))
                return
            sep = self.start_stance.sep
        else:  # self.start_stance.is_double_support
            sep = self.next_stance.sep
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
                if self.start_stance.is_single_support:
                    # we are at the end of an SS phase
                    self.primal_vrep = [vertices]
                else:  # self.start_stance.is_double_support
                    # we are in DS but polytope is included in the next SS-SEP
                    self.primal_vrep = [vertices, vertices]
                print "compute_primal_vrep(): %.1f ms" % (
                    1000. * (time.time() - t0))
                return
            if self.start_stance.is_single_support:
                mid_vertex = start_vertex + 0.95 * (mid_vertex - start_vertex)
                vertices0.append(start_vertex)
                vertices0.append(mid_vertex)
                vertices1.append(start_vertex)
                vertices1.append(end_vertex)
            else:  # self.start_stance.is_double_support
                mid_vertex = end_vertex + 0.95 * (mid_vertex - end_vertex)
                vertices0.append(start_vertex)
                vertices0.append(end_vertex)
                vertices1.append(mid_vertex)
                vertices1.append(end_vertex)
        self.primal_vrep = [vertices0, vertices1]
        print "compute_primal_vrep(): %.1f ms" % (1000. * (time.time() - t0))

    def compute_primal_hrep(self):
        t0 = time.time()
        for (stance_id, vertices) in enumerate(self.primal_vrep):
            try:
                self.primal_hrep.append(Polytope.hrep(vertices))
            except RuntimeError as e:
                raise TubeError("Could not compute primal hrep: %s" % str(e))
        print "compute_primal_hrep(): %.1f ms" % (1000. * (time.time() - t0))

    def compute_dual_vrep(self):
        t0 = time.time()
        gravity = pymanoid.get_gravity()
        for (stance_id, vertices) in enumerate(self.primal_vrep):
            if stance_id == 0:
                A_O = self.start_stance.cwc
            else:  # stance_id == 1
                A_O = self.next_stance.cwc
            B_list, c_list = [], []
            for (i, v) in enumerate(vertices):
                B = A_O[:, :3] + cross(A_O[:, 3:], v)
                c = dot(B, gravity)
                B_list.append(B)
                c_list.append(c)
            B = vstack(B_list)
            c = hstack(c_list)
            try:
                cone_vertices = reduce_polar_system(B, c)
                self.dual_vrep.append(cone_vertices)
            except QhullError:
                raise TubeError("Cannot reduce polar of stance %d" % stance_id)
        print "compute_dual_vrep(): %.1f ms" % (1000. * (time.time() - t0))

    def compute_dual_hrep(self):
        t0 = time.time()
        for (stance_id, cone_vertices) in enumerate(self.dual_vrep):
            # cone_vertices = self.compute_dual_vrep(stance_id)
            B_new, c_new = Polytope.hrep(cone_vertices)
            B, c = (B_new.astype(float64), c_new.astype(float64))
            self.dual_hrep.append((B, c))
        print "compute_dual_hrep(): %.1f ms" % (1000. * (time.time() - t0))

    def contains(self, com):
        E, f = self.primal_hrep[0]
        return all(dot(E, com) <= f)
