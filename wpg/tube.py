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

# from numpy import float64  # if using pyparma
from numpy import array, cross, dot, hstack, ones, vstack
from polygons import compute_polygon_hull, intersect_line_cylinder
from scipy.spatial.qhull import QhullError

from polygons import intersect_polygons
from pymanoid.misc import normalize
from pymanoid.polyhedra import Polytope


class TubeError(Exception):
    pass


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

    return compute_polygon_hull(B2, ones(len(c)))


def polar_to_polytope(vertices2d):
    gravity = pymanoid.get_gravity()
    g = -gravity[2]

    def vertices_at(z):
        v = [array([a * (g + z), b * (g + z)]) for (a, b) in vertices2d]
        return [array([x, y, z]) for (x, y) in v]

    return [array([0, 0, -g])] + vertices_at(z=+g)


class COMTube(object):

    """
    When there is an SS-to-DS contact switch, this strategy computes one primal
    tube and two dual intersection cones.

    The primal tube is, as described in the paper, a parallelepiped containing
    both the COM current and target locations. Its dual cone is used during the
    DS phase. The dual cone for the SS phase is calculated by intersecting the
    latter with the dual cone of the current COM position in single-contact.
    """

    def __init__(self, start_com, target_com, start_stance, next_stance, radius,
                 margin=0.01):
        """
        Create a new COM trajectory tube.

        INPUT:

        - ``start_com`` -- start position of the COM
        - ``target_com`` -- end position of the COM
        - ``start_stance`` -- stance used to compute the contact wrench cone
        - ``radius`` -- side of the cross-section square (for ``shape`` > 2)
        - ``margin`` -- safety margin (in [m]) before/after start/end COM
                        positions (default: 1 [cm])
        """
        self.comp_times = []
        self.dual_hrep = []
        self.dual_vrep = []
        self.margin = margin
        self.next_stance = next_stance
        self.primal_hrep = []
        self.primal_vrep = []
        self.radius = radius
        self.start_com = start_com
        self.start_stance = start_stance
        self.target_com = target_com
        self.compute_double_description()

    def compute_double_description(self):
        """Compute primal and dual H-rep and V-reps."""
        t0 = time.time()
        self.compute_primal_vrep()
        t1 = time.time()
        self.compute_primal_hrep()
        t2 = time.time()
        self.compute_dual_vrep()
        t3 = time.time()
        self.compute_dual_hrep()
        t4 = time.time()
        self.comp_times = [t1 - t0, t2 - t1, t3 - t2, t4 - t3]

    def compute_primal_vrep(self):
        """Compute vertices of the primal tube."""
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
        tube_start = self.start_com - self.margin * n
        tube_end = self.target_com + self.margin * n
        vertices = (
            [tube_start + s for s in cross_section] +
            [tube_end + s for s in cross_section])
        self.full_vrep = vertices
        if self.start_stance.is_single_support:
            if all(abs(self.start_stance.com - self.target_com) < 1e-3):
                self.primal_vrep = [vertices]
            else:  # beginning of SS phase
                self.primal_vrep = [
                    [self.start_com],  # single-support
                    vertices]          # ensuing double-support
        else:  # self.start_stance.is_double_support
            self.primal_vrep = [
                vertices,             # double-support
                [self.target_com]]    # final single-support

    def compute_primal_hrep(self):
        """
        Compute halfspaces of the primal tube.

        NB: not optimized, we simply call cdd here.
        """
        try:
            self.full_hrep = (Polytope.hrep(self.full_vrep))
        except RuntimeError as e:
            raise TubeError("Could not compute primal hrep: %s" % str(e))

    def compute_dual_vrep(self):
        """Compute vertices of the dual cones."""
        gravity = pymanoid.get_gravity()

        def compute_stance_dual(stance_id, primal_vertices):
            stance = self.start_stance if stance_id == 0 else self.next_stance
            A_O = stance.cwc
            B_list, c_list = [], []
            for (i, v) in enumerate(primal_vertices):
                B = A_O[:, :3] + cross(A_O[:, 3:], v)
                c = dot(B, gravity)
                B_list.append(B)
                c_list.append(c)
            B = vstack(B_list)
            c = hstack(c_list)
            try:
                return reduce_polar_system(B, c)
            except QhullError:
                raise TubeError("Cannot reduce polar of stance %d" % stance_id)

        if len(self.primal_vrep) == 1:
            vertices = compute_stance_dual(0, self.primal_vrep[0])
            self.dual_vrep = [polar_to_polytope(vertices)]
        else:  # len(self.primal_vrep) == 2
            ss_id, ds_id = (1, 0) if len(self.primal_vrep[0]) > 1 else (0, 1)
            ds_vertices = compute_stance_dual(ds_id, self.full_vrep)
            ss_vertices = compute_stance_dual(ss_id, self.primal_vrep[ss_id])
            ss_vertices = intersect_polygons(ds_vertices, ss_vertices)
            self.dual_vrep = [
                polar_to_polytope(ss_vertices),
                polar_to_polytope(ds_vertices)]
            if ss_id == 1:
                self.dual_vrep.reverse()

    def compute_dual_hrep(self):
        """
        Compute halfspaces of the dual cones.

        NB: not optimized, we simply call cdd here.
        """
        for (stance_id, cone_vertices) in enumerate(self.dual_vrep):
            B, c = Polytope.hrep(cone_vertices)
            # B, c = (B.astype(float64), c.astype(float64))  # if using pyparma
            self.dual_hrep.append((B, c))


class DoubleCOMTube(COMTube):

    """
    In this strategy, two eight-vertex tubes are computed: the double-support
    one is the same as in COMTube, while the single-support one is equal to the
    intersection of the latter with the static-equilibrium cylinder.
    """

    def compute_primal_vrep(self):
        """Compute vertices for both primal tubes."""
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
        tube_start = self.start_com - self.margin * n
        tube_end = self.target_com + self.margin * n
        if self.start_stance.is_single_support:
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

    def compute_primal_hrep(self):
        """
        Compute halfspaces of the primal tubes (contrary to COMTube, there are
        two tubes here).

        NB: not optimized, we simply call cdd here.
        """
        for (stance_id, vertices) in enumerate(self.primal_vrep):
            try:
                self.primal_hrep.append(Polytope.hrep(vertices))
            except RuntimeError as e:
                raise TubeError("Could not compute primal hrep: %s" % str(e))

    def compute_dual_vrep(self):
        """
        Compute vertices of the dual cone for each primal tube.

        NB: the two tubes can have shared vertices at which dual-cone
        computations can be factored. We don't implement this optimization here.
        """
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
                vertices2d = reduce_polar_system(B, c)
                cone_vertices = polar_to_polytope(vertices2d)
                self.dual_vrep.append(cone_vertices)
            except QhullError:
                raise TubeError("Cannot reduce polar of stance %d" % stance_id)
