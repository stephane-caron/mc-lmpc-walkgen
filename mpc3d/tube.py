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

from numpy import array, cross, dot, float64, hstack, ones, vstack
from polygons import compute_polygon_hull
from pymanoid.draw import draw_3d_cone, draw_line, draw_polyhedron
from pymanoid.polyhedra import Polytope
from pymanoid.utils import normalize
from warnings import warn

# import time
# from numpy import average, std
# comp_times = []


def reduce_polar_system(B, c):
    gravity = pymanoid.get_gravity()
    g = -gravity[2]
    assert g > 0
    # assert all(c > 0), "c > 0 assertion failed"
    # assert all(B[:, 2] < 0)
    if any(abs(c) < 1e-10):
        print "ici!!!!!"
        I = [i for i in xrange(len(c)) if abs(c[i]) > 1e-10]
        B, c = B[I], c[I]
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


class TrajectoryTube(object):

    LINE = 2
    DIAMOND = 6
    BRICK = 8

    def __init__(self, init_com, target_com, init_stance, shape,
                 section_size=0.03):
        """
        Create a new COM trajectory tube.

        INPUT:

        - ``init_com`` -- start position of the COM
        - ``target_com`` -- end position of the COM
        - ``init_stance`` -- stance used to compute the contact wrench cone
        - ``shape`` -- number of vertices of the tube (2, 6 or 8)
        - ``section_size`` -- side of the cross-section square for ``shape`` > 2

        .. NOTE::

            We are assuming that self.init_stance applies to all vertices of the
            tube. See the paper for a discussion of this technical choice.
        """
        delta = target_com - init_com
        if dot(delta, delta) < 1e-6:
            self.vertices = [init_com]
            return
        # center = .5 * (init_com + target_com)
        n = normalize(delta)
        t = array([0., 0., 1.])
        t -= dot(t, n) * n
        t = normalize(t)
        b = cross(n, t)
        square = [dx * t + dy * b for (dx, dy) in [
            (+section_size, +section_size),
            (+section_size, -section_size),
            (-section_size, +section_size),
            (-section_size, -section_size)]]
        if shape == TrajectoryTube.BRICK:
            vertices = \
                [init_com + s for s in square] + \
                [target_com + s for s in square]
        elif shape == TrajectoryTube.DIAMOND:
            mid_com = 0.5 * init_com + 0.5 * target_com
            vertices = [init_com] + [mid_com + s for s in square] + [target_com]
        else:  # default is TrajectoryTube.LINE
            vertices = [init_com, target_com]
        self.init_com = init_com
        self.init_stance = init_stance
        self.target_com = target_com
        self.vertices = vertices

    @property
    def nb_vertices(self):
        return len(self.vertices)

    def compute_bare_dual_cone(self):
        A_O = self.init_stance.get_cwc_pyparma([0, 0, 0])
        gravity = pymanoid.get_gravity()
        B_list, c_list = [], []
        for (i, v) in enumerate(self.vertices):
            B = A_O[:, :3] + cross(A_O[:, 3:], v)
            c = dot(B, gravity)
            B_list.append(B)
            c_list.append(c)
        B = vstack(B_list)
        c = hstack(c_list)
        return (B, c)

    def compute_dual_cone(self):
        # t0 = time.time()
        B0, c0 = self.compute_bare_dual_cone()
        self.cone_vertices = reduce_polar_system(B0, c0)
        try:
            B_new, c_new = Polytope.hrep(self.cone_vertices)
            # comp_times.append(time.time() - t0)
            # if len(comp_times) % 10 == 0:
            #     print "%d (%d):\t%.1f +/- %.1f" % (
            #         self._shape, len(comp_times),
            #         1000 * average(comp_times),
            #         1000 * std(comp_times))
            B, c = (B_new.astype(float64), c_new.astype(float64))
            return (B, c)
        except:
            warn("Polytope.hrep(cone_vertices) failed")
            return None

    def draw(self):
        if not self.vertices:
            return None
        elif len(self.vertices) == 2:
            return draw_line(
                self.vertices[0], self.vertices[1], color=[0., 0.5, 0.5],
                linewidth=5)
        return draw_polyhedron(self.vertices, 'c.-#')

    def draw_dual_cone(self, scale=0.1):
        if not self.cone_vertices:
            return None
        apex = self.target_com
        vscale = [apex + scale * array(v) for v in self.cone_vertices]
        # remember that self.cone_vertices[0] is [0, 0, +g]
        return draw_3d_cone(
            apex=apex, axis=[0, 0, 1], section=vscale[1:])
