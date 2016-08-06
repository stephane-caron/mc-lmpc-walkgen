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

from numpy import array, hstack, cross, dot, sqrt
from scipy.spatial import ConvexHull


def normalize(v):
    return v / sqrt(dot(v, v))


def compute_polygon_hull(B, c, using_pyparma=False):
    #
    # vertices, _ = project_polytope_cdd(B, c, None, None, eye(2), zeros(2))
    # return [p + outbox.p[:2] for p in vertices]
    #
    assert B.shape[1] == 2
    if any(abs(c) < 1e-10):
        assert False, "ici"
        I = [i for i in xrange(len(c)) if abs(c[i]) > 1e-10]
        B, c = B[I], c[I]
    # if not using_pyparma:
    #     # there may be integer overflows in pyparma
    #     assert all(c > 0), "kron"
    B_polar = hstack([
        (B[:, column] * 1. / c).reshape((B.shape[0], 1))
        for column in xrange(2)])

    def axis_intersection(i, j):
        ai, bi = c[i], B[i]
        aj, bj = c[j], B[j]
        x = (ai * bj[1] - aj * bi[1]) * 1. / (bi[0] * bj[1] - bj[0] * bi[1])
        y = (bi[0] * aj - bj[0] * ai) * 1. / (bi[0] * bj[1] - bj[0] * bi[1])
        return array([x, y])

    hull = ConvexHull([row for row in B_polar], qhull_options='Pp')
    vertices = [axis_intersection(i, j) for (i, j) in hull.simplices]
    return vertices
    # return simplify_polygon(vertices)


def simplify_polygon(vertices, thres=0.1):
    """

    thres -- 0.1 ~ 5 deg deviation
    """
    nr_vertices = []
    n = len(vertices)
    for i, b in enumerate(vertices):
        a = vertices[(i + n - 1) % n]
        c = vertices[(i + 1) % n]
        u_ab = normalize(b - a)
        u_ac = normalize(c - a)
        if abs(cross(u_ab, u_ac)) > thres:
            # print "cross =", abs(cross(u_ab, u_ac))
            nr_vertices.append(b)
    # print "%d -> %d" % (len(vertices), len(nr_vertices))
    return nr_vertices
