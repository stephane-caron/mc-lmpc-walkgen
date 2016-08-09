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

from numpy import array, hstack
from scipy.spatial import ConvexHull


def compute_polygon_hull(B, c):
    """
    Compute the vertex representation of a polygon defined by:

        B * x <= c

    The origin [0, 0] should lie inside the polygon (c >= 0) in order to build
    the polar form. This case can always be reached if there is a solution by
    translating to an interior point of the polygon. (This function will not
    compute the interior point automatically.)

    INPUT:

    - ``B`` -- (2 x K) matrix
    - ``c`` -- vector of length K and positive coordinates

    OUTPUT:

    List of 2D vertices.
    """
    assert B.shape[1] == 2, "Input is not a polygon"
    assert all(c > 0), "Polygon should contain the origin"

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
