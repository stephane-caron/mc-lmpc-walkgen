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

from __future__ import division
from numpy import array, hstack
from scipy.spatial import ConvexHull
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import Polygon as ShapelyPolygon


def compute_polygon_hull(B, c):
    """
    Compute the vertex representation of a polygon defined by:

        B * x <= c

    where x is a 2D vector. The origin [0, 0] should lie inside the polygon (c
    >= 0) in order to build the polar form. This case can always be reached if
    there is a solution by translating to an interior point of the polygon.
    (This function will not compute the interior point automatically.)

    INPUT:

    - ``B`` -- (2 x K) matrix
    - ``c`` -- vector of length K and positive coordinates

    OUTPUT:

    List of 2D vertices.

    .. NOTE::

        Checking that (c > 0) is not optional. The rest of the algorithm can be
        executed when some coordinates c_i < 0, but the result would be wrong.
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


def intersect_line_polygon0(line, vertices):
    """
    Intersect a line segment with a polygon.

    INPUT:

    - ``line`` -- list of two points
    - ``vertices`` -- vertices of the polygon

    OUTPUT:

    Returns a numpy array of shape (2,).
    """
    def in_line(p):
        for q in line:
            if abs(p[0] - q[0]) < 1e-5 and abs(p[1] - q[1]) < 1e-5:
                return True
        return False

    s_polygon = ShapelyPolygon(vertices)
    s_line = ShapelyLineString(line)
    try:
        coords = (array(p) for p in s_polygon.intersection(s_line).coords)
        coords = [p for p in coords if not in_line(p)]
    except NotImplementedError:
        coords = []
    return coords


def intersect_line_polygon(p1, p2, points):
    """
    Returns the first intersection found between [p1, p2] and a polygon.

    INPUT:

    - ``p1`` -- end point of line segment
    - ``p2`` -- end point of line segment
    - ``points`` -- vertices of the polygon

    OUTPUT:

    An intersection point if found, None otherwise.

    .. NOTE::

        Adapted from <http://stackoverflow.com/a/20679579>. This variant
        %timeits around 90 us on my machine, vs. 150 us when using shapely.
    """
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def intersection(L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if abs(D) < 1e-5:
            return None
        x = Dx / D
        y = Dy / D
        return x, y

    hull = ConvexHull(points)
    vertices = array([points[i] for i in hull.vertices])
    n = len(vertices)
    L1 = line(p1, p2)
    x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
    for i, v1 in enumerate(vertices):
        v2 = vertices[(i + 1) % n]
        L2 = line(v1, v2)
        p = intersection(L1, L2)
        if p is not None:
            if x_min < p[0] < x_max and y_min < p[1] < y_max:
                return array(p)
    return None
