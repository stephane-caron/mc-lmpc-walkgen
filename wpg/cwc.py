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

from fractions import Fraction
from pyparma import Polyhedron
from numpy import dot, hstack, vectorize, vstack, zeros


def compute_cwc_pyparma(contact_set, p, check_conical=False):
    S = contact_set.compute_wrench_span(p)
    tV = vstack([
        hstack([zeros((S.shape[1], 1)), S.T]),
        hstack([1, zeros(S.shape[0])])])
    # fractionize = vectorize(lambda x: Fraction(str(x)))
    fractionize = vectorize(lambda x: Fraction(x))
    poly = Polyhedron(vrep=fractionize(tV))
    hrep = poly.hrep()
    A_O = -hrep[:, 1:]
    if check_conical:
        b_O = hrep[:, 0]
        assert dot(b_O, b_O) < 1e-20
    return A_O
