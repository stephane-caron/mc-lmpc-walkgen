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

import IPython
import os
import sys

try:
    import pymanoid
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../pymanoid')
    import pymanoid

from numpy import pi
from numpy.random import random
from polygon import compute_static_polygon_bretl
from polygon import compute_static_polygon_cdd_hull
from polygon import compute_static_polygon_pyparma_hull
from polygon import compute_static_polygon_cdd_only


black = (0., 0., 0., 0.5)
cyan = (0., 0.5, 0.5, 0.5)
dt = 3e-2  # [s]
green = (0., 0.5, 0., 0.5)
magenta = (0.5, 0., 0.5, 0.5)
robot_mass = 39  # [kg], updated once robot model is loaded
yellow = (0.5, 0.5, 0., 0.5)


def benchmark():
    # first, we call this one once as it will round contacts RPY
    compute_static_polygon_cdd_only(contacts, robot_mass)
    print ""
    print "Benchmarking computation times"
    print "------------------------------"
    function_calls = ['compute_static_polygon_cdd_hull(contacts)',
                      'compute_static_polygon_pyparma_hull(contacts)',
                      'compute_static_polygon_bretl(contacts, solver="glpk")',
                      'compute_static_polygon_bretl(contacts, solver=None)',
                      'compute_static_polygon_cdd_only(contacts, robot_mass)']
    for call in function_calls:
        print "\n%%timeit %s" % call,
        for _ in xrange(1):
            IPython.get_ipython().magic(u'timeit %s' % call)


def sample_contacts():
    for c in contacts.contacts:
        c.set_pos(1. * (2. * random(3) - 1.))
        c.set_rpy(pi / 2. * (2. * random(3) - 1.))
    try:
        # check that polygon contains the origin:
        compute_static_polygon_cdd_hull(contacts)
    except:
        return sample_contacts()


if __name__ == "__main__":
    if IPython.get_ipython() is None:
        # we use IPython (in interactive mode) for the %timeit function
        print "Usage: ipython -i %s" % os.path.basename(__file__)
        exit(-1)

    pymanoid.init(show=False)
    fname = sys.argv[1] if len(sys.argv) > 1 else 'stances/figure2-triple.json'
    contacts = pymanoid.ContactSet.from_json(fname)
    __w0 = compute_static_polygon_bretl  # W0611
    __w1 = compute_static_polygon_pyparma_hull  # W0611

    for _ in xrange(10):
        sample_contacts()
        benchmark()
