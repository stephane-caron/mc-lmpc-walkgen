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

import os
import sys
import timeit

from numpy import pi
from numpy.random import random

try:  # use local pymanoid submodule
    script_path = os.path.realpath(__file__)
    sys.path = [os.path.dirname(script_path) + '/../pymanoid'] + sys.path
    import pymanoid
except:  # this is to avoid warning E402 from Pylint
    pass

try:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../wpg')
    from polygon import compute_static_polygon_cdd_hull
    from stats import AvgStdEstimator
except:  # this is to avoid warning E402 from Pylint
    pass


nb_contact_sets = 100
robot_mass = 39.  # [kg]

function_calls = [
    'compute_static_polygon_cdd_hull(contacts)',
    'compute_static_polygon_hull_only(A_O)',
    'compute_static_polygon_pyparma_hull(contacts)',
    'compute_static_polygon_bretl(contacts, solver="glpk")',
    # 'compute_static_polygon_bretl(contacts, solver=None)',
    'compute_static_polygon_cdd_only(contacts, robot_mass)']

setup = """
from polygon import compute_static_polygon_bretl
from polygon import compute_static_polygon_cdd_hull
from polygon import compute_static_polygon_hull_only
from polygon import compute_static_polygon_cdd_only
from polygon import compute_static_polygon_pyparma_hull
from __main__ import contacts, robot_mass, A_O
"""

results = {f: AvgStdEstimator() for f in function_calls}
outliers = {}


def benchmark_contacts(nb_iter=10):
    """Run all methods for the current contact set."""
    global outliers
    times = {}
    for call in function_calls:
        try:
            timer = timeit.Timer(call, setup=setup)
            time_ms = 1000. * min(timer.repeat(3, nb_iter)) / nb_iter
            times[call] = time_ms
        except Exception as e:
            print "Exception:", e
    output = results
    if len(times) < len(function_calls):
        if not outliers:
            outliers = {f: AvgStdEstimator() for f in function_calls}
        output = outliers
    for call in function_calls:
        if call in times:
            output[call].add(times[call])


def sample_contacts():
    """Sample a new contact set."""
    found = False
    while not found:
        for c in contacts.contacts:
            c.set_pos(1. * (2. * random(3) - 1.))
            c.set_rpy(pi / 2. * (2. * random(3) - 1.))
        try:
            # check that polygon contains the origin:
            compute_static_polygon_cdd_hull(contacts)
            found = True
        except:
            pass


def print_results(d):
    for call, estimator in d.iteritems():
        avg, std, n = estimator.get_all()
        if n > 0:
            print "%s:\t%.2f +/- %.2f (%d samples)" % (call, avg, std, n)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['single', 'double', 'triple']:
        print "Usage: %s <single|double|triple> [nb_contact_sets]" % sys.argv[0]
        sys.exit(-1)
    if len(sys.argv) > 2:
        nb_contact_sets = int(sys.argv[2])
    pymanoid.init(set_viewer=False)
    size = sys.argv[1]
    if size == 'triple':
        function_calls.pop()  # don't do cdd_only for > 2 contacts
    fname = '../stances/%s.json' % size
    contacts = pymanoid.ContactSet.from_json(fname)
    for i in xrange(nb_contact_sets):
        print "Contact set %d / %d..." % (i + 1, nb_contact_sets)
        sample_contacts()
        A_O = contacts.compute_wrench_cone([0, 0, 0])
        benchmark_contacts()
    print "\nRESULTS"
    print "=======\n"
    print_results(results)
    if outliers:
        print "\nOUTLIERS"
        print "========\n"
        print_results(outliers)
