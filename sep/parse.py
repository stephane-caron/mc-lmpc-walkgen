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
import re
import sys

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path) + '/../wpg')

from stats import AvgStdEstimator

if __name__ == '__main__':
    assert len(sys.argv) > 1
    fname = sys.argv[1]
    d = {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            if "timeit" not in line:
                continue
            names = re.findall('[a-z_]+\(.*\)', line)
            times = re.findall('\: \d+ ms', line)
            if len(times) == 1:
                time = float(times[0][2:-3])
            else:
                times = re.findall('\d+\.\d+ ms', line)
                if len(times) == 1:
                    time = float(times[0][:-3])
                else:
                    times = re.findall('\d+ us', line)
                    if len(times) == 1:
                        time = float(times[0][:-3]) / 1000.
            if len(names) != 1 or len(times) != 1:
                print "IGNORED:\n", line, "END OF IGNORED LINE"
                continue
            # print "line=", line, "time=", time
            name = names[0]
            if name not in d:
                d[name] = AvgStdEstimator()
            d[name].add(time)

for key, estimator in d.iteritems():
    avg, std, n = estimator.get_all()
    print "%s:\t%.2f +/- %.2f (%d samples)" % (key, avg, std, n)
