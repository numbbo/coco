#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# based on code by Thanh-Do Tran 2012--2015
# adapted by Dimo Brockhoff 2016

from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange  # let range always be an iterator
except NameError: pass
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from cocoex import Suite, Observer, log_level
verbose = 1

import generate_plots



suite_name = "bbob-biobj"
suite_instance = "year:2016"
suite_options = "dimensions: 2,3,5,10,20,40"

suite = Suite(suite_name, suite_instance, suite_options)






f1_id = 8 # in function_ids
f2_id = 15
f1_instance = 2 
f2_instance = 3
dim = 5 # in dimensions
instancesubset = range(1,11) # bbob-biobj instances actually displayed
# Note: in single-objective bbobdocfunctions.pdf documentation, '0' seems to be the instance used

functions = (2,6)
instances = {1: (2, 4), 
             2: (3, 5),
             3: (7, 8),
             4: (9, 10),
             5: (11, 12),
             6: (13, 14),
             7: (15, 16),
             8: (17, 18),
             9: (19, 21),
             10: (21, 22)}

for f1_id in functions:
    for f2_id in functions:
        if f2_id < f1_id:
            continue # take care of not having combination f2 with f1 for example
        for i in instancesubset:
            generate_plots.generate_plots(f1_id, f2_id, instances[i][0],
                                          instances[i][1], dim,
                                          folder='plots/', tofile=True)