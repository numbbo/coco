#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Checks whether the old COCO code, written in python (from coco.gforge.inria.fr)
# does the same than the new COCO code from github.com/numbbo/coco.
#
# written by Dimo Brockhoff in 2016

from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange  # let range always be an iterator
except NameError: pass
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from cocoex import Suite, Observer, log_level
verbose = 1

import bbobbenchmarks as bm


#functions = np.arange(1,6)
#instances = np.arange(1,10)
#dim = [2,3,5,10,20,40,80]

suite = Suite('bbob', 'year:2010', '')

numberOfDifferences = 0
largestAbsDifference = 0
largestRelDifference = 0
numberOfProblems = 0
numberOfSearchPoints = 0

for problem_index, problem in enumerate(suite):
    
    f = int(problem.id.lower().split('_f')[1].split('_')[0])
    d = int(problem.id.lower().split('_d')[1].split('_')[0])
    i = int(problem.id.lower().split('_i')[1].split('_')[0])
    numberOfProblems = numberOfProblems + 1

    xrand = -4 + 8*np.random.rand(d)
    numberOfSearchPoints = numberOfSearchPoints + 1
    
    fun, fopt = bm.instantiate(f, iinstance=i)
    fold = fun.evaluate(xrand)
    
    fnew = problem(xrand)
    
    if (fnew-fold > 1e-10 or fnew-fold < -1e-10):
        print(problem.id + ":")
        print("%e, %e, %e" % (fnew-fold, fnew, fold))
        print('!!!!!!!!!!!!!!!!!!!')
        numberOfDifferences = numberOfDifferences + 1
        if abs(fnew-fold) > largestAbsDifference:
            largestAbsDifference = abs(fnew-fold)
        if abs(fnew-fold)/min(abs(fnew), abs(fold)) > largestRelDifference:
            largestRelDifference = abs(fnew-fold)/min(abs(fnew), abs(fold))
       
print("---------------------------------------------------------")
print("number of problems investigated: %d" % numberOfProblems)
print("number of search points investigated: %d" % numberOfSearchPoints)
print("number of times, there was a difference: %d" % numberOfDifferences)
print("i.e., there was a difference in %f %% of the sampled points" % (numberOfDifferences/numberOfSearchPoints))
print("largest absolute difference: %e" % largestAbsDifference)
print("largest relative difference: %e" % largestRelDifference)

