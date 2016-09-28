#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
 Plots the empirical runtime distribution (aka aRT values) in objective space
 of all given runs for single function and dimension pairs of the bbob-biobj
 function suite obtained from the available archives of an algorithm output.


 adapted by Dimo Brockhoff 2016
"""

from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange  # let range always be an iterator
except NameError: pass
import numpy as np  # "pip install numpy" installs numpy
import cocoex
from cocoex import Suite, Observer, log_level
verbose = 1

import generate_ERD_plot



###########################################
# parameters to play with:
dims = (5,)
#functions = range(1,56)
functions = (3,)
#inputarchivefolder = 'C:/Users/dimo/Desktop/coco-master-git/MAT-SMS/archive/'
#inputarchivefolder = 'C:/Users/dimo/Desktop/coco-master-git/SMSEMOA_pmsbx_norestart_on_bbob-biobj/SMSEMOA_on_bbob-biobj-001/archive/'
inputarchivefolder = 'C:/Users/dimo/Desktop/coco-master-git/gamultiobj/gamultiobj_on_bbob-biobj/archive/'
outputfolder = 'plots/'
tofile = False # if True: files are written; if False: no files but screen output
logscale = True # plot in logscale
downsample = True # downsample archive to a reasonable number of points (for efficiency reasons)
with_grid = True # if True the aRT values on a regular grid are plotted
                  # if False, the aRT values on the downsampled points are plotted
###########################################


suite_name = "bbob-biobj"
suite_instance = "year:2016"
suite_options = "dimensions: 2,3,5,10,20,40"
suite = Suite(suite_name, suite_instance, suite_options)

for problem_index, problem in enumerate(suite):
    
    f = int(problem.id.lower().split('_f')[1].split('_')[0])
    d = int(problem.id.lower().split('_d')[1].split('_')[0])
    i = int(problem.id.lower().split('_i')[1].split('_')[0])
    
    f1_id = int(problem.name.lower().split('_f')[1].split('_')[0])
    f2_id = int(problem.name.lower().split('_f')[2].split('_')[0])
        
    print("processing %s..." % problem.id)
    
    generate_ERD_plot.generate_ERD_plot(f, d, f1_id, f2_id,
                                  outputfolder=outputfolder, 
                                  inputfolder=inputarchivefolder,
                                  tofile=tofile,
                                  logscale=logscale,
                                  downsample=downsample,
                                  with_grid=with_grid)