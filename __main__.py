#!/usr/bin/env python
"""python test_bbob_pproc.py should run through smoothly. 

"""
import os, sys

os.system('python bbob_pproc/rungeneric.py test_bbob_pproc_input_data/DIRECT')
os.system('python bbob_pproc/rungeneric.py --omit-single' +
                ' test_bbob_pproc_input_data/DIRECT ' + 
                ' test_bbob_pproc_input_data/Cauchy-EDA') 
