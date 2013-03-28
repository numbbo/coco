#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains some variables settings for COCO.

These variables are used for producing figures and tables 
in rungeneric1, -2, and -many.

For setting variables dynamically see config.py, where some 
of the variables here and some 

"""
import numpy as np
np.seterr(under='ignore')  # ignore underflow

#global instancesOfInterest, tabDimsOfInterest, tabValsOfInterest, figValsOfInterest, rldDimsOfInterest, rldValsOfInterest
#set_trace()
test = False  # debug/test flag, set to False for committing the final version
force_assertions = False  # another debug flag for time-consuming assertions
in_a_hurry = 0 # [0, 1000] lower resolution, no eps, saves 30% time
maxevals_fix_display = None  # 3e2 is the expensive setting only used in config, yet to be improved?
runlength_based_targets = 'auto'  # 'auto' means automatic choice, otherwise True or False
dimensions_to_display = (2, 3, 5, 10, 20, 40)  # this could be used to set the dimensions in respective modules
scaling_figures_with_boxes = True 
# should replace ppfigdim.dimsBBOB, ppfig2.dimensions, ppfigparam.dimsBBOB?

# Variables used in the routines defining desired output for BBOB.
tabDimsOfInterest = (5, 20)  # dimension which are displayed in the tables
target_runlengths_in_scaling_figs = [0.5, 1.2, 3, 10, 50]  # used in config
target_runlengths_in_table = [0.5, 1.2, 3, 10, 50]  # [0.5, 2, 10, 50]  # used in config
target_runlengths_in_single_rldistr = [0.5, 2, 10, 50]  # used in config
xlimit_expensive = 1e3  # used in 
tableconstant_target_function_values = (1e1, 1e0, 1e-1, 1e-3, 1e-5, 1e-7) # used as input for pptables.main in rungenericmany 
# tableconstant_target_function_values = (1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7) # for post-workshop landscape tables
rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8)

tabValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
#tabValsOfInterest = (10, 1.0, 1e-1, 1e-3, 1e-5, 1.0e-8)

rldDimsOfInterest = (5, 20)
figValsOfInterest = (10, 1e-1, 1e-4, 1e-8) # this is a bad name that should improve, which fig, what vals???
# figValsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8) #in/outcomment if desired
##Put backward to have the legend in the same order as the lines.

simulated_runlength_bootstrap_sample_size = 10 + 990 / (1 + 10 * max((0, in_a_hurry)))
simulated_runlength_bootstrap_sample_size_rld = 10 + 90 / (1 + 10 * max((0, in_a_hurry)))

# single_target_pprldistr_values = (10., 1e-1, 1e-4, 1e-8)  # used as default in pprldistr.plot method, on graph for each
# single_target_function_values = (1e1, 1e0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8)  # one figure for each, seems not in use
# summarized_target_function_values = (1e0, 1e-1, 1e-3, 1e-5, 1e-7)   # currently not in use, all in one figure (graph?)
# summarized_target_function_values = (100, 10, 1, 1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8) 
# summarized_target_function_values = tuple(10**np.r_[-8:2:0.2]) # 1e2 and 1e-8
# summarized_target_function_values = tuple(10**numpy.r_[-7:-1:0.2]) # 1e2 and 1e-1 
# summarized_target_function_values = [-1, 3] # easy easy 
# summarized_target_function_values = (10, 1e0, 1e-1)   # all in one figure (means what?)
# not (yet) in use: pprldmany_target_values = pproc.TargetValues().set_targets(10**np.arange(-8, 2, 0.2)) (might not work because of cyclic import

fig_formats = ('eps', 'pdf') if not in_a_hurry else ('pdf', )

instancesOfInterest = {1:1, 2:1, 3:1, 4:1, 5:1, 31:1, 32:1, 33:1, 34:1, 35:1,
                       36:1, 37:1, 38:1, 39:1, 40:1}  # only for consistency checking
line_styles = [  # used by ppfigs and pprlmany  
          {'marker': 'o', 'markersize': 25, 'linestyle': '-', 'color': 'b'},
          {'marker': 'v', 'markersize': 30, 'linestyle': '-', 'color': 'r'}, 
          {'marker': '*', 'markersize': 31, 'linestyle': '-', 'color': 'c'},
          {'marker': 's', 'markersize': 20, 'linestyle': '-', 'color': 'm'}, # square
          {'marker': '^', 'markersize': 27, 'linestyle': '-', 'color': 'k'},
          {'marker': 'd', 'markersize': 26, 'linestyle': '-', 'color': 'y'},
          {'marker': 'h', 'markersize': 25, 'linestyle': '-', 'color': 'g'},
          {'marker': 'p', 'markersize': 24, 'linestyle': '-', 'color': 'b'},
          {'marker': 'H', 'markersize': 24, 'linestyle': '-', 'color': 'r'},
          {'marker': '<', 'markersize': 24, 'linestyle': '-', 'color': 'c'},
          {'marker': 'D', 'markersize': 24, 'linestyle': '-', 'color': 'm'},
          {'marker': '1', 'markersize': 24, 'linestyle': '-', 'color': 'k'},
          {'marker': '2', 'markersize': 24, 'linestyle': '-', 'color': 'y'},
          {'marker': '4', 'markersize': 24, 'linestyle': '-', 'color': 'g'},
          {'marker': '3', 'markersize': 24, 'linestyle': '-', 'color': 'g'}
          ]

if 11 < 3:  # in case using my own linestyles
    line_styles = [  # used by ppfigs and pprlmany, to be modified  
          {'marker': 'o', 'markersize': 25, 'linestyle': '-', 'color': 'b'},
          {'marker': 'o', 'markersize': 30, 'linestyle': '-', 'color': 'r'}, 
          {'marker': '*', 'markersize': 31, 'linestyle': '-', 'color': 'b'},
          {'marker': '*', 'markersize': 20, 'linestyle': '-', 'color': 'r'}, 
          {'marker': '^', 'markersize': 27, 'linestyle': '-', 'color': 'b'},
          {'marker': '^', 'markersize': 26, 'linestyle': '-', 'color': 'r'},
          {'marker': 'h', 'markersize': 25, 'linestyle': '-', 'color': 'g'},
          {'marker': 'p', 'markersize': 24, 'linestyle': '-', 'color': 'b'},
          {'marker': 'H', 'markersize': 24, 'linestyle': '-', 'color': 'r'},
          {'marker': '<', 'markersize': 24, 'linestyle': '-', 'color': 'c'},
          {'marker': 'D', 'markersize': 24, 'linestyle': '-', 'color': 'm'},
          {'marker': '1', 'markersize': 24, 'linestyle': '-', 'color': 'k'},
          {'marker': '2', 'markersize': 24, 'linestyle': '-', 'color': 'y'},
          {'marker': '4', 'markersize': 24, 'linestyle': '-', 'color': 'g'},
          {'marker': '3', 'markersize': 24, 'linestyle': '-', 'color': 'g'}
          ]
    
minmax_algorithm_fontsize = [10, 15]  # depending on the number of algorithms

rcaxeslarger = {"labelsize": 24, "titlesize": 28.8}
rcticklarger = {"labelsize": 24}
rcfontlarger = {"size": 24}
rclegendlarger = {"fontsize": 24}

rcaxes = {"labelsize": 20, "titlesize": 24}
rctick = {"labelsize": 20}
rcfont = {"size": 20}
rclegend = {"fontsize": 20}
    
class Testbed(object):
    """this might become the future way to have settings related to testbeds"""
    pass

class GECCOBBOBTestbed(Testbed):
    def __init__(self):
        # TODO: should become a function, as low_budget is a display setting
        # not a testbed setting
        pass
    
class GECCOBBOBNoisefreeTestbed(GECCOBBOBTestbed):
    pass

# TODO: this needs to be set somewhere, e.g. in rungeneric*
# or even better by investigating in the data attributes
current_testbed = GECCOBBOBNoisefreeTestbed() 

if in_a_hurry:
    print 'in_a_hurry like', in_a_hurry, '(should finally be set to zero)'
    