#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains some global variables settings for COCO.

These variables may be modified (in particular in rungeneric1, -2, and
-many) and are used for producing figures and tables.

For setting variables dynamically or changing other module settings
here see `config.py`.

"""

import numpy as np

test = False  # debug/test flag, set to False for committing the final version
interactive_mode = True  # open browser with results, grayscale setting (deprecated) deactivates interactive mode

force_assertions = False  # another debug flag for time-consuming assertions
in_a_hurry = 1000  # [0, 1000] lower resolution, no eps, saves 30% time
warning_level = 1  # higher levels show more warnings, experimental, will not work on all warnings
maxevals_fix_display = None  # 3e2 is the expensive setting only used in config, yet to be improved!?
runlength_based_targets = False  # may be overwritten by expensive setting
figure_file_formats = ['svg', 'pdf']
scaling_figures_with_boxes = True
scaling_plots_with_axis_labels = False

balance_instances = True
""" give even weight to all instances by added columns with copies of
    underrepresented instances in DataSet.evals
    """
appended_evals_minimal_trials = 6
""" minimum number of instances required in the ``appended_evals`` array such
    that `DataSet.appended_evals` is created to be different from `DataSet.evals`
    """
weight_evaluations_constraints = (1, 1)
""" weights used to sum function evaluations and constraints evaluations
    in attribute DataSet._evals, if any constraints evaluations are found
    """
target_runlengths_in_scaling_figs = [0.5, 1.2, 3, 10, 50]  # used in config
target_runlengths_in_single_rldistr = [0.5, 2, 10, 50]  # used in config
target_runlengths_pprldmany = np.logspace(np.log10(0.5), np.log10(50), 31) # used in config
target_runlengths_ppscatter = np.logspace(np.log10(0.5), np.log10(50), 8) # used in config
target_runlength = 10  # used in ppfigs.main
single_runlength_factors = [0.5, 1.2, 3, 10] + [10 ** i for i in range(2, 12)] # used in pprldistr

xlimit_expensive = 1e3  # used in 
#tableconstant_target_function_values = (
#1e1, 1e0, 1e-1, 1e-3, 1e-5, 1e-7)  # used as input for pptables.main in rungenericmany
# tableconstant_target_function_values = (1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7) # for post-workshop landscape tables

# tabValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
# tabValsOfInterest = (10, 1.0, 1e-1, 1e-3, 1e-5, 1.0e-8)

dim_related_markers = ('+', 'v', '*', 'o', 's', 'D', 'x')
dim_related_colors = ('c', 'g', 'b', 'k', 'r', 'm', 'k', 'y', 'k', 'c', 'r', 'm')

simulated_runlength_bootstrap_sample_size = 10 + 990 // (1 + 10 * max((0, in_a_hurry)))  # for tables and plots
"""10000 would be better for a final camera-ready paper version"""


# single_target_pprldistr_values = (10., 1e-1, 1e-4, 1e-8)  # used as default in pprldistr.plot method, on graph for each
# single_target_function_values = (1e1, 1e0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8)  # one figure for each, seems not in use
# summarized_target_function_values = (1e0, 1e-1, 1e-3, 1e-5, 1e-7)   # currently not in use, all in one figure (graph?)
# summarized_target_function_values = (100, 10, 1, 1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8) 
# summarized_target_function_values = tuple(10**np.r_[-8:2:0.2]) # 1e2 and 1e-8
# summarized_target_function_values = tuple(10**numpy.r_[-7:-1:0.2]) # 1e2 and 1e-1 
# summarized_target_function_values = [-1, 3] # easy easy 
# summarized_target_function_values = (10, 1e0, 1e-1)   # all in one figure (means what?)

instancesOfInterest2009 = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3}  # 2009 instances
instancesOfInterest2010 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
                           10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1}  # 2010 instances
instancesOfInterest2012 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 21: 1, 22: 1, 23: 1, 24: 1,
                           25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1}  # 2012 instances
instancesOfInterest2013 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 31: 1, 32: 1, 33: 1, 34: 1,
                           35: 1, 36: 1, 37: 1, 38: 1, 39: 1, 40: 1}  # 2013 instances
instancesOfInterest2015 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 41: 1, 42: 1, 43: 1, 44: 1,
                           45: 1, 46: 1, 47: 1, 48: 1, 49: 1, 50: 1}  # 2015 instances
instancesOfInterest2016 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 51: 1, 52: 1, 53: 1, 54: 1,
                           55: 1, 56: 1, 57: 1, 58: 1, 59: 1, 60: 1}  # 2016 instances
instancesOfInterest2017 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 61: 1, 62: 1, 63: 1, 64: 1,
                           65: 1, 66: 1, 67: 1, 68: 1, 69: 1, 70: 1}  # 2017 instances
instancesOfInterest2018 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 71: 1, 72: 1, 73: 1, 74: 1,
                           75: 1, 76: 1, 77: 1, 78: 1, 79: 1, 80: 1}  # 2018 instances
instancesOfInterestBiobj2016 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}  # bi-objective 2016 instances
instancesOfInterestBiobj2017 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
                                10: 1, 11: 1, 12: 1, 13: 1, 14:1, 15:1}  # bi-objective 2017 instances
instancesOfInterestBiobj2018 = instancesOfInterestBiobj2017  # bi-objective 2018 instances

instancesOfInterest = [instancesOfInterest2009,
                       instancesOfInterest2010,
                       instancesOfInterest2012,
                       instancesOfInterest2013,
                       instancesOfInterest2015,
                       instancesOfInterest2016,
                       instancesOfInterest2017,
                       instancesOfInterest2018,
                       instancesOfInterestBiobj2016,
                       instancesOfInterestBiobj2017,
                       instancesOfInterestBiobj2018]

line_styles = [  # used by ppfigs and pprlmany  
    {'marker': 'o', 'markersize': 31, 'linestyle': '-', 'color': '#000080'},  # 'NavyBlue'
    {'marker': 'd', 'markersize': 26, 'linestyle': '-', 'color': '#ff00ff'},  # 'Magenta'
    {'marker': '*', 'markersize': 33, 'linestyle': '-', 'color': '#ff7500'},  # 'Orange' (was too yellow)
    {'marker': 'v', 'markersize': 28, 'linestyle': '-', 'color': '#6495ed'},  # 'CornflowerBlue'
    {'marker': 'h', 'markersize': 30, 'linestyle': '-', 'color': 'r'},  # 'Red'
    {'marker': '^', 'markersize': 25, 'linestyle': '-', 'color': '#9acd32'},  # 'YellowGreen' (glaring)
    #          {'marker': '*', 'markersize': 31, 'linestyle': '-', 'color': 'g'}, # 'green' avoid green because of
    #          {'marker': '*', 'markersize': 31, 'linestyle': '-', 'color': '#ffd700'}, # 'Goldenrod' seems too light
    #          {'marker': '^', 'markersize': 27, 'linestyle': '-', 'color': 'k'}, # 'Black' is too close to NavyBlue
    #          {'marker': 's', 'markersize': 20, 'linestyle': '-', 'color': '#d02090'}, # square, 'VioletRed' seems too close to red
    {'marker': 'p', 'markersize': 24, 'linestyle': '-', 'color': 'c'},
    #{'marker': 'H', 'markersize': 23, 'linestyle': '-', 'color': '#bebebe'},  # 'Gray'
    # {'marker': 'o', 'markersize': 23, 'linestyle': '-', 'color': '#ffff00'}, # 'Yellow'
    {'marker': '3', 'markersize': 23, 'linestyle': '-', 'color': '#adff2f'},  # 'GreenYellow'
    {'marker': '1', 'markersize': 23, 'linestyle': '-', 'color': '#228b22'},  # 'ForestGreen'
    {'marker': 'D', 'markersize': 23, 'linestyle': '-', 'color': '#ffc0cb'},  # 'Lavender'
    {'marker': '<', 'markersize': 23, 'linestyle': '-', 'color': '#87ceeb'},  # 'SkyBlue' close to CornflowerBlue
    {'marker': 'v', 'markersize': 23, 'linestyle': '-', 'color': '#000080'},  # 'NavyBlue'
    {'marker': '*', 'markersize': 23, 'linestyle': '-', 'color': 'r'},  # 'Red'
    {'marker': 's', 'markersize': 23, 'linestyle': '-', 'color': '#ffd700'},  # 'Goldenrod'
    {'marker': 'd', 'markersize': 23, 'linestyle': '-', 'color': '#d02090'},  # square, 'VioletRed'
    {'marker': '^', 'markersize': 23, 'linestyle': '-', 'color': '#6495ed'},  # 'CornflowerBlue'
    {'marker': '<', 'markersize': 23, 'linestyle': '-', 'color': '#ff7500'},  # 'Orange'
    {'marker': 'h', 'markersize': 23, 'linestyle': '-', 'color': '#ff00ff'},  # 'Magenta'
    # {'marker': 's', 'markersize': 20, 'linestyle': '-', 'color': 'm'}, # square, magenta
    {'marker': 'p', 'markersize': 23, 'linestyle': '-', 'color': '#bebebe'},  # 'Gray'
    {'marker': 'H', 'markersize': 23, 'linestyle': '-', 'color': '#87ceeb'},  # 'SkyBlue'
    {'marker': '1', 'markersize': 23, 'linestyle': '-', 'color': '#ffc0cb'},  # 'Lavender'
    {'marker': '2', 'markersize': 23, 'linestyle': '-', 'color': '#228b22'},  # 'ForestGreen'
    {'marker': '4', 'markersize': 23, 'linestyle': '-', 'color': '#32cd32'},  # 'LimeGreen'
    {'marker': '3', 'markersize': 23, 'linestyle': '-', 'color': '#9acd32'},  # 'YellowGreen'
    {'marker': 'D', 'markersize': 23, 'linestyle': '-', 'color': '#adff2f'},  # 'GreenYellow'
    {'marker': 'd', 'markersize': 31, 'linestyle': '-', 'color': '#000080'},  # 'NavyBlue'
    {'marker': '*', 'markersize': 26, 'linestyle': '-', 'color': '#ff00ff'},  # 'Magenta'
    {'marker': 'v', 'markersize': 33, 'linestyle': '-', 'color': '#ffa500'},  # old 'Orange'
    {'marker': 'h', 'markersize': 28, 'linestyle': '-', 'color': '#6495ed'},  # 'CornflowerBlue'
    {'marker': '^', 'markersize': 30, 'linestyle': '-', 'color': 'r'},  # 'Red'
    {'marker': 'p', 'markersize': 25, 'linestyle': '-', 'color': '#9acd32'},  # 'YellowGreen'
    {'marker': 'H', 'markersize': 24, 'linestyle': '-', 'color': 'c'},
    {'marker': '3', 'markersize': 23, 'linestyle': '-', 'color': '#bebebe'},  # 'Gray'

]
line_styles_old = [  # used by ppfigs and pprlmany
    {'marker': 'o', 'markersize': 25, 'linestyle': '-', 'color': 'b'},
    {'marker': 'v', 'markersize': 30, 'linestyle': '-', 'color': 'r'},
    {'marker': '*', 'markersize': 31, 'linestyle': '-', 'color': 'c'},
    {'marker': 's', 'markersize': 20, 'linestyle': '-', 'color': 'm'},  # square
    {'marker': '^', 'markersize': 27, 'linestyle': '-', 'color': 'k'},
    {'marker': 'd', 'markersize': 26, 'linestyle': '-', 'color': 'y'},
    {'marker': 'h', 'markersize': 25, 'linestyle': '-', 'color': 'g'},
    {'marker': 's', 'markersize': 24, 'linestyle': '-', 'color': 'b'},
    {'marker': 'H', 'markersize': 24, 'linestyle': '-', 'color': 'r'},
    {'marker': '<', 'markersize': 24, 'linestyle': '-', 'color': 'c'},
    {'marker': 'D', 'markersize': 24, 'linestyle': '-', 'color': 'm'},
    {'marker': '1', 'markersize': 24, 'linestyle': '-', 'color': 'k'},
    {'marker': '2', 'markersize': 24, 'linestyle': '-', 'color': 'y'},
    {'marker': '4', 'markersize': 24, 'linestyle': '-', 'color': 'g'},
    {'marker': '3', 'markersize': 24, 'linestyle': '-', 'color': 'g'},
    {'marker': 'o', 'markersize': 25, 'linestyle': '-', 'color': 'r'},
    {'marker': 'v', 'markersize': 30, 'linestyle': '-', 'color': 'b'},
    {'marker': '*', 'markersize': 31, 'linestyle': '-', 'color': 'm'},
    {'marker': 's', 'markersize': 20, 'linestyle': '-', 'color': 'c'},  # square
    {'marker': '^', 'markersize': 27, 'linestyle': '-', 'color': 'y'},
    {'marker': 'd', 'markersize': 26, 'linestyle': '-', 'color': 'k'},
    {'marker': 'h', 'markersize': 25, 'linestyle': '-', 'color': 'b'},
    {'marker': 's', 'markersize': 24, 'linestyle': '-', 'color': 'g'},
    {'marker': 'H', 'markersize': 24, 'linestyle': '-', 'color': 'c'},
    {'marker': '<', 'markersize': 24, 'linestyle': '-', 'color': 'r'},
    {'marker': 'D', 'markersize': 24, 'linestyle': '-', 'color': 'k'},
    {'marker': '1', 'markersize': 24, 'linestyle': '-', 'color': 'm'},
    {'marker': '2', 'markersize': 24, 'linestyle': '-', 'color': 'g'},
    {'marker': '4', 'markersize': 24, 'linestyle': '-', 'color': 'y'},
    {'marker': '3', 'markersize': 24, 'linestyle': '-', 'color': 'r'}
]

more_old_line_styles = [  # used by ppfigs and pprlmany
    {'marker': 'o', 'markersize': 25, 'linestyle': '-', 'color': '#000080'},  # 'NavyBlue'
    {'marker': 'v', 'markersize': 30, 'linestyle': '-', 'color': 'r'},  # 'Red'
    {'marker': '*', 'markersize': 31, 'linestyle': '-', 'color': '#ffd700'},  # 'Goldenrod' seems too light
    {'marker': 's', 'markersize': 20, 'linestyle': '-', 'color': '#d02090'},  # square, 'VioletRed'
    {'marker': '^', 'markersize': 27, 'linestyle': '-', 'color': 'k'},  # 'Black' is too close to NavyBlue
    {'marker': 'd', 'markersize': 26, 'linestyle': '-', 'color': '#6495ed'},  # 'CornflowerBlue'
    {'marker': 'h', 'markersize': 25, 'linestyle': '-', 'color': '#ffa500'},  # 'Orange'
    {'marker': 'p', 'markersize': 24, 'linestyle': '-', 'color': '#ff00ff'},  # 'Magenta'
    {'marker': 'H', 'markersize': 24, 'linestyle': '-', 'color': '#bebebe'},  # 'Gray'
    {'marker': '<', 'markersize': 24, 'linestyle': '-', 'color': '#87ceeb'},  # 'SkyBlue'
    {'marker': 'D', 'markersize': 24, 'linestyle': '-', 'color': '#ffc0cb'},  # 'Lavender'
    {'marker': '1', 'markersize': 24, 'linestyle': '-', 'color': '#228b22'},  # 'ForestGreen'
    {'marker': '2', 'markersize': 24, 'linestyle': '-', 'color': '#32cd32'},  # 'LimeGreen'
    {'marker': '4', 'markersize': 24, 'linestyle': '-', 'color': '#9acd32'},  # 'YellowGreen'
    {'marker': '3', 'markersize': 24, 'linestyle': '-', 'color': '#adff2f'},  # 'GreenYellow'
    # {'marker': 'o', 'markersize': 25, 'linestyle': '-', 'color': '#ffff00'}, # 'Yellow'
    {'marker': 'v', 'markersize': 30, 'linestyle': '--', 'color': '#000080'},  # 'NavyBlue'
    {'marker': '*', 'markersize': 31, 'linestyle': '--', 'color': 'r'},  # 'Red'
    {'marker': 's', 'markersize': 20, 'linestyle': '--', 'color': '#ffd700'},  # 'Goldenrod'
    {'marker': 'd', 'markersize': 27, 'linestyle': '--', 'color': '#d02090'},  # square, 'VioletRed'
    {'marker': '^', 'markersize': 26, 'linestyle': '--', 'color': '#6495ed'},  # 'CornflowerBlue'
    {'marker': '<', 'markersize': 25, 'linestyle': '--', 'color': '#ffa500'},  # 'Orange'
    {'marker': 'h', 'markersize': 24, 'linestyle': '--', 'color': '#ff00ff'},  # 'Magenta'
    {'marker': 'p', 'markersize': 24, 'linestyle': '--', 'color': '#bebebe'},  # 'Gray'
    {'marker': 'H', 'markersize': 24, 'linestyle': '--', 'color': '#87ceeb'},  # 'SkyBlue'
    {'marker': '1', 'markersize': 24, 'linestyle': '--', 'color': '#ffc0cb'},  # 'Lavender'
    {'marker': '2', 'markersize': 24, 'linestyle': '--', 'color': '#228b22'},  # 'ForestGreen'
    {'marker': '4', 'markersize': 24, 'linestyle': '--', 'color': '#32cd32'},  # 'LimeGreen'
    {'marker': '3', 'markersize': 24, 'linestyle': '--', 'color': '#9acd32'},  # 'YellowGreen'
    {'marker': 'D', 'markersize': 24, 'linestyle': '--', 'color': '#adff2f'},  # 'GreenYellow'
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

minmax_algorithm_fontsize = [9, 17]  # depending on the number of algorithms

rcaxeslarger = {"labelsize": 24, "titlesize": 28.8}
rcticklarger = {"labelsize": 24}
rcfontlarger = {"size": 24}
rclegendlarger = {"fontsize": 24}

rcaxes = {"labelsize": 20, "titlesize": 24}
rctick = {"labelsize": 20}
rcfont = {"size": 20}
rclegend = {"fontsize": 20}

single_algorithm_file_name = 'index1'
many_algorithm_file_name = 'index'
index_html_file_name = 'index'
ppconv_file_name = 'ppconv'
pprldmany_file_name = 'pprldmany'
pprldmany_group_file_name = 'pprldmany_gr'
ppfigs_file_name = 'ppfigs'
ppscatter_file_name = 'ppscatter'
pptables_file_name = 'pptables'
pprldistr2_file_name = 'pprldistr2'
ppfigdim_file_name = 'ppfigdim'

latex_commands_for_html = 'latex_commands_for_html'

extraction_folder_prefix = '.extracted_'

# default settings for rungeneric, rungeneric1 and rungenericmany
inputCrE = 0.
isFig = True
isTab = True
isNoisy = False
isNoiseFree = False
isConv = False
verbose = False
outputdir = 'ppdata'
inputsettings = 'color'
isExpensive = False
isRldOnSingleFcts = True
isRLDistr = True

# usage: background = {(color, linestyle): [alg1, alg2, ...], }
# for example:
# background = {('#d8d8d8', '-'): ['data/BFGS_ros_noiseless.tgz'], ('#f88017', ':'): ['data/NEWUOA_ros_noiseless.tgz', 'data/RANDOMSEARCH_auger_noiseless.tgz']}
background = {}  # TODO: we should have a more instructive name here
background_default_style = (3 * (0.9,), '-')  # very light gray

foreground_algorithm_list = []
"""a list of data files/folders as those specified in cocopp.main"""

##
isLogLoss = True  # only affects rungeneric1
isPickled = False  # only affects rungeneric1
##    
isScatter = True  # only affects rungenericmany

