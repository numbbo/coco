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
    in the attribute DataSet._evals when data are loaded.
    """
target_runlengths_in_scaling_figs = [0.5, 1.2, 3, 10, 50]  # used in config
target_runlengths_in_single_rldistr = [0.5, 2, 10, 50]  # used in config
target_runlengths_pprldmany = np.logspace(np.log10(0.5), np.log10(50), 31) # used in config
target_runlengths_ppscatter = np.logspace(np.log10(0.5), np.log10(50), 8) # used in config
target_runlength = 10  # used in ppfigs.main
single_runlength_factors = [0.5, 1.2, 3, 10] + [10 ** i for i in range(2, 12)] # used in pprldistr

xlimit_pprldmany = 1e7
"""maximal run length multiplier used in `pprldmany`.

   This sets ``cocopp.compall.pprldmany.x_limit`` via
   calling `cocopp.config.config()``.

   Noisy setting should be rather 1e8?
   ?maybe better: ``10 * genericsettings.evaluation_setting[1]``?"""
xlimit_expensive = 1e3
"""maximal run length multiplier in expensive setting, used in config for
   `pprldmany` and `ppfigdim`"""
minor_grid_alpha_in_pprldmany = 0.15
"""used in `pprldmany` for empirical runtime distributions, 0 means no minor grid"""
len_of_names_in_pprldmany_legend = None
"""set the length, for example when we want to remove characters that are
   not fully displayed, 9 == len('best 2009')"""

#tableconstant_target_function_values = (
#1e1, 1e0, 1e-1, 1e-3, 1e-5, 1e-7)  # used as input for pptables.main in rungenericmany
# tableconstant_target_function_values = (1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7) # for post-workshop landscape tables

# tabValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
# tabValsOfInterest = (10, 1.0, 1e-1, 1e-3, 1e-5, 1.0e-8)

dim_related_markers = ('+', 'v', '*', 'o', 's', 'D', 'x')
dim_related_colors = ('c', 'g', 'b', 'k', 'r', 'm', 'k', 'y', 'k', 'c', 'r', 'm')

simulated_runlength_bootstrap_sample_size = 30 + int(970 / (1 + 10 * max((0, in_a_hurry))))
"""bootstrap samples, 30 is a multiple of 10 and 15.
   Used for tables and plots. `int(1e4)` would be preferable
   for a final camera-ready paper version.
   """


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
                           75: 1, 76: 1, 77: 1, 78: 1, 79: 1, 80: 1}  # 2018-2020 instances
instancesOfInterest2021 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 91: 1, 92: 1, 93: 1, 94: 1,
                           95: 1, 96: 1, 97: 1, 98: 1, 99: 1, 100: 1}  # 2021-2022 instances
instancesOfInterest2023 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 101: 1, 102: 1, 103: 1, 104: 1,
                           105: 1, 106: 1, 107: 1, 108: 1, 109: 1, 110: 1}  # instances since 2023
instancesOfInterestBiobj2016 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}  # bi-objective 2016 instances
instancesOfInterestBiobj2017 = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
                                10: 1, 11: 1, 12: 1, 13: 1, 14:1, 15:1}  # bi-objective instances since 2017
instancesOfInterest = [instancesOfInterest2009,
                       instancesOfInterest2010,
                       instancesOfInterest2012,
                       instancesOfInterest2013,
                       instancesOfInterest2015,
                       instancesOfInterest2016,
                       instancesOfInterest2017,
                       instancesOfInterest2018,
                       instancesOfInterest2021,
                       instancesOfInterest2023,
                       instancesOfInterestBiobj2016,
                       instancesOfInterestBiobj2017]

reference_algorithm_styles = {'linestyle': '-', 'linewidth': 5,  # used in compall.pprldmany
                              'marker': 'D', 'markersize': 1*0.8*11., 'markeredgewidth': 1*1.5/1.5,
                              'color': 'linen',  # from heavy to light: 'antiquewhite', 'wheat', 'linen'
                              # we could set 'markerfacecolor' and/or 'markeredgecolor' differently
                             }
marker_size_multiplier = 1.1
'''multiply all marker sizes in below line_styles, used in compall.pprldmany.main'''
line_styles = [  # used by ppfigs and pprlmany, linewidth=1 can also be set here but is also modified later using style and zorder
    {'color': '#1874b4', 'linestyle': '-', 'marker': 'o', 'markersize': 7, 'zorder': 2},
    # {'color': '#ff7d0b', 'linestyle': '-', 'marker': 'd', 'markersize': 7, 'zorder': 2},
    {'color': '#fa7009', 'linestyle': '-', 'marker': 'd', 'markersize': 7, 'zorder': 2},
    {'color': '#22a022', 'linestyle': '-', 'marker': '*', 'markersize': 8.7, 'zorder': 2},
    {'color': '#d61e1f', 'linestyle': '-', 'marker': 'P', 'markersize': 8, 'zorder': 2},
    {'color': '#8a52bd', 'linestyle': '-', 'marker': 'p', 'markersize': 8, 'zorder': 2},
    {'color': '#8c493c', 'linestyle': '-', 'marker': 'v', 'markersize': 8, 'zorder': 2},
    {'color': '#e35fbb', 'linestyle': '-', 'marker': 'X', 'markersize': 8, 'zorder': 2},
    {'color': '#11bdcf', 'linestyle': '-', 'marker': '^', 'markersize': 8, 'zorder': 2},
    {'color': '#7f7f7f', 'linestyle': '-', 'marker': 'D', 'markersize': 6, 'zorder': 2},
    {'color': '#bcbd1a', 'linestyle': '-', 'marker': '<', 'markersize': 8, 'zorder': 2},
    {'color': '#440154', 'linestyle': '--', 'marker': '>', 'markersize': 8, 'zorder': 2.001},
#    {'color': '#00ff5c', 'linestyle': '-', 'marker': 'h', 'markersize': 8, 'zorder': 1.999},  # shiny green
    {'color': '#2500bb', 'linestyle': '--', 'marker': 'o', 'markersize': 7, 'zorder': 2.001},  # blue, ff->bb
    {'color': '#f800fd', 'linestyle': '--', 'marker': 'd', 'markersize': 7, 'zorder': 2.001},
    {'color': '#91b6e8', 'linestyle': '-', 'marker': '*', 'markersize': 8.5, 'zorder': 1.4},
    {'color': '#ffae5f', 'linestyle': '-', 'marker': 'P', 'markersize': 8, 'zorder': 1.4},
    {'color': '#82df70', 'linestyle': '-', 'marker': 'p', 'markersize': 8, 'zorder': 1.4},
    {'color': '#ff7b79', 'linestyle': '-', 'marker': 'v', 'markersize': 8, 'zorder': 1.4},
    {'color': '#ba96d5', 'linestyle': '-', 'marker': 'X', 'markersize': 8, 'zorder': 1.4},
    {'color': '#f797c0', 'linestyle': '-', 'marker': '^', 'markersize': 8, 'zorder': 1.4},
    {'color': '#dbdb73', 'linestyle': '-', 'marker': 'D', 'markersize': 6, 'zorder': 1.4},
    {'color': '#82d6e5', 'linestyle': '-', 'marker': '<', 'markersize': 8, 'zorder': 1.4},
    {'color': '#1874b4', 'linestyle': '--', 'marker': '>', 'markersize': 8, 'zorder': 2.001},
    {'color': '#ff7d0b', 'linestyle': '--', 'marker': 'h', 'markersize': 8, 'zorder': 2.001},
    {'color': '#22a022', 'linestyle': '--', 'marker': 'o', 'markersize': 7, 'zorder': 2.001},
    {'color': '#d61e1f', 'linestyle': '--', 'marker': 'd', 'markersize': 7, 'zorder': 2.001},
    {'color': '#8a52bd', 'linestyle': '--', 'marker': '*', 'markersize': 8.5, 'zorder': 2.001},
    {'color': '#8c493c', 'linestyle': '--', 'marker': 'P', 'markersize': 8, 'zorder': 2.001},
    {'color': '#e35fbb', 'linestyle': '--', 'marker': 'p', 'markersize': 8, 'zorder': 2.001},
    {'color': '#11bdcf', 'linestyle': '--', 'marker': 'v', 'markersize': 8, 'zorder': 2.001},
    {'color': '#7f7f7f', 'linestyle': '--', 'marker': 'X', 'markersize': 8, 'zorder': 2.001},
    {'color': '#bcbd1a', 'linestyle': '--', 'marker': '^', 'markersize': 8, 'zorder': 2.001},
    {'color': '#440154', 'linestyle': '-', 'marker': 'D', 'markersize': 6, 'zorder': 2},
    {'color': '#00ff5c', 'linestyle': '--', 'marker': '<', 'markersize': 8, 'zorder': 2.001},
    {'color': '#2500bb', 'linestyle': '-', 'marker': '>', 'markersize': 8, 'zorder': 1.999},  # blue, ff->bb
    {'color': '#f800fd', 'linestyle': '-', 'marker': 'h', 'markersize': 8, 'zorder': 1.999},
    {'color': '#91b6e8', 'linestyle': '--', 'marker': 'o', 'markersize': 7, 'zorder': 1.4},
    {'color': '#ffae5f', 'linestyle': '--', 'marker': 'd', 'markersize': 7, 'zorder': 1.4},
    {'color': '#82df70', 'linestyle': '--', 'marker': '*', 'markersize': 8.5, 'zorder': 1.4},
    {'color': '#ff7b79', 'linestyle': '--', 'marker': 'P', 'markersize': 8, 'zorder': 1.4},
    {'color': '#ba96d5', 'linestyle': '--', 'marker': 'p', 'markersize': 8, 'zorder': 1.4},
    {'color': '#f797c0', 'linestyle': '--', 'marker': 'v', 'markersize': 8, 'zorder': 1.4},
    {'color': '#dbdb73', 'linestyle': '--', 'marker': 'X', 'markersize': 8, 'zorder': 1.4},
    {'color': '#82d6e5', 'linestyle': '--', 'marker': '^', 'markersize': 8, 'zorder': 1.4},
    {'color': '#1874b4', 'linestyle': '-', 'marker': 'D', 'markersize': 6, 'zorder': 2},
    {'color': '#ff7d0b', 'linestyle': '-', 'marker': '<', 'markersize': 8, 'zorder': 2},
    {'color': '#22a022', 'linestyle': '-', 'marker': '>', 'markersize': 8, 'zorder': 2},
    {'color': '#d61e1f', 'linestyle': '-', 'marker': 'h', 'markersize': 8, 'zorder': 2},
    {'color': '#8a52bd', 'linestyle': '-', 'marker': 'o', 'markersize': 7, 'zorder': 2},
    {'color': '#8c493c', 'linestyle': '-', 'marker': 'd', 'markersize': 7, 'zorder': 2}]

# see old_line_styles for older line styles

figsize = [6.4, 4.8]  # == rcParamsDefault['figure.figsize'], used in compall.pprldmany

minmax_algorithm_fontsize = [9, 14]  # used in pprldmany, depending on the number of algorithms

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
ppfigcons_file_name = 'ppfigcons'
ppfigcons1_file_name = 'ppfigcons1'
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
