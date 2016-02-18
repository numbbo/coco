#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains settings for outputting grayscale figures.

This module modifies module-defined variables so
:py:func:`bbob_pproc.rungeneric.main` will output grayscale figures.

"""

import numpy
from bbob_pproc import ppfigdim, pprldistr, pplogloss
from bbob_pproc.comp2 import ppscatter, ppfig2, pprldistr2
from bbob_pproc.compall import pprldmany, ppfigs


def convtograyscale(rgb):
    """Conversion of RGB to grayscale.

    http://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale

    :keyword seq rgb: sequence of rgb float values (0 to 1)

    :returns: Float for grayscale value

    """

    return (rgb[0]*.3 + rgb[1]*.59 + rgb[2]*.11)

#global instancesOfInterest, tabDimsOfInterest, tabValsOfInterest, figValsOfInterest, rldDimsOfInterest
    #set_trace()
print "Using grayscale settings."

instancesOfInterest = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,
                       11:1, 12:1, 13:1, 14:1, 15:1}

# Variables used in the routines defining desired output for BBOB.
tabDimsOfInterest = (5, 20)    # dimension which are displayed in the tables

# function-dependent target function values: hard coded here before we come up
# with something smarter. It is supposed the number of level of difficulties
# are the same, it is just the target function value that differs.
# tabValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
#tabValsOfInterest = (10, 1.0, 1e-1, 1e-3, 1e-5, 1.0e-8)
tabValsOfInterest = ({1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10,
                      9: 10, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10,
                      16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10,
                      23: 10, 24: 10, 101: 10, 102: 10, 103: 10, 104: 10,
                      105: 10, 106: 10, 107: 10, 108: 10, 109: 10, 110: 10,
                      111: 10, 112: 10, 113: 10, 114: 10, 115: 10, 116: 10,
                      117: 10, 118: 10, 119: 10, 120: 10, 121: 10, 122: 10,
                      123: 10, 124: 10, 125: 10, 126: 10, 127: 10, 128: 10,
                      129: 10, 130: 10},
                     {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
                      8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0,
                      14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0,
                      20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 101: 1.0,
                      102: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0,
                      107: 1.0, 108: 1.0, 109: 1.0, 110: 1.0, 111: 1.0,
                      112: 1.0, 113: 1.0, 114: 1.0, 115: 1.0, 116: 1.0,
                      117: 1.0, 118: 1.0, 119: 1.0, 120: 1.0, 121: 1.0,
                      122: 1.0, 123: 1.0, 124: 1.0, 125: 1.0, 126: 1.0,
                      127: 1.0, 128: 1.0, 129: 1.0, 130: 1.0},
                      {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1,
                       8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1, 12: 0.1, 13: 0.1,
                       14: 0.1, 15: 0.1, 16: 0.1, 17: 0.1, 18: 0.1, 19: 0.1,
                       20: 0.1, 21: 0.1, 22: 0.1, 23: 0.1, 24: 0.1, 101: 0.1,
                       102: 0.1, 103: 0.1, 104: 0.1, 105: 0.1, 106: 0.1,
                       107: 0.1, 108: 0.1, 109: 0.1, 110: 0.1, 111: 0.1,
                       112: 0.1, 113: 0.1, 114: 0.1, 115: 0.1, 116: 0.1,
                       117: 0.1, 118: 0.1, 119: 0.1, 120: 0.1, 121: 0.1,
                       122: 0.1, 123: 0.1, 124: 0.1, 125: 0.1, 126: 0.1,
                       127: 0.1, 128: 0.1, 129: 0.1, 130: 0.1},
                      {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001, 5: 0.001,
                       6: 0.001, 7: 0.001, 8: 0.001, 9: 0.001, 10: 0.001,
                       11: 0.001, 12: 0.001, 13: 0.001, 14: 0.001, 15: 0.001,
                       16: 0.001, 17: 0.001, 18: 0.001, 19: 0.001, 20: 0.001,
                       21: 0.001, 22: 0.001, 23: 0.001, 24: 0.001, 101: 0.001,
                       102: 0.001, 103: 0.001, 104: 0.001, 105: 0.001,
                       106: 0.001, 107: 0.001, 108: 0.001, 109: 0.001,
                       110: 0.001, 111: 0.001, 112: 0.001, 113: 0.001,
                       114: 0.001, 115: 0.001, 116: 0.001, 117: 0.001,
                       118: 0.001, 119: 0.001, 120: 0.001, 121: 0.001,
                       122: 0.001, 123: 0.001, 124: 0.001, 125: 0.001,
                       126: 0.001, 127: 0.001, 128: 0.001, 129: 0.001,
                       130: 0.001},
                      {1: 1e-05, 2: 1e-05, 3: 1e-05, 4: 1e-05, 5: 1e-05,
                       6: 1e-05, 7: 1e-05, 8: 1e-05, 9: 1e-05, 10: 1e-05,
                       11: 1e-05, 12: 1e-05, 13: 1e-05, 14: 1e-05, 15: 1e-05,
                       16: 1e-05, 17: 1e-05, 18: 1e-05, 19: 1e-05, 20: 1e-05,
                       21: 1e-05, 22: 1e-05, 23: 1e-05, 24: 1e-05, 101: 1e-05,
                       102: 1e-05, 103: 1e-05, 104: 1e-05, 105: 1e-05,
                       106: 1e-05, 107: 1e-05, 108: 1e-05, 109: 1e-05,
                       110: 1e-05, 111: 1e-05, 112: 1e-05, 113: 1e-05,
                       114: 1e-05, 115: 1e-05, 116: 1e-05, 117: 1e-05,
                       118: 1e-05, 119: 1e-05, 120: 1e-05, 121: 1e-05,
                       122: 1e-05, 123: 1e-05, 124: 1e-05, 125: 1e-05,
                       126: 1e-05, 127: 1e-05, 128: 1e-05, 129: 1e-05,
                       130: 1e-05},
                      {1: 1e-08, 2: 1e-08, 3: 1e-08, 4: 1e-08, 5: 1e-08,
                       6: 1e-08, 7: 1e-08, 8: 1e-08, 9: 1e-08, 10: 1e-08,
                       11: 1e-08, 12: 1e-08, 13: 1e-08, 14: 1e-08, 15: 1e-08,
                       16: 1e-08, 17: 1e-08, 18: 1e-08, 19: 1e-08, 20: 1e-08,
                       21: 1e-08, 22: 1e-08, 23: 1e-08, 24: 1e-08, 101: 1e-08,
                       102: 1e-08, 103: 1e-08, 104: 1e-08, 105: 1e-08,
                       106: 1e-08, 107: 1e-08, 108: 1e-08, 109: 1e-08,
                       110: 1e-08, 111: 1e-08, 112: 1e-08, 113: 1e-08,
                       114: 1e-08, 115: 1e-08, 116: 1e-08, 117: 1e-08,
                       118: 1e-08, 119: 1e-08, 120: 1e-08, 121: 1e-08,
                       122: 1e-08, 123: 1e-08, 124: 1e-08, 125: 1e-08,
                       126: 1e-08, 127: 1e-08, 128: 1e-08, 129: 1e-08,
                       130: 1e-08})

#figValsOfInterest = (10, 1e-1, 1e-4, 1e-8)
#figValsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8)
figValsOfInterest = ({1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10,
                      9: 10, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10,
                      16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10,
                      23: 10, 24: 10, 101: 10, 102: 10, 103: 10, 104: 10,
                      105: 10, 106: 10, 107: 10, 108: 10, 109: 10, 110: 10,
                      111: 10, 112: 10, 113: 10, 114: 10, 115: 10, 116: 10,
                      117: 10, 118: 10, 119: 10, 120: 10, 121: 10, 122: 10,
                      123: 10, 124: 10, 125: 10, 126: 10, 127: 10, 128: 10,
                      129: 10, 130: 10},
                     {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
                      8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0,
                      14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0,
                      20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 101: 1.0,
                      102: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0,
                      107: 1.0, 108: 1.0, 109: 1.0, 110: 1.0, 111: 1.0,
                      112: 1.0, 113: 1.0, 114: 1.0, 115: 1.0, 116: 1.0,
                      117: 1.0, 118: 1.0, 119: 1.0, 120: 1.0, 121: 1.0,
                      122: 1.0, 123: 1.0, 124: 1.0, 125: 1.0, 126: 1.0,
                      127: 1.0, 128: 1.0, 129: 1.0, 130: 1.0},
                      {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1,
                       8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1, 12: 0.1, 13: 0.1,
                       14: 0.1, 15: 0.1, 16: 0.1, 17: 0.1, 18: 0.1, 19: 0.1,
                       20: 0.1, 21: 0.1, 22: 0.1, 23: 0.1, 24: 0.1, 101: 0.1,
                       102: 0.1, 103: 0.1, 104: 0.1, 105: 0.1, 106: 0.1,
                       107: 0.1, 108: 0.1, 109: 0.1, 110: 0.1, 111: 0.1,
                       112: 0.1, 113: 0.1, 114: 0.1, 115: 0.1, 116: 0.1,
                       117: 0.1, 118: 0.1, 119: 0.1, 120: 0.1, 121: 0.1,
                       122: 0.1, 123: 0.1, 124: 0.1, 125: 0.1, 126: 0.1,
                       127: 0.1, 128: 0.1, 129: 0.1, 130: 0.1},
                      {1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01,
                       6: 0.01, 7: 0.01, 8: 0.01, 9: 0.01, 10: 0.01,
                       11: 0.01, 12: 0.01, 13: 0.01, 14: 0.01, 15: 0.01,
                       16: 0.01, 17: 0.01, 18: 0.01, 19: 0.01, 20: 0.01,
                       21: 0.01, 22: 0.01, 23: 0.01, 24: 0.01, 101: 0.01,
                       102: 0.01, 103: 0.01, 104: 0.01, 105: 0.01,
                       106: 0.01, 107: 0.01, 108: 0.01, 109: 0.01,
                       110: 0.01, 111: 0.01, 112: 0.01, 113: 0.01,
                       114: 0.01, 115: 0.01, 116: 0.01, 117: 0.01,
                       118: 0.01, 119: 0.01, 120: 0.01, 121: 0.01,
                       122: 0.01, 123: 0.01, 124: 0.01, 125: 0.01,
                       126: 0.01, 127: 0.01, 128: 0.01, 129: 0.01,
                       130: 0.01},
                      {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001, 5: 0.001,
                       6: 0.001, 7: 0.001, 8: 0.001, 9: 0.001, 10: 0.001,
                       11: 0.001, 12: 0.001, 13: 0.001, 14: 0.001, 15: 0.001,
                       16: 0.001, 17: 0.001, 18: 0.001, 19: 0.001, 20: 0.001,
                       21: 0.001, 22: 0.001, 23: 0.001, 24: 0.001, 101: 0.001,
                       102: 0.001, 103: 0.001, 104: 0.001, 105: 0.001,
                       106: 0.001, 107: 0.001, 108: 0.001, 109: 0.001,
                       110: 0.001, 111: 0.001, 112: 0.001, 113: 0.001,
                       114: 0.001, 115: 0.001, 116: 0.001, 117: 0.001,
                       118: 0.001, 119: 0.001, 120: 0.001, 121: 0.001,
                       122: 0.001, 123: 0.001, 124: 0.001, 125: 0.001,
                       126: 0.001, 127: 0.001, 128: 0.001, 129: 0.001,
                       130: 0.001},
                      {1: 1e-05, 2: 1e-05, 3: 1e-05, 4: 1e-05, 5: 1e-05,
                       6: 1e-05, 7: 1e-05, 8: 1e-05, 9: 1e-05, 10: 1e-05,
                       11: 1e-05, 12: 1e-05, 13: 1e-05, 14: 1e-05, 15: 1e-05,
                       16: 1e-05, 17: 1e-05, 18: 1e-05, 19: 1e-05, 20: 1e-05,
                       21: 1e-05, 22: 1e-05, 23: 1e-05, 24: 1e-05, 101: 1e-05,
                       102: 1e-05, 103: 1e-05, 104: 1e-05, 105: 1e-05,
                       106: 1e-05, 107: 1e-05, 108: 1e-05, 109: 1e-05,
                       110: 1e-05, 111: 1e-05, 112: 1e-05, 113: 1e-05,
                       114: 1e-05, 115: 1e-05, 116: 1e-05, 117: 1e-05,
                       118: 1e-05, 119: 1e-05, 120: 1e-05, 121: 1e-05,
                       122: 1e-05, 123: 1e-05, 124: 1e-05, 125: 1e-05,
                       126: 1e-05, 127: 1e-05, 128: 1e-05, 129: 1e-05,
                       130: 1e-05},
                      {1: 1e-08, 2: 1e-08, 3: 1e-08, 4: 1e-08, 5: 1e-08,
                       6: 1e-08, 7: 1e-08, 8: 1e-08, 9: 1e-08, 10: 1e-08,
                       11: 1e-08, 12: 1e-08, 13: 1e-08, 14: 1e-08, 15: 1e-08,
                       16: 1e-08, 17: 1e-08, 18: 1e-08, 19: 1e-08, 20: 1e-08,
                       21: 1e-08, 22: 1e-08, 23: 1e-08, 24: 1e-08, 101: 1e-08,
                       102: 1e-08, 103: 1e-08, 104: 1e-08, 105: 1e-08,
                       106: 1e-08, 107: 1e-08, 108: 1e-08, 109: 1e-08,
                       110: 1e-08, 111: 1e-08, 112: 1e-08, 113: 1e-08,
                       114: 1e-08, 115: 1e-08, 116: 1e-08, 117: 1e-08,
                       118: 1e-08, 119: 1e-08, 120: 1e-08, 121: 1e-08,
                       122: 1e-08, 123: 1e-08, 124: 1e-08, 125: 1e-08,
                       126: 1e-08, 127: 1e-08, 128: 1e-08, 129: 1e-08,
                       130: 1e-08})

rldDimsOfInterest = (5, 20)

single_target_function_values = (1e1, 1e0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8)  # one figure for each
summarized_target_function_values = (1e0, 1e-1, 1e-3, 1e-5, 1e-7)   # all in one figure
summarized_target_function_values = (100, 10, 1e0, 1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8) 
summarized_target_function_values = tuple(10**numpy.r_[-8:2:0.2]) # 1e2 and 1e-8
#summarized_target_function_values = tuple(10**numpy.r_[-7:-1:0.2]) # 1e2 and 1e-8  
#summarized_target_function_values = tuple(10**numpy.r_[-1:2:0.2]) # easy easy 
# summarized_target_function_values = (10, 1e0, 1e-1)   # all in one figure

#tableconstant_target_function_values = (1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7)
tableconstant_target_function_values = (1e1, 1e0, 1e-1, 1e-3, 1e-5, 1e-7)

rcaxeslarger = {"labelsize": 24, "titlesize": 28.8}
rcticklarger = {"labelsize": 24}
rcfontlarger = {"size": 24}
rclegendlarger = {"fontsize": 24}

rcaxes = {"labelsize": 20, "titlesize": 24}
rctick = {"labelsize": 20}
rcfont = {"size": 20}
rclegend = {"fontsize": 20}

ppfigdim.styles = [{'color': '0.', 'marker': 'o', 'markeredgecolor': '0.'},
                   {'color': '0.11'},
                   {'color': '0.525', 'marker': 'v', 'markeredgecolor': '0.525'},
                   {'color': '0.295'},
                   {'color': '0.6675', 'marker': '^', 'markeredgecolor': '0.6675'},
                   {'color': '0.3075'},
                   {'color': '0.', 'marker': 's', 'markeredgecolor': '0.'}] # sort of rainbow style
ppfigdim.refcolor = '0.88'

pprldistr.rldStyles = ({'color': '0.', 'ls': '--'},
                       {'color': '0.525'},
                       {'color': '0.3075', 'ls': '--'},
                       {'color': '0.3', 'linewidth': 3.},
                       {'color': '0.'},
                       {'color': '0.525'},
                       {'color': '0.3075'},
                       {'color': '0.3'},
                       {'color': '0.'},
                       {'color': '0.525'},
                       {'color': '0.3075'},
                       {'color': '0.3'})
pprldistr.rldUnsuccStyles = ({'color': '0.', 'ls': '--'},
                             {'color': '0.525'},
                             {'color': '0.3075', 'ls': '--'},
                             {'color': '0.'},
                             {'color': '0.525', 'ls': '--'},
                             {'color': '0.3075'},
                             {'color': '0.', 'ls': '--'},
                             {'color': '0.525'},
                             {'color': '0.3075', 'ls': '--'},
                             {'color': '0.'},
                             {'color': '0.525', 'ls': '--'},
                             {'color': '0.3075'})  # should not be too short
pprldistr.refcolor = '0.88'

pplogloss.whiskerscolor = '0.11'
pplogloss.boxescolor = '0.11'
pplogloss.medianscolor = '0.3'
pplogloss.capscolor = '0.'
pplogloss.flierscolor = '0.'

ppfig2.linewidth = 4.
ppfig2.styles = [{'color': '0.525', 'marker': '+', 'markeredgecolor': '0.525',
                  'markerfacecolor': 'None'},
                 {'color': '0.295', 'marker': 'v', 'markeredgecolor': '0.295',
                  'markerfacecolor': 'None'},
                 {'color': '0.11', 'marker': '*', 'markeredgecolor': '0.11',
                  'markerfacecolor': 'None'},
                 {'color': '0.', 'marker': 'o', 'markeredgecolor': '0.',
                  'markerfacecolor': 'None'},
                 {'color': '0.3', 'marker': 's', 'markeredgecolor': '0.3',
                  'markerfacecolor': 'None'},
                 {'color': '0.3075', 'marker': 'D', 'markeredgecolor': '0.3075',
                  'markerfacecolor': 'None'},
                 {'color': '0.'},
                 {'color': '0.6675'},
                 {'color': '0.'},
                 {'color': '0.525'},
                 {'color': '0.3'},
                 {'color': '0.3075'}]

pprldistr2.rldStyles = ({'color': '0.', 'ls': '--'},
                        {'color': '0.525'},
                        {'color': '0.3075', 'ls': '--'},
                        {'color': '0.3'},
                        {'color': '0.'},
                        {'color': '0.525'},
                        {'color': '0.3075'},
                        {'color': '0.3'},
                        {'color': '0.'},
                        {'color': '0.525'},
                        {'color': '0.3075'},
                        {'color': '0.3'})

ppscatter.markersize = 14.
ppscatter.colors = ('0.525', '0.295', '0.11', '0.', '0.3', '0.3075', '0.', '0.6675', '0.', '0.525', '0.3', '0.3075')

pprldmany.fontsize = 20.0
pprldmany.styles = [{'marker': 'o', 'linestyle': '-', 'color': '0.11'},
                    {'marker': 'd', 'linestyle': '-', 'color': '0.295'},
                    {'marker': 's', 'linestyle': '-', 'color': '0.3'},
                    {'marker': 'v', 'linestyle': '-', 'color': '0.525'},
                    {'marker': '*', 'linestyle': '-', 'color': '0.3075'},
                    {'marker': 'h', 'linestyle': '-', 'color': '0.6675'},
                    {'marker': '^', 'linestyle': '-', 'color': '0.'},
                    {'marker': 'p', 'linestyle': '-', 'color': '0.11'},
                    {'marker': 'H', 'linestyle': '-', 'color': '0.295'},
                    {'marker': '<', 'linestyle': '-', 'color': '0.3'},
                    {'marker': 'D', 'linestyle': '-', 'color': '0.525'},
                    {'marker': '>', 'linestyle': '-', 'color': '0.3075'},
                    {'marker': '1', 'linestyle': '-', 'color': '0.6675'},
                    {'marker': '2', 'linestyle': '-', 'color': '0.'},
                    {'marker': '3', 'linestyle': '-', 'color': '0.11'},
                    {'marker': '4', 'linestyle': '-', 'color': '0.295'}]
pprldmany.refcolor = '0.88'

ppfigs.styles = [{'marker': 'o', 'markersize': 25, 'linestyle': '-', 'color': '0.11'},
                 {'marker': 'd', 'markersize': 30, 'linestyle': '-', 'color': '0.295'},
                 {'marker': 's', 'markersize': 25, 'linestyle': '-', 'color': '0.3'},
                 {'marker': 'v', 'markersize': 30, 'linestyle': '-', 'color': '0.525'},
                 {'marker': '*', 'markersize': 30, 'linestyle': '-', 'color': '0.3075'},
                 {'marker': 'h', 'markersize': 30, 'linestyle': '-', 'color': '0.6675'},
                 {'marker': '^', 'markersize': 30, 'linestyle': '-', 'color': '0.'},
                 {'marker': 'p', 'markersize': 30, 'linestyle': '-', 'color': '0.11'},
                 {'marker': 'H', 'markersize': 30, 'linestyle': '-', 'color': '0.295'},
                 {'marker': '<', 'markersize': 30, 'linestyle': '-', 'color': '0.3'},
                 {'marker': 'D', 'markersize': 30, 'linestyle': '-', 'color': '0.525'},
                 {'marker': '>', 'markersize': 30, 'linestyle': '-', 'color': '0.3075'},
                 {'marker': '1', 'markersize': 30, 'linestyle': '-', 'color': '0.6675'},
                 {'marker': '2', 'markersize': 30, 'linestyle': '-', 'color': '0.'},
                 {'marker': '3', 'markersize': 30, 'linestyle': '-', 'color': '0.11'},
                 {'marker': '4', 'markersize': 30, 'linestyle': '-', 'color': '0.295'}]
ppfigs.refcolor = '0.88'

