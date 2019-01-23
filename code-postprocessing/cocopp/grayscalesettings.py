#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This module contains settings for outputting grayscale figures.

This module modifies module-defined variables so
:py:func:`cocopp.rungeneric.main` will output grayscale figures.

"""

from __future__ import print_function
from . import ppfigdim, pprldistr, pplogloss, genericsettings
from .comp2 import ppscatter, ppfig2, pprldistr2
from .compall import pprldmany, ppfigs


def convtograyscale(rgb):
    """Conversion of RGB to grayscale.

    http://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale

    :keyword seq rgb: sequence of rgb float values (0 to 1)

    :returns: Float for grayscale value

    """

    return (rgb[0]*.3 + rgb[1]*.59 + rgb[2]*.11)

print("Using grayscale settings.")

instancesOfInterest = genericsettings.instancesOfInterest

#single_target_function_values = (1e1, 1e0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8)  # one figure for each
#summarized_target_function_values = (1e0, 1e-1, 1e-3, 1e-5, 1e-7)   # all in one figure
#summarized_target_function_values = (100, 10, 1e0, 1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8) 
#summarized_target_function_values = tuple(10**numpy.r_[-8:2:0.2]) # 1e2 and 1e-8

#tableconstant_target_function_values = (1e1, 1e0, 1e-1, 1e-3, 1e-5, 1e-7)

rcaxeslarger = genericsettings.rcaxeslarger
rcticklarger = genericsettings.rcticklarger
rcfontlarger = genericsettings.rcfontlarger
rclegendlarger = genericsettings.rclegendlarger
rcaxes = genericsettings.rcaxes
rctick = genericsettings.rctick
rcfont = genericsettings.rcfont
rclegend = genericsettings.rclegend

genericsettings.dim_related_colors = ('0.', '0.8', '0.2', '0.6', '0.4', '0.', '0.8', '0.2', '0.6', '0.4',)
refcolor = '0.88' # color of reference algorithm


genericsettings.line_styles = [  # used by ppfigs and pprlmany  
    {'marker': 'o', 'markersize': 31, 'linestyle': '-', 'color': '0'},
    {'marker': 'd', 'markersize': 26, 'linestyle': '-', 'color': '0.6'},
    {'marker': '*', 'markersize': 33, 'linestyle': '-', 'color': '0.9'},
    {'marker': 'v', 'markersize': 28, 'linestyle': '-', 'color': '0.3'},
    {'marker': 'h', 'markersize': 30, 'linestyle': '--', 'color': '0.8'},
    {'marker': '^', 'markersize': 25, 'linestyle': '--', 'color': '0.4'},
    {'marker': 'p', 'markersize': 24, 'linestyle': '--', 'color': '0.2'},
    {'marker': 'H', 'markersize': 23, 'linestyle': ':', 'color': '0.7'},
    {'marker': '3', 'markersize': 23, 'linestyle': ':', 'color': '0.5'},
    {'marker': '1', 'markersize': 23, 'linestyle': ':', 'color': '0.3'},
    {'marker': 'D', 'markersize': 23, 'linestyle': ':', 'color': '0.1'},
    {'marker': '<', 'markersize': 23, 'linestyle': '-', 'color': '0.15'},
    {'marker': 'v', 'markersize': 23, 'linestyle': '-', 'color': '0.45'},
    {'marker': '*', 'markersize': 23, 'linestyle': '-', 'color': '0.75'},
    {'marker': 's', 'markersize': 23, 'linestyle': '--', 'color': '0.9'},
    {'marker': 'd', 'markersize': 23, 'linestyle': '--', 'color': '0.1'},
    {'marker': '^', 'markersize': 23, 'linestyle': '--', 'color': '0.6'},
    {'marker': '<', 'markersize': 23, 'linestyle': '--', 'color': '0.3'},
    {'marker': 'h', 'markersize': 23, 'linestyle': '--', 'color': '0.5'},
    {'marker': 'p', 'markersize': 23, 'linestyle': '--', 'color': '0.7'},
    {'marker': 'H', 'markersize': 23, 'linestyle': ':', 'color': '0'},
    {'marker': '1', 'markersize': 23, 'linestyle': ':', 'color': '0.2'},
    {'marker': '2', 'markersize': 23, 'linestyle': ':', 'color': '0.9'},
    {'marker': '4', 'markersize': 23, 'linestyle': ':', 'color': '0.6'},
    {'marker': '3', 'markersize': 23, 'linestyle': ':', 'color': '0.8'},
    {'marker': 'D', 'markersize': 23, 'linestyle': ':', 'color': '0.4'},
]


ppfigdim.styles = [{'color': '0.', 'marker': 'o', 'markeredgecolor': '0.', 'markeredgewidth': 2, 'linewidth': 4},
                   {'color': '0.11', 'marker': '.', 'linewidth': 4},
                   {'color': '0.525', 'marker': '^', 'markeredgecolor': '0.525', 'markeredgewidth': 2, 'linewidth': 4},
                   {'color': '0.295', 'marker': '.', 'linewidth': 4},
                   {'color': '0.6675', 'marker': 'v', 'markeredgecolor': '0.6675', 'markeredgewidth': 2, 'linewidth': 4},
                   {'color': '0.3075', 'marker': '.', 'linewidth': 4},
                   {'color': '0.', 'marker': 'o', 'markeredgecolor': '0.', 'markeredgewidth': 2, 'linewidth': 4}]
ppfigdim.refcolor = refcolor

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
pprldistr.refcolor = refcolor

pplogloss.whiskerscolor = '0.11'
pplogloss.boxescolor = '0.11'
pplogloss.medianscolor = '0.3'
pplogloss.capscolor = '0.'
pplogloss.flierscolor = '0.'

ppfig2.linewidth = 4.


ppfig2.styles = [{'color': '0.', 'marker': '+', 'markeredgecolor': '0.525',
                  'markerfacecolor': 'None'},
                 {'color': '0.', 'marker': 'v', 'markeredgecolor': '0.295',
                  'markerfacecolor': 'None'},
                 {'color': '0.', 'marker': '*', 'markeredgecolor': '0.11',
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

pprldmany.fontsize = 20.0
pprldmany.styles = [d.copy() for d in genericsettings.line_styles]
pprldmany.refcolor = refcolor

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
ppfigs.styles = genericsettings.line_styles
ppfigs.refcolor = refcolor
