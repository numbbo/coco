#! /usr/bin/env python

# Creates pictures for BBOB post-processing.

import os
import sys
import matplotlib.pyplot as plt
import scipy
from pdb import set_trace

def createFigure(data, figHandle = None):
    """ Create a figure from an array.

        Mandatory input:
        data - 2d-array where the 1st column contains x-values
               and the remaining columns contain y-values

        Optional inputs:
        figHandle - handle to existing figure, if omitted a new figure
                    will be created

        Output:
        lines - handles of the Line2D instances
    """

    # initialize figure and handles
    if figHandle is None:
        figHandle = plt.figure()
    else:
        plt.figure(figHandle.number)

    lines = []
    # Plot data sets
    for i in range(1,len(data[0,:])):
        yValues = data[:,i]

        # Plot figure
        tmp = scipy.where(scipy.isfinite(data[:,0]))[0]
        if tmp.size > 0:
            lines.extend(plt.plot(data[tmp,0], yValues[tmp]))
            lines.extend(plt.plot([data[tmp[-1],0]], [yValues[tmp[-1]]],
                                  marker='+'))
            #TODO: larger size.
        #else: ???

    return lines


def customizeFigure(figHandle, figureName, title = '',
                    fileFormat = ('png','eps'),
                    labels = None, scale = ['linear','linear'],
                    legend = list(), locLegend = 'best',verbose = True):
    """ Customize a figure by adding a legend, axis label, etc. At the
        end the fiugre is saved.

        Inputs:
        figHandle - handle to existing figure
        figureName - name of the output figure

        Optional Inputs:
        fileFormat - list of formats of output files
        labels - list with xlabel and ylabel
        scale - scale for x-axis and y-axis
        legend - if this list is not empty a legend will be generated from
                 the entries of this list
        locLegend - location of legend

    """

    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = figHandle.gca()
    axisHandle.set_xscale(scale[0])
    axisHandle.set_yscale(scale[1])

    # Annotate figure
    if labels is None: #Couldn't it be ''?
        axisHandle.set_xlabel(labels[0])
        axisHandle.set_ylabel(labels[1])
    #axisHandle.invert_xaxis()

    # Grid options
    axisHandle.grid('True')
    #set_trace()
    tmp = axisHandle.get_xticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(scipy.log10(i)))
    axisHandle.set_xticklabels(tmp2)
    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(scipy.log10(i)))
    axisHandle.set_yticklabels(tmp2)

    # Legend
    if len(legend) > 0:
        axisHandle.legend(legend,locLegend)

    axisHandle.set_title(title)

    # Save figure
    if isinstance(fileFormat, basestring):
        plt.savefig(figureName + '.' + fileFormat, dpi = 120,
                    format = entry)
        if verbose:
            print 'Wrote figure in %s.' %(figureName + '.' + fileFormat)
    else:
        for entry in fileFormat:
            plt.savefig(figureName + '.' + entry, dpi = 120,
                        format = entry)
            if verbose:
                print 'Wrote figure in %s.' %(figureName + '.' + entry)

    # Close figure
    plt.close(figHandle)

    # TODO:    *much more options available (styles, colors, markers ...)
    #       *output directory - contained in the file name or extra parameter?
