#! /usr/bin/env python

# Creates ENFEs and convergence figures for BBOB post-processing.

import os
import sys
import matplotlib.pyplot as plt
import numpy
from pdb import set_trace
from bbob_pproc import bootstrap

def createFigure(data, figHandle=None):
    """ Create a plot in a figure (eventually new) from a data array.

        Mandatory input:
        data - 2d-array where the 1st column contains x-values
               and the remaining columns contain y-values

        Optional inputs:
        figHandle - handle to existing figure, if omitted a new figure
                    will be created

        Output:
        lines - handles of the Line2D instances.
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
        tmp = numpy.where(numpy.isfinite(data[:,i]))[0]
        if tmp.size > 0:
            lines.extend(plt.plot(data[tmp,0], yValues[tmp]))
            lines.extend(plt.plot([data[tmp[-1],0]], [yValues[tmp[-1]]],
                                  marker='+'))
            #TODO: larger size.
        #else: ???

    return lines


def customizeFigure(figHandle, figureName = None, title='',
                    fileFormat=('png','eps'), labels=None, 
                    scale=['linear','linear'], legend=list(), locLegend='best',
                    verbose=True):
    """ Customize a figure by adding a legend, axis label, etc. At the
        end the figure is saved.

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
    tmp = axisHandle.get_xlim()
    axisHandle.set_xlim((max(tmp[0], 1e-8), tmp[1])) #TODO 1e-8 arbitrarilyt set.
    # Annotate figure
    if labels is None: #Couldn't it be ''?
        axisHandle.set_xlabel(labels[0])
        axisHandle.set_ylabel(labels[1])
    #axisHandle.invert_xaxis()

    # Grid options
    axisHandle.grid('True')
    tmp = axisHandle.get_xticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    axisHandle.set_xticklabels(tmp2)
    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    axisHandle.set_yticklabels(tmp2)

    # Legend
    if len(legend) > 0:
        axisHandle.legend(legend,locLegend)
    axisHandle.invert_xaxis()
    axisHandle.set_title(title)

    # Save figure
    if not (figureName is None or fileFormat is None):
        if isinstance(fileFormat, basestring):
            plt.savefig(figureName + '.' + fileFormat, dpi = 120,
                        format = entry)
            if verbose:
                print 'Wrote figure in %s.' %(figureName + '.' + fileFormat)
        else:
            #TODO: that is if fileFormat is iterable.
            for entry in fileFormat:
                plt.savefig(figureName + '.' + entry, dpi = 120,
                            format = entry)
                if verbose:
                    print 'Wrote figure in %s.' %(figureName + '.' + entry)

    # Close figure
    plt.close(figHandle)

    # TODO:    *much more options available (styles, colors, markers ...)
    #       *output directory - contained in the file name or extra parameter?


def sortIndexEntries(indexEntries):
    """From a list of IndexEntry, returns a sorted dictionary."""
    sortByFunc = {}
    dims = set()
    funcs = set()
    for elem in indexEntries:
        sortByFunc.setdefault(elem.funcId,{})
        sortByFunc[elem.funcId][elem.dim] = elem
        funcs.add(funcId)
        dims.add(elem.dim)

    return sortByFunc


def generateData(indexEntry):
    res = []
    for i in indexEntry.hData:
        tmp = numpy.array(i[0])
        tmp2 = []
        for j in i[indexEntry.nbRuns+1:]:
            tmp2.append(j <= i[0])
        tmp = numpy.append(tmp, bootstrap.sp1(i[1:indexEntry.nbRuns + 1], 
                                              issuccessful=tmp2))
        tmp = numpy.append(tmp, bootstrap.prctile(i[1:indexEntry.nbRuns + 1], 
                                                  [0.5])[0])
        #set_trace()
        res.append(tmp)

    return numpy.vstack(res)


def main(indexEntries, filename):
    """From a list of IndexEntry, returns a convergence and ENFEs figure."""

    sortByFunc = sortIndexEntries(indexEntries)

    for dim in sorted(sortByFunc[func]):
        entry = sortByFunc[func][dim].hData
        set_trace()
        #h = ppfig.createFigure(entry.hData[:,[0,sp1index]], fig)
        #for i in h:
            #plt.setp(i,'color',colors[dim])
        #h = ppfig.createFigure(entry.arrayFullTab[:,[0,medianindex]], fig)
        #for i in h:
            #plt.setp(h,'color',colors[dim],'linestyle','--')
        #Do all this in createFigure?
    if isBenchmarkinfosFound:
        title = funInfos[entry.funcId]
    else:
        title = ''

    customizeFigure(plt.gcf(), filename, title=title,
                    fileFormat=('eps','png'), labels=['', ''],
                    scale=['log','log'], locLegend='best',
                    verbose=verbose)


    # TODO: how do we make a user define what color or line style?