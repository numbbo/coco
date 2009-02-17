#! /usr/bin/env python

"""Creates run length distribution figures."""


from __future__ import absolute_import

import os
import numpy
import matplotlib.pyplot as plt
from pdb import set_trace

#__all__ = []

#rldColors = ['b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm']  # might not be long enough
rldColors = ('g', 'c', 'b', 'r', 'm', 'g', 'c', 'b', 'r', 'm')  # should not be too short

#maxEvalsFactor = 1e6

def beautifyRLD(figHandle, figureName, maxEvalsF=maxEvalsFactor,
                fileFormat=('png', 'eps'), verbose=True):
    """Format the figure of the run length distribution and save into files."""
    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')
    axisHandle.set_xlim((1.0, maxEvalsF))
    axisHandle.set_ylim((0.0, 1.0))
    axisHandle.set_xlabel('log10 of FEvals / DIM')
    axisHandle.set_ylabel('proportion of successful runs')
    # Grid options
    axisHandle.grid('True')
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(numpy.log10(j)))
    axisHandle.set_xticklabels(newxtic)

    #set_trace()
    plt.legend(loc='best')
    #if legend:
        #axisHandle.legend(legend, locLegend)

    # Save figure
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 120,
                    format = entry)
        if verbose:
            print 'Wrote figure in %s.' %(figureName + '.' + entry)



def plotRLDistr(indexEntries, fvalueToReach, maxEvalsF=maxEvalsFactor,
                verbose=True):
    """Creates run length distributions from a sequence of indexEntries.

    Keyword arguments:
    indexEntries
    fvalueToReach
    verbose

    Outputs:
    res -- resulting plot.
    fsolved -- number of different functions solved.
    funcs -- number of different function considered.
    """

    x = []
    nn = 0
    fsolved = set()
    funcs = set()
    for i in indexEntries:
        funcs.add(i.funcId)
        for j in i.hData:
            if j[0] <= fvalueToReach:
                #This loop is needed because though some number of function
                #evaluations might be below maxEvals, the target function value
                #might not be reached yet. This is because the horizontal data
                #do not go to maxEvals.

                for k in range(1, i.nbRuns() + 1):
                    if j[i.nbRuns() + k] <= fvalueToReach:
                        x.append(j[k] / i.dim)
                        fsolved.add(i.funcId)
                break
        nn += i.nbRuns()

    label = ('%+d:%d/%d' %
             (numpy.log10(fvalueToReach), len(fsolved), len(funcs)))
    n = len(x)
    if n == 0:
        res = plt.plot([], [], label=label)
    else:
        x.sort()
        x2 = numpy.hstack([numpy.repeat(x, 2), maxEvalsF])
        #maxEvalsF: used for the limit of the plot.
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, n+1) / float(nn), 2)])
        res = plt.plot(x2, y2, label=label)

    return res#, fsolved, funcs


def beautifyFVD(figHandle, figureName, fileFormat=('png','eps'), verbose=True):
    """Formats the figure of the run length distribution."""

    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')
    tmp = axisHandle.get_xlim()
    axisHandle.set_xlim((1., tmp[1]))
    #axisHandle.invert_xaxis()
    axisHandle.set_ylim((0.0, 1.0))
    axisHandle.set_xlabel('log10 of Df / Dftarget')
    # axisHandle.set_ylabel('proportion of successful runs')
    # Grid options
    axisHandle.grid('True')
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(numpy.log10(j)))
    axisHandle.set_xticklabels(newxtic)

    # Save figure
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 120,
                    format = entry)
        if verbose:
            print 'Wrote figure in %s.' %(figureName + '.' + entry)


def plotFVDistr(indexEntries, fvalueToReach=1.e-8, maxEvalsF=maxEvalsFactor,
                verbose=True):
    """Creates empirical cumulative distribution functions of final function
    values plot from a sequence of indexEntries.

    Keyword arguments:
    indexEntries -- sequence of IndexEntry to process.
    fvalueToReach -- float used for the lower limit of the plot
    maxEvalsF -- indicates which vertical data to display.
    verbose -- controls verbosity.

    Outputs: a plot of a run length distribution.
    """

    x = []
    nn = 0
    for i in indexEntries:
        for j in i.vData:
            if j[0] >= maxEvalsF * i.dim:
                break
        x.extend(j[i.nbRuns()+1:] / fvalueToReach)
        nn += i.nbRuns()

    x.sort()
    x2 = numpy.hstack([numpy.repeat(x, 2)])
    #not efficient if some vals are repeated a lot
    #y2 = numpy.hstack([0.0, numpy.repeat(numpy.arange(1, n)/float(nn), 2),
                       #float(n)/nn, float(n)/nn])
    y2 = numpy.hstack([0.0, numpy.repeat(numpy.arange(1, nn)/float(nn), 2),
                       1.0])
    #set_trace()
    res = plt.plot(x2, y2)

    return res


def main(indexEntries, valuesOfInterest, outputdir='', info='default',
         verbose=True):
    """Generate figures of empirical cumulative distribution functions.

    Keyword arguments:
    indexEntries -- list of IndexEntry instances to process.
    valuesOfInterest -- target function values to be displayed.
    outputdir -- output directory (must exist)
    info --- string suffix for output file names.

    Outputs:
    Image files of the empirical cumulative distribution functions.

    """

    #sortedIndexEntries = sortIndexEntries(indexEntries)

    plt.rc("axes", labelsize=20, titlesize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("font", size=20)
    plt.rc("legend", fontsize=20)

    maxEvalsFactor = max(i.mMaxEvals()/i.dim for i in indexEntries)
    maxEvalsFactor = numpy.power(10, round(numpy.log10(maxEvalsFactor)))

    figureName = os.path.join(outputdir,'pprldistr%s' %('_' + info))
    fig = plt.figure()
    legend = []
    for j in range(len(valuesOfInterest)):
        tmp = plotRLDistr(indexEntries, valuesOfInterest[j], maxEvalsFactor,
                          verbose)
        #set_trace()
        if not tmp is None:
            plt.setp(tmp, 'color', rldColors[j])
            #set_trace()
            #legend.append('%+d:%d/%d' %  
                          #(numpy.log10(valuesOfInterest[j]), len(fsolved), 
                           #len(f)))
    beautifyRLD(fig, figureName, maxEvalsFactor, verbose=verbose)
    plt.close(fig)

    figureName = os.path.join(outputdir,'ppfvdistr_%s' %(info))
    fig = plt.figure()
    for j in range(len(valuesOfInterest)):
        #set_trace()
        tmp = plotFVDistr(indexEntries, valuesOfInterest[j], maxEvalsFactor,
                          verbose=verbose)
        #if not tmp is None:
        plt.setp(tmp, 'color', rldColors[j])

    tmp = numpy.log10(maxEvalsFactor)
    maxEvalsF = numpy.power(10, numpy.arange(tmp, 0, -1) - 1)

    #The last index of valuesOfInterest is still used in this loop.
    #set_trace()
    for k in range(len(maxEvalsF)):
        tmp = plotFVDistr(indexEntries, valuesOfInterest[j],
                          maxEvalsF=maxEvalsF[k], verbose=verbose)
        plt.setp(tmp, 'color', rldColors[j+k+1])

    beautifyFVD(fig, figureName, verbose=verbose)
    plt.close(fig)

    plt.rcdefaults()
