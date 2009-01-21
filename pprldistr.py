#! /usr/bin/env python

"""Creates run length distribution figures."""


from __future__ import absolute_import

import os
import scipy
import matplotlib.pyplot as plt
from pdb import set_trace

#__all__ = []

rldColors = ['b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm']  # might not be long enough

plt.rc("axes", labelsize=20, titlesize=24)
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)
plt.rc("font", size=20)
plt.rc("legend", fontsize=20)
#Warning! this affects all other plots in the package.
#TODO: put it elsewhere.

maxEvalsFactor = 1e4

def beautifyRLD(figHandle, figureName, legend='', locLegend='best', 
                fileFormat=('png', 'eps'), verbose=True):
    """Formats the figure of the run length distribution."""

    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')
    axisHandle.set_xlim((1.0, maxEvalsFactor))
    axisHandle.set_ylim((0.0, 1.0))
    axisHandle.set_xlabel('log10 of FEvals / DIM')
    axisHandle.set_ylabel('proportion of successful runs')
    # Grid options
    axisHandle.grid('True')
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(scipy.log10(j)))
    axisHandle.set_xticklabels(newxtic)

    if legend:
        axisHandle.legend(legend, locLegend)

    # Save figure
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 120,
                    format = entry)
        if verbose:
            print 'Wrote figure in %s.' %(figureName + '.' + entry)



def plotRLDistr(indexEntries, fvalueToReach, verbose=True):
    """Creates run length distributions from a sequence of indexEntries.

    Returns a plot of a run length distribution.

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

                for k in range(1, i.nbRuns + 1):
                    if j[i.nbRuns + k] <= fvalueToReach:
                        x.append(j[k] / i.dim)
                        fsolved.add(i.funcId)
                break
        nn += i.nbRuns

    n = len(x)
    if n == 0:
        res = plt.plot([], [])
    else:
        x.sort()
        x2 = scipy.hstack([scipy.repeat(x, 2), maxEvalsFactor])
        #maxEvalsFactor : used for the limit of the plot.
        y2 = scipy.hstack([0.0,
                           scipy.repeat(scipy.arange(1, n+1) / float(nn), 2)])
        res = plt.plot(x2, y2)

    return res, fsolved, funcs


def beautifyFVD(figHandle, figureName, fileFormat=('png','eps'), verbose=True):
    """Formats the figure of the run length distribution."""

    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')
    tmp = axisHandle.get_xlim()
    axisHandle.set_xlim((1., tmp[1]))
    #axisHandle.invert_xaxis()
    axisHandle.set_ylim((0.0, 1.0))
    axisHandle.set_xlabel('log10 of Deltaf_best/Deltaf')
    axisHandle.set_ylabel('proportion of successful runs')
    # Grid options
    axisHandle.grid('True')
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(scipy.log10(j)))
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

    Returns a plot of a run length distribution.
    args -- maxEvalsFactor : used for the limit of the plot.

    """

    x = []
    nn = 0
    for i in indexEntries:
        for j in i.vData:
            if j[0] >= maxEvalsF * i.dim:
                break
        x.extend(j[i.nbRuns+1:] / fvalueToReach)
        nn += i.nbRuns

    x.sort()
    x2 = scipy.hstack([scipy.repeat(x, 2)])
    #not efficient if some vals are repeated a lot
    #y2 = scipy.hstack([0.0, scipy.repeat(scipy.arange(1, n)/float(nn), 2),
                       #float(n)/nn, float(n)/nn])
    y2 = scipy.hstack([0.0, scipy.repeat(scipy.arange(1, nn)/float(nn), 2),
                       1.0])
    #set_trace()
    res = plt.plot(x2, y2)

    return res


def main(indexEntries, valuesOfInterest, info, outputdir, verbose):
    """Generate image files of run length distribution figures.
    args:
    info --- string suffix for output files.

    """

    #sortedIndexEntries = sortIndexEntries(indexEntries)
    #set_trace()
    figureName = os.path.join(outputdir,'pprldistr%s' %('_' + info))
    fig = plt.figure()
    legend = []
    for j in range(len(valuesOfInterest)):
        (tmp, fsolved, f) = plotRLDistr(indexEntries, valuesOfInterest[j],
                                        verbose)
        #set_trace()
        if not tmp is None:
            plt.setp(tmp, 'color', rldColors[j])
            #set_trace()
            legend.append('%+d:%d/%d' %  
                          (scipy.log10(valuesOfInterest[j]), len(fsolved), 
                           len(f)))
    beautifyRLD(fig, figureName, legend=legend, verbose=verbose)
    plt.close(fig)

    figureName = os.path.join(outputdir,'ppfvdistr_%s' %(info))
    fig = plt.figure()
    for j in range(len(valuesOfInterest)):
        #set_trace()
        tmp = plotFVDistr(indexEntries, valuesOfInterest[j], maxEvalsFactor,
                          verbose=verbose)
        if not tmp is None:
            plt.setp(tmp, 'color', rldColors[j])

    tmp = scipy.log10(maxEvalsFactor)
    maxEvalsF = scipy.power(10, scipy.arange(tmp, 0, -1) - 1)

    #The last index of valuesOfInterest is still used in this loop.
    for k in range(len(maxEvalsF)):
        tmp = plotFVDistr(indexEntries, valuesOfInterest[j],
                          maxEvalsF=maxEvalsF[k], verbose=verbose)
        if not tmp is None:
            plt.setp(tmp, 'color', rldColors[j+k+1])

    beautifyFVD(fig, figureName, verbose=verbose)
    plt.close(fig)
