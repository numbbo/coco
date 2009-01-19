#! /usr/bin/env python

"""Creates run length distribution figures."""


from __future__ import absolute_import

import os
import scipy
import matplotlib.pyplot as plt
from pdb import set_trace

rldColors = ['b', 'g', 'r', 'c', 'm']

#% SEPARABLE
#1 Sphere
#2 Ellipsoid separable with monotone x-transformation, condition 1e6
#3 Rastrigin separable with asymmetric x-transformation "condition" 10
#4 Skew Rastrigin-Bueche separable, "condition" 10, skew-"condition" 100
#5 Linear slope, neutral extension outside the domain (but not flat)

#% LOW OR MODERATE CONDITION
#6 Attractive sector
#7 Step-ellipsoid, condition 100
#8 Rosenbrock, non-rotated
#9 Rosenbrock, rotated

#% HIGH CONDITION
#10 Ellipsoid with monotone x-transformation, condition 1e6
#11 Discus with monotone x-transformation, condition 1e6
#12 Bent cigar with asymmetric x-transformation, condition 1e6
#13 Sharp ridge, slope 1:100, condition 10
#14 Sum of different powers

#% MULTI-MODAL
#15 Rastrigin with asymmetric x-transformation, "condition" 10
#16 Weierstrass with monotone x-transformation, condition 100
#17 Schaffer F7 with asymmetric x-transformation, condition 10
#18 Schaffer F7 with asymmetric x-transformation, condition 1000
#19 F8F2 composition of 2-D Griewank-Rosenbrock

#% MULTI-MODAL WITH WEAK GLOBAL STRUCTURE
#20 Schwefel x*sin(x) with tridiagonal transformation, condition 10
#21 Gallagher 99 Gaussian peaks, global rotation, condition up to 1000
#22 Gallagher 99 Gaussian peaks, local rotations, condition up to 1000
#23 Katsuura
#24 Lunacek bi-Rastrigin, condition 100


def sortIndexEntries(indexEntries):
    """Puts the indexEntry instances into bins for the distribution."""

    if not indexEntries:
        return ()
    sorted = {}
    #The bins correspond to whether the dimension is greater than 10 or not.
    #Still needs to be sorted by functions.
    for i in indexEntries:
        if i.dim == 5 and i.funcId in range(0,6):
            sorted.setdefault('dim05separ',[]).append(i)
        elif i.dim == 5 and i.funcId in range(6,10):
            sorted.setdefault('dim05lcond',[]).append(i)
        elif i.dim == 5 and i.funcId in range(10,15):
            sorted.setdefault('dim05hcond',[]).append(i)
        elif i.dim == 5 and i.funcId in range(15,20):
            sorted.setdefault('dim05multi',[]).append(i)
        elif i.dim == 5 and i.funcId in range(20,25):
            sorted.setdefault('dim05mult2',[]).append(i)
        elif i.dim == 20 and i.funcId in range(0,6):
            sorted.setdefault('dim20separ',[]).append(i)
        elif i.dim == 20 and i.funcId in range(6,10):
            sorted.setdefault('dim20lcond',[]).append(i)
        elif i.dim == 20 and i.funcId in range(11,15):
            sorted.setdefault('dim20hcond',[]).append(i)
        elif i.dim == 20 and i.funcId in range(15,20):
            sorted.setdefault('dim20multi',[]).append(i)
        elif i.dim == 20 and i.funcId in range(20,25):
            sorted.setdefault('dim20mult2',[]).append(i)
    return sorted


def beautifyRLD(figHandle, figureName, maxEvalsFactor, fileFormat=('png','eps'),
             verbose=True):
    """Formats the figure of the run length distribution."""

    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')
    axisHandle.set_xlim((1.0,maxEvalsFactor))
    axisHandle.set_ylim((0.0,1.0))
    axisHandle.set_xlabel('log10 of FEvals / DIM')
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


def plotRLDistr(indexEntries, fvalueToReach, maxEvalsFactor=1e5, verbose=True):
    """Creates run length distributions from a sequence of indexEntries.

    Returns a plot of a run length distribution.
    args -- maxEvalsFactor : used for the limit of the plot.

    """

    x = []
    nn = 0
    for i in indexEntries:
        for j in i.hData:
            if j[0] <= fvalueToReach:
                x.extend(j[1:i.nbRuns+1]/i.dim)
                break
        nn += i.nbRuns

    x.sort()
    n = len(x)
    res = None
    if n > 0:
        x2 = scipy.hstack([scipy.repeat(x, 2), maxEvalsFactor])
        #not efficient if some vals are repeated a lot
        y2 = scipy.hstack([0.0, scipy.repeat(scipy.arange(1, n)/float(nn), 2),
                           float(n)/nn, float(n)/nn])
        res = plt.plot(x2, y2)

    return res


def beautifyFVD(figHandle, figureName, fvalueToReach, fileFormat=('png','eps'),
             verbose=True):
    """Formats the figure of the run length distribution."""

    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')
    #axisHandle.set_xlim((fvalueToReach))
    #axisHandle.invert_xaxis()
    axisHandle.set_ylim((0.0,1.0))
    axisHandle.set_xlabel('log10 of Delta f')
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


def plotFVDistr(indexEntries, maxEvals, fvalueToReach=1.e-8, verbose=True):
    """Creates final function values distributions from a sequence of indexEntries.

    Returns a plot of a run length distribution.
    args -- maxEvalsFactor : used for the limit of the plot.

    """

    x = []
    nn = 0
    for i in indexEntries:
        for j in i.vData:
            if j[0] >= maxEvals:
                x.extend(j[i.nbRuns+1:] / fvalueToReach)
                break
        nn += i.nbRuns

    x.sort()
    n = len(x)
    res = None
    if n > 0:
        x2 = scipy.hstack([scipy.repeat(x, 2), fvalueToReach])
        #not efficient if some vals are repeated a lot
        y2 = scipy.hstack([0.0, scipy.repeat(scipy.arange(1, n)/float(nn), 2),
                           float(n)/nn, float(n)/nn])
        res = plt.plot(x2, y2)

    return res


def main(indexEntries, valuesOfInterest, outputdir, verbose):
    """Generate image files of run length distribution figures."""
    sortedIndexEntries = sortIndexEntries(indexEntries)
    #set_trace()
    for key, indexEntries in sortedIndexEntries.iteritems():
        figureName = os.path.join(outputdir,'pprldistr_%s' %(key))
        #figureName = os.path.join(outputdir,'ppfvdistr_%s' %(key))

        fig = plt.figure()
        for j in range(len(valuesOfInterest)):
        #for j in [0]:
            maxEvalsFactor = 1e3 #TODO: Global?
            tmp = plotRLDistr(indexEntries, valuesOfInterest[j],
                              maxEvalsFactor, verbose)
            if not tmp is None:
                plt.setp(tmp, 'color', rldColors[j])
        beautifyRLD(fig, figureName, maxEvalsFactor, verbose=verbose)
        plt.close(fig)

