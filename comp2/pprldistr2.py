#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates run length distribution figures for the comparison of 2 algorithms."""

from __future__ import absolute_import

import os
import numpy
import matplotlib.pyplot as plt
from bbob_pproc import toolsstats
from bbob_pproc.ppfig import saveFigure, consecutiveNumbers, plotUnifLogXMarkers
from pdb import set_trace

#__all__ = []

rldStyles = ({'color': 'k', 'ls': '-'},
             {'color': 'c'},
             {'color': 'm', 'ls': '-'},
             {'color': 'r'},
             {'color': 'b'},
             {'color': 'g'},
             {'color': 'm'},
             {'color': 'r'},
             {'color': 'k'},
             {'color': 'c'},
             {'color': 'm'},
             {'color': 'r'})

# Used as a global to store the largest xmax and align the FV ECD figures.
fmax = None
evalfmax = None

def beautify(handles):
    """Format the figure of the run length distribution."""

    axisHandle = plt.gca()
    axisHandle.set_xscale('log')
    plt.axvline(1, ls='-', color='k');  # symmetry line for ERT1/ERT0 = 1
    xlim = max(numpy.abs(numpy.log10(plt.xlim())))
    xlim = (min(0.1, 10.**(-xlim)), max(10., 10.**(xlim)))
    plt.axhline(0.5, ls=':', color='k', lw=2);  # symmetry line at y=0.5
    plt.xlim(xlim)
    plt.yticks(numpy.array((0., 0.25, 0.5, 0.75, 1.0)),
               ('', '', '', '', ''))
    plt.ylim(-0.01, 1.01)
    axisHandle.set_xlabel('log10 of FEvals ratio')
    axisHandle.set_ylabel('proportion')
    axisHandle.grid(True)
    xticks = axisHandle.get_xticks()
    newxticks = []
    for i in xticks:
        newxticks.append('%d' % round(numpy.log10(i)))
    axisHandle.set_xticklabels(newxticks)

    # Prolong to the boundary...
    xmin, xmax = plt.xlim()
    for i in handles:
        try:
            xdata, ydata = i.get_data()
        except AttributeError:
            xdata = i.get_xdata()
            ydata = i.get_ydata()
        if len(xdata) == 0 or len(ydata) == 0:
            continue
        if not hasattr(xdata, 'dtype') or xdata.dtype != float:
            xdata = numpy.array(xdata, dtype=float) 
        xdata = numpy.insert(xdata, 0, xmin)
        xdata = numpy.insert(xdata, len(xdata), xmax)
        ydata = numpy.insert(ydata, 0, ydata[0])
        ydata = numpy.insert(ydata, len(ydata), ydata[-1])
        i.set_data(xdata, ydata)

    plt.legend(loc='best')

    # Inverted xticks
    x = axisHandle.get_xticks(minor=True)
    # Operation for reverting the ticks for x < 1
    x[x<1] = sorted(1/(x[x<1]*numpy.power(10, -2*numpy.floor(numpy.log10(x[x<1]))-1)))
    x = x[(x<xmax) * (x>xmin)] # why?
    axisHandle.set_xticks(x, minor=True)

def computeERT(fevals, maxevals):
    data = fevals.copy()
    success = (numpy.isnan(data)==False)
    if any(numpy.isnan(data)):
        data[numpy.isnan(data)] = maxevals[numpy.isnan(data)]
    res = toolsstats.sp(data, issuccessful=success)
    return res[0]

def plotLogAbs(dsList0, dsList1, fvalueToReach, verbose=True):
    """Creates ECDF of run length ratios.

    :param DataSetList dsList0: reference
    :param DataSetList dsList1: data set list of algorithm of interest
    :param seq fvalueToReach: target function values
    :param bool verbose: controls verbosity

    :returns: handles

    """
    res = []
    dictFunc0 = dsList0.dictByFunc()
    dictFunc1 = dsList1.dictByFunc()
    evals0 = {}
    evals1 = {}

    targets = fvalueToReach  # yet to come: TargetValues class needs to be considered

    succ0 = [0] * len(targets)
    succ1 = [0] * len(targets)

    # TODO: check all functions are there...
    for func in set(dictFunc0.keys()) & set(dictFunc1.keys()):
        i0 = dictFunc0[func][0]
        i1 = dictFunc1[func][0]

        #tmp = list(i[func] for i in fvalueToReach)
        #if tmp != targets:
        #    set_trace() # should not occur
        #    warnings.warn('')

        evals0[func] = i0.detEvals(targets)
        for i, evals in enumerate(evals0[func]):
            tmp = numpy.isnan(evals)
            evals[tmp] = numpy.inf
            if not tmp.all():
                succ0[i] += 1

        evals1[func] = i1.detEvals(targets)
        for i, evals in enumerate(evals1[func]):
            tmp = numpy.isnan(evals)
            evals[tmp] = numpy.inf
            if not tmp.all():
                succ1[i] += 1

    for i, target in enumerate(reversed(sorted(targets))):

        x = []
        for func in evals0:
            # Compute the pair-wise ratio
            tmp1 = numpy.reshape(evals1[func][i], (1, len(evals1[func][i])))
            tmp0 = numpy.reshape(evals0[func][i], (len(evals0[func][i]), 1))
            try:
                x.append((tmp1/tmp0).flatten())  # inf/inf results in nan
            except FloatingPointError: 
                if numpy.isfinite(tmp1).all() or numpy.isfinite(tmp1).all():
                    raise
                
                #TODO: check division, check numpy.inf...

        label = '%+d: %d/%d' % (numpy.log10(target), succ1[i], succ0[i])
        if len(x) > 0:  # prevent warning/error
            x = numpy.hstack(x)
            x = x[numpy.isnan(x)==False] # Is it correct?
        n = len(x)

        if n == 0:
            res.extend(plt.plot([], [], label=label, linewidth=3., **rldStyles[i]))
            continue # no plot?

        x.sort()
        #Catch negative values: zeros are not a problem...
        tmp = len(list(i for i in x if i <= 0))
        x = x[tmp:]
        #Catch inf, those could be a problem with the log scale...
        #tmp2 = 0
        tmp2 = len(list(i for i in x if i > 0 and numpy.isinf(i)))
        if tmp2 > 0:
            x = x[:-tmp2]

        #xbound = max(abs(numpy.floor(numpy.log10(x[0]))),
        #             abs(numpy.ceil(numpy.log10(x[-1]))))
        if len(x) == 0:
            res.append(plt.axhline(tmp/float(n), label=label,
                                   linewidth=3., **rldStyles[i]))
            # tmp/float(n) == (n-tmp2)/float(n) # TODO: check
        else:
            x2 = numpy.hstack([numpy.repeat(x, 2)])
            #maxEvalsF: used for the limit of the plot.
            y2 = numpy.hstack([tmp/float(n),
                               numpy.repeat(numpy.arange(tmp+1, n-tmp2) / float(n), 2),
                               (n-tmp2)/float(n)])
            #res.extend(plt.plot(x2, y2, label=label, linewidth=3., **rldStyles[i]))
            plotArgs = rldStyles[i].copy()
            plotArgs['label'] = label
            plotArgs['linewidth'] = 3.
            #res.extend(plotUnifLogXMarkers(x2, y2, 3, plotArgs))
            res.append(plotUnifLogXMarkers(x2, y2, nbperdecade=3, logscale=False, **plotArgs)[0])

        # TODO: check if all of evalsX[func] is numpy.inf and so on...

    return res

def plotLogRel(indexEntries0, indexEntries1, isByInstance=True, verbose=True):
    """Creates one run length distribution from a sequence of indexEntries.

    The function and dimension are given.
    Keyword arguments:
    indexEntries0 -- reference
    indexEntries1
    isByInstance -- loop over the function instances instead of the functions
    verbose

    Outputs:
    res -- resulting plot.
    fsolved -- number of different functions solved.
    funcs -- number of different function considered.
    """

    res = [] #List of the plot handles.

    maxevals = 0
    for i in indexEntries0:
        if i.mMaxEvals() > maxevals:
            maxevals = i.mMaxEvals()
    for i in indexEntries1:
        if i.mMaxEvals() > maxevals:
            maxevals = i.mMaxEvals()

    funIndexEntries0 = indexEntries0.dictByFunc()
    funIndexEntries1 = indexEntries1.dictByFunc()
    #Suppose we only have one dimension...
    curevals = indexEntries0[0].dim # is supposed to be the same as i1.dim    

    while curevals < maxevals:
        x = []
        nn = 0
        for func in set(funIndexEntries0.keys()).union(funIndexEntries1.keys()):
            try:
                i0 = funIndexEntries0[func][0]
                i1 = funIndexEntries1[func][0]
            except KeyError:
                continue

            #Could gain time by storing the iterators over all functions...
            #Get the curDf
            it0 = iter(i0.funvals)
            it1 = iter(i1.funvals)
            try:
                nline0 = it0.next()
                while nline0[0] < curevals:
                    line0 = nline0
                    nline0 = it0.next()
            except StopIteration:
                pass #we keep the last line obtained.
            try:
                nline1 = it1.next()
                while nline1[0] < curevals:
                    line1 = nline1
                    nline1 = it1.next()
            except StopIteration:
                pass #we keep the last line obtained.

            ERT = []
            if not isByInstance:
                curDf = min(numpy.append(line0[1:], line1[1:]))
                for i, entry in enumerate((i0, i1)):
                    for j in entry.evals:
                        if j[0] <= curDf:
                            break
                    ERT.append(computeERT(j, entry.maxevals))

                if not numpy.isnan(ERT[1]/ERT[0]):
                    x.append(ERT[1]/ERT[0])
                    nn += 1
                #TODO check it is the same as ERT[1] ???
            else:
                lines = (line0, line1)
                #Set curDf
                curDf = {}
                for k in (set(i0.instancenumbers) & set(i1.instancenumbers)):
                    curDf[k] = []
                    for i, entry in enumerate((i0, i1)):
                        for j in range(len(entry.instancenumbers)):
                            if entry.instancenumbers[j] == k:
                                curDf[k].append(lines[i][1+j])
                    curDf[k] = min(curDf[k])

                for i, entry in enumerate((i0, i1)):
                    dictinstance = {}
                    for j in range(len(entry.instancenumbers)):
                        dictinstance.setdefault(entry.instancenumbers[j], []).append(j)
                    ERT.append(dictinstance.copy())
                    for k in dictinstance:
                        for j in entry.evals:
                            if j[0] <= curDf[k]:
                                break
                        ERT[i][k] = computeERT(j[list(1+i for i in dictinstance[k])],
                            entry.maxevals[list(i for i in dictinstance[k])])

                s0 = set(ERT[0])
                s1 = set(ERT[1])
                #Could be done simpler
                for j in s0 - s1:
                    x.append(0)
                    nn += 1
                for j in s0 & s1:
                    if not numpy.isnan(ERT[1][j]/ERT[0][j]):
                        x.append(ERT[1][j]/ERT[0][j])
                        nn += 1
                for j in s1 - s0:
                    x.append(numpy.inf)  # 12/02/25: was inf and therefore could never execute
                    nn += 1

        label = '1e%+d * DIM' % numpy.log10(curevals/indexEntries0[0].dim)
        n = len(x)
        x.sort()
        #Catch negative values, those could be a problem with the log scale...
        #tmp = 0
        tmp = len(list(i for i in x if i <= 0))
        x = x[tmp:]
        #Catch inf, those could be a problem with the log scale...
        #tmp2 = 0
        tmp2 = len(list(i for i in x if numpy.isinf(i))) #Also catches negative inf
        if tmp2 > 0:
            x = x[:-tmp2]

        if not x:
            res = plt.plot([], [], label=label)
        else:
            xbound = max(abs(numpy.floor(numpy.log10(x[0]))),
                         abs(numpy.ceil(numpy.log10(x[-1]))))
            x2 = numpy.hstack([10.**(-xbound),
                               numpy.repeat(x, 2),
                               10.**xbound])
            #maxEvalsF: used for the limit of the plot.
            y2 = numpy.hstack([tmp/float(nn), tmp/float(nn),
                               numpy.repeat(numpy.arange(tmp+1, n-tmp2) / float(nn), 2),
                               (n-tmp2)/float(nn), (n-tmp2)/float(nn)])
            res.append(plt.plot(x2, y2, label=label))

        #Update the curDf
        curevals *= 10

    return res#, fsolved, funcs

def main(dsList0, dsList1, valuesOfInterest=None,
         outputdir='', info='default', verbose=True):
    """Generate figures of empirical cumulative distribution functions.

    :param DataSetList dsList0: data set of reference algorithm
    :param DataSetList dsList1: data set of algorithm of concern
    :param sequence valuesOfInterest: target function values to be
                                      displayed
    :param bool isStoringXMax: if set to True, the first call BeautifyVD
                               sets the globals :py:data:`fmax` and 
                               :py:data:`maxEvals` and all subsequent
                               calls will use these values as rightmost
                               xlim in the generated figures.
    :param string outputdir: output directory (must exist)
    :param string info: string suffix for output file names.
    :param bool verbose: control verbosity

    Outputs:
    Image files of the empirical cumulative distribution functions.

    """
    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    figureName = os.path.join(outputdir,'pplogabs_%s' %(info))

    handles = plotLogAbs(dsList0, dsList1,
                         valuesOfInterest, verbose=verbose)

    beautify(handles)

    funcs = set(dsList0.dictByFunc().keys()) & set(dsList1.dictByFunc().keys())
    text = 'f%s' % consecutiveNumbers(sorted(funcs))
    if len(dsList0.dictByDim().keys()) == len(dsList1.dictByDim().keys()) == 1: 
        text += ',%d-D' % dsList0.dictByDim().keys()[0]

    plt.text(0.98, 0.02, text, horizontalalignment="right",
             transform=plt.gca().transAxes)

    saveFigure(figureName, verbose=verbose)
    plt.close()

    #plt.rcdefaults()

