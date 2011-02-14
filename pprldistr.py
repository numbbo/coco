#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates run length distribution figures."""

from __future__ import absolute_import

import os
import numpy
import pickle
import matplotlib.pyplot as plt
from pdb import set_trace
from bbob_pproc import bootstrap
from bbob_pproc.ppfig import consecutiveNumbers, plotUnifLogXMarkers, saveFigure

__all__ = ['beautify', 'comp', 'main', 'plot']

rldColors = ('k', 'c', 'm', 'r', 'k', 'c', 'm', 'r', 'k', 'c', 'm', 'r')
rldUnsuccColors = ('k', 'c', 'm', 'k', 'c', 'm', 'k', 'c', 'm', 'k', 'c', 'm')  # should not be too short

rldStyles = ({'color': 'k', 'ls': '--'},
             {'color': 'c'},
             {'color': 'm', 'ls': '--'},
             {'color': 'r', 'linewidth': 3.},
             {'color': 'k'},
             {'color': 'c'},
             {'color': 'm'},
             {'color': 'r'},
             {'color': 'k'},
             {'color': 'c'},
             {'color': 'm'},
             {'color': 'r'})
rldUnsuccStyles = ({'color': 'k', 'ls': '--'},
                   {'color': 'c'},
                   {'color': 'm', 'ls': '--'},
                   {'color': 'k'},
                   {'color': 'c', 'ls': '--'},
                   {'color': 'm'},
                   {'color': 'k', 'ls': '--'},
                   {'color': 'c'},
                   {'color': 'm', 'ls': '--'},
                   {'color': 'k'},
                   {'color': 'c', 'ls': '--'},
                   {'color': 'm'})  # should not be too short
refcolor = 'wheat'

# Used as a global to store the largest xmax and align the FV ECD figures.
fmax = None
evalfmax = None
figformat = ('eps', 'pdf') # Controls the output when using the main method

#filename = 'pprldistrever_1e-8.pickle'
filename = 'pprldistr2009_1e-8.pickle'
filename = os.path.join(os.path.split(__file__)[0], filename)
isBestAlgorithmFound = True
try:
    f = open(filename,'r')
    dictbestalg = pickle.load(f)
except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    isBestAlgorithmFound = False
    print 'Could not find file: ', filename
else:
    f.close()

def plotECDF(x, n=None, plotArgs={}):
    if n is None:
        n = len(x)
    nx = len(x)
    if n == 0 or nx == 0:
        res = plt.plot([], [], **plotArgs)
    else:
        x2 = numpy.hstack(numpy.repeat(sorted(x), 2))
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, nx) / float(n), 2),
                           float(nx)/n])
        res = plt.plot(x2, y2, **plotArgs)
    return res

def beautifyECDF(axish=None):
    if axish is None:
        axish = plt.gca()
    plt.ylim(-0.01, 1.01)
    plt.yticks(numpy.array((0., 0.25, 0.5, 0.75, 1.0)),
               ('0.0', '', '0.5', '', '1.0'))
    axish.grid(True)
    #    handles = plt.getp(axish, 'children')
    #    for h in handles:
    #        plt.setp(h, 'clip_on', False)

def beautifyRLD():
    """Format and save the figure of the run length distribution."""

    # TODO: This method should not save file.

    axisHandle = plt.gca()
    axisHandle.set_xscale('log')
    #plt.axvline(x=maxEvalsF, color='k')

    axisHandle.set_xlabel('log10 of FEvals / DIM')
    axisHandle.set_ylabel('proportion of trials')
    # Grid options
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(numpy.log10(j)))
    axisHandle.set_xticklabels(newxtic)

    beautifyECDF()

    #set_trace()
    plt.legend(loc='best')
    #if legend:
        #axisHandle.legend(legend, locLegend)

def plotRLDistr(dsList, fvalueToReach, maxEvalsF, plotArgs={}):
    """Creates run length distributions from a sequence dataSetList.

    Keyword arguments:
    dsList -- Input data sets
    fvalueToReach -- dictionary of the function value to reach.
    maxEvalsF -- maximum number of function evaluations. Helps set the
    rightmost boundary
    plotArgs -- arguments to pass to the plot command

    Outputs:
    res -- resulting plot.

    """

    # TODO use **kwargs

    x = []
    nn = 0
    fsolved = set()
    funcs = set()
    for i in dsList:
        funcs.add(i.funcId)
        for j in i.evals:
            if j[0] <= fvalueToReach[i.funcId]:
                #set_trace()
                tmp = j[1:]
                x.extend(tmp[numpy.isfinite(tmp)]/float(i.dim))
                fsolved.add(i.funcId)
                #TODO: what if j[numpy.isfinite(j)] is empty
                break
        nn += i.nbRuns()
    #set_trace()
    kwargs = plotArgs.copy()
    try:
        label = ''
        if len(set(fvalueToReach.values())):
            label += '%+d:' % (numpy.log10(fvalueToReach[i.funcId]))
        label += '%d/%d' % (len(fsolved), len(funcs))
        kwargs['label'] = kwargs.setdefault('label', label)
    except TypeError: # fvalueToReach == 0. for instance...
        # no label
        pass

    #TODO: res = plotECDF(x, nn, kwargs) # Why not?
    n = len(x)
    if n == 0:
        res = plt.plot([], [], **kwargs)
    else:
        x.sort()
        x2 = numpy.hstack([numpy.repeat(x, 2), maxEvalsF ** 1.05])
        # maxEvalsF: used for the limit of the plot
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, n+1)/float(nn), 2)])
        res = plt.plot(x2, y2, **kwargs)

    return res#, fsolved, funcs

def plotERTDistr(dsList, fvalueToReach, plotArgs=None):
    """Creates estimated run time distributions from a DataSetList.

    Keyword arguments:
    dsList -- Input data sets
    fvalueToReach -- target function value
    plotArgs -- keyword arguments to pass to plot command

    Outputs:
    res -- resulting plot.

    """

    # TODO: **plotArgs

    x = []
    nn = 0
    samplesize = 1000 # samplesize is at least 1000
    percentiles = 0.5 # could be anything...

    for i in dsList:
        #funcs.add(i.funcId)
        for j in i.evals:
            if j[0] <= fvalueToReach[i.funcId]:
                runlengthsucc = j[1:][numpy.isfinite(j[1:])]
                runlengthunsucc = i.maxevals[numpy.isnan(j[1:])]
                tmp = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                       percentiles=percentiles,
                                       samplesize=samplesize)
                x.extend(tmp[1])
                break
        nn += samplesize
    #set_trace()
    res = plotECDF(x, nn, plotArgs)

    return res

def generateRLData(evals, targets):
    """Determine the running lengths for attaining the targets.

    Keyword arguments:
    evals -- numpy array with the first column corresponding to the
      function values and the following columns being the number of
      function evaluations for reaching this function value
    targets -- target function values of interest

    Output:
    list of arrays containing the number of function evaluations for
    reaching the target function values in target.

    """

    res = {}
    it = reversed(evals) # expect evals to be sorted by decreasing function values
    prevline = numpy.array([-numpy.inf] + [numpy.nan] * (numpy.shape(evals)[1]-1))
    try:
        line = it.next()
    except StopIteration:
        # evals is an empty array
        return res

    for t in sorted(targets):
        while line[0] <= t:
            prevline = line
            try:
                line = it.next()
            except StopIteration:
                break
        res[t] = prevline.copy()
    return res

def beautifyFVD(isStoringXMax=False):
    """Formats the figure of the run length distribution.

    This function is to be used with plotFVDistr

    Keyword arguments:
    isStoringMaxF -- if set to True, the first call beautifyFVD sets the
    global fmax and all subsequent call will have the same maximum xlim

    """

    # TODO: This method should not save file.

    axisHandle = plt.gca()
    axisHandle.set_xscale('log')

    if isStoringXMax:
        global fmax
    else:
        fmax = None

    if not fmax:
        xmin, fmax = plt.xlim()
    plt.xlim(1., fmax)

    #axisHandle.invert_xaxis()
    axisHandle.set_xlabel('log10 of Df / Dftarget')
    # axisHandle.set_ylabel('proportion of successful trials')
    # Grid options
    beautifyECDF()

    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(numpy.log10(j)))
    axisHandle.set_xticklabels(newxtic)
    axisHandle.set_yticklabels(())

def plotFVDistr(dataSetList, fvalueToReach, maxEvalsF, plotArgs={},
                 verbose=True):
    """Creates empirical cumulative distribution functions of final
    function values plot from a sequence of indexEntries.

    Keyword arguments:
    indexEntries -- sequence of IndexEntry to process.
    fvalueToReach -- float used for the lower limit of the plot
    maxEvalsF -- indicates which vertical data to display.
    verbose -- controls verbosity.

    Outputs: a plot of a run length distribution.

    """

    # TODO: **plotArgs

    x = []
    nn = 0
    for i in dataSetList:
        for j in i.funvals:
            if j[0] >= maxEvalsF * i.dim:
                break

        tmp = j[1:].copy() / fvalueToReach[i.funcId]
        tmp[tmp<=0.] = 1.
        # TODO: HACK, is almost ok since the xmin in the figure is 1
        x.extend(tmp)
        nn += i.nbRuns()

    res = plotECDF(x, nn, plotArgs)

    return res

def plotRLDistr2(dsList, fvalueToReach, maxEvalsF, plotArgs={}):
    """Creates run length distributions from a sequence dataSetList.

    Keyword arguments:
    dsList
    fvalueToReach
    maxEvalsF
    plotArgs

    Outputs:
    res -- resulting plot.
    fsolved -- number of different functions solved.
    funcs -- number of different function considered.

    """

    # TODO: check for plotRLDistr
    # TODO: **plotArgs

    x = []
    nn = 0
    fsolved = set()
    funcs = set()
    for i in dsList:
        funcs.add(i.funcId)
        try:
            target = fvalueToReach[i.funcId]
        except TypeError:
            target = fvalueToReach
        for j in i.evals:
            if j[0] <= target:
                #set_trace()
                tmp = j[1:]
                x.extend(tmp[numpy.isfinite(tmp)]/float(i.dim))
                fsolved.add(i.funcId)
                #TODO: what if j[numpy.isfinite(j)] is empty
                break
        nn += i.nbRuns()
    #set_trace()
    kwargs = plotArgs.copy()
    label = ''
    try:
        label += '%+d:' % (numpy.log10(target))
    except NameError:
        pass
    label += '%d/%d' % (len(fsolved), len(funcs))
    kwargs['label'] = kwargs.setdefault('label', label)

    #TODO: res = plotECDF(x, nn, kwargs) # Why not?
    n = len(x)
    if n == 0:
        res = plt.plot([], [], **kwargs)
    else:
        x.sort()
        x2 = numpy.hstack([numpy.repeat(x, 2), maxEvalsF ** 1.05])
        # maxEvalsF: used for the limit of the plot
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, n+1)/float(nn), 2)])
        res = plotUnifLogXMarkers(x2, y2, 1, logscale=True, **kwargs)

    return res#, fsolved, funcs

def plotECDF2(x, n=None, plotArgs={}):
    if n is None:
        n = len(x)
    nx = len(x)
    if n == 0 or nx == 0:
        res = plt.plot([], [], **plotArgs)
    else:
        x2 = numpy.hstack(numpy.repeat(sorted(x), 2))
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, nx) / float(n), 2),
                           float(nx)/n])
        #res = plt.plot(x2, y2, **plotArgs)
        res = plotUnifLogXMarkers(x2, y2, 1, logscale=True, **plotArgs)
    return res

def plotFVDistr2(dataSetList, fvalueToReach, maxEvalsF, plotArgs={}):
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
    for i in dataSetList:
        for j in i.funvals:
            if j[0] >= maxEvalsF * i.dim:
                break
        try:
            tmp = j[1:].copy() / fvalueToReach[i.funcId]
        except TypeError:
            tmp = j[1:].copy() / fvalueToReach
        tmp[tmp<=0.] = 1.
        # TODO: HACK, is almost ok since the xmin in the figure is 1
        x.extend(tmp)
        nn += i.nbRuns()

    res = plotECDF2(x, nn, plotArgs)

    return res

def comp(dsList0, dsList1, valuesOfInterest, isStoringXMax=False,
         outputdir='', info='default', verbose=True):
    """Generate figures of empirical cumulative distribution functions.
    Dashed lines will correspond to ALG0 and solid lines to ALG1.

    Keyword arguments:
    dsList0 -- list of DataSet instances for ALG0.
    dsList1 -- list of DataSet instances for ALG1
    valuesOfInterest -- target function values to be displayed.
    isStoringXMax -- if set to True, the first call BeautifyFVD sets the
      globals fmax and maxEvals and all subsequent calls will use these
      values as rightmost xlim in the generated figures.
    outputdir -- output directory (must exist)
    info --- string suffix for output file names.

    """

    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    maxEvalsFactor = max(max(i.mMaxEvals()/i.dim for i in dsList0),
                         max(i.mMaxEvals()/i.dim for i in dsList1))

    if isStoringXMax:
        global evalfmax
    else:
        evalfmax = None

    if not evalfmax:
        evalfmax = maxEvalsFactor

    filename = os.path.join(outputdir,'pprldistr_%s' %(info))
    fig = plt.figure()
    legend = []
    for j in range(len(valuesOfInterest)):
        kwargs = rldStyles[j].copy()
        kwargs['marker'] = '+'
        tmp = plotRLDistr2(dsList0, valuesOfInterest[j], evalfmax, kwargs)

        if not tmp is None:
            plt.setp(tmp, 'label', None) # Hack for the legend
            plt.setp(tmp, markersize=20.,
                     markeredgewidth=plt.getp(tmp[-1], 'linewidth'),
                     markeredgecolor=plt.getp(tmp[-1], 'color'),
                     markerfacecolor='none')

        kwargs = rldStyles[j].copy()
        kwargs['marker'] = 'o'
        tmp = plotRLDistr2(dsList1, valuesOfInterest[j], evalfmax, kwargs)

        if not tmp is None:
            ## Hack for the legend.
            plt.setp(tmp[-1], 'marker', '',
                     'label', ('%+d' % (numpy.log10(valuesOfInterest[j][1]))))
            plt.setp(tmp, markersize=15.,
                     markeredgewidth=plt.getp(tmp[-1], 'linewidth'),
                     markeredgecolor=plt.getp(tmp[-1], 'color'),
                     markerfacecolor='none')

    funcs = set(i.funcId for i in dsList0) | set(i.funcId for i in dsList1)
    text = 'f%s' % (consecutiveNumbers(sorted(funcs)))

    if isBestAlgorithmFound:
        d = set.union(set(i.dim for i in dsList0),
                      set(i.dim for i in dsList1)).pop() # Get only one element...
        for alg in dictbestalg:
            x = []
            nn = 0
            try:
                tmp = dictbestalg[alg]
                for f in funcs:
                    tmp[f][d] # simply test that they exists
            except KeyError:
                continue

            for f in funcs:
                tmp2 = tmp[f][d][0][1:]
                # [0], because the maximum #evals is also recorded
                # [1:] because the target function value is recorded
                x.append(tmp2[numpy.isnan(tmp2) == False])
                nn += len(tmp2)

            if x:
                x.append([(evalfmax*d) ** 1.05])
                x = numpy.hstack(x)

                plotECDF(x[numpy.isfinite(x)]/d, nn,
                         {'color': refcolor, 'ls': '-', 'zorder': -1})

    plt.axvline(max(i.mMaxEvals()/i.dim for i in dsList0), ls='--', color='k')
    plt.axvline(max(i.mMaxEvals()/i.dim for i in dsList1), color='k')
    beautifyRLD()
    plt.xlim(1.0, evalfmax ** 1.05)
    plt.text(0.5, 0.98, text, horizontalalignment="center",
             verticalalignment="top", transform=plt.gca().transAxes)
             #bbox=dict(ec='k', fill=False), 
    saveFigure(filename, figFormat=figformat, verbose=verbose)

    plt.close(fig)

def beautify():
    """Format the figure of the run length distribution."""

    plt.subplot(121)
    axisHandle = plt.gca()
    axisHandle.set_xscale('log')
    axisHandle.set_xlabel('log10 of FEvals / DIM')
    axisHandle.set_ylabel('proportion of trials')
    # Grid options
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(numpy.log10(j)))
    axisHandle.set_xticklabels(newxtic)

    beautifyECDF()

    plt.subplot(122)
    axisHandle = plt.gca()
    axisHandle.set_xscale('log')

    xmin, fmax = plt.xlim()
    plt.xlim(1., fmax)

    axisHandle.set_xlabel('log10 of Df / Dftarget')
    beautifyECDF()

    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(numpy.log10(j)))
    axisHandle.set_xticklabels(newxtic)
    axisHandle.set_yticklabels(())

    plt.gcf().set_size_inches(16.35, 6.175)
#     try:
#         set_trace()
#         plt.setp(plt.gcf(), 'figwidth', 16.35)
#     except AttributeError: # version error?
#         set_trace()
#         plt.setp(plt.gcf(), 'figsize', (16.35, 6.))

def plot(dsList, valuesOfInterest=(10., 1e-1, 1e-4, 1e-8), kwargs={}):
    """Plot ECDF of final function values and evaluations."""

    # TODO: **kwargs

    res = []
    plt.subplot(121)
    maxEvalsFactor = max(i.mMaxEvals()/i.dim for i in dsList)
    evalfmax = maxEvalsFactor

    legend = []
    for j in range(len(valuesOfInterest)):
        tmp = plotRLDistr2(dsList, valuesOfInterest[j], evalfmax, rldStyles[j % len(rldStyles)])
    res.extend(tmp)

    funcs = list(i.funcId for i in dsList)
    text = 'f%s' % (consecutiveNumbers(sorted(funcs)))

    plt.axvline(x=maxEvalsFactor, color='k')
    plt.text(0.5, 0.98, text, horizontalalignment="center",
             verticalalignment="top", transform=plt.gca().transAxes)
    plt.xlim(1.0, maxEvalsFactor ** 1.05)

    plt.subplot(122)
    for j in range(len(valuesOfInterest)):
        tmp = plotFVDistr2(dsList, valuesOfInterest[j], evalfmax,
                           rldStyles[j % len(rldStyles)])
    res.extend(tmp)

    tmp = numpy.floor(numpy.log10(evalfmax))
    # coloring right to left:
    maxEvalsF = numpy.power(10, numpy.arange(0, tmp))

    for j in range(len(maxEvalsF)):
        tmp = plotFVDistr2(dsList, valuesOfInterest[-1], maxEvalsF[j],
                           rldUnsuccStyles[j % len(rldUnsuccStyles)])
    plt.text(0.98, 0.02, text, horizontalalignment="right",
             transform=plt.gca().transAxes)

def main(dsList, valuesOfInterest, isStoringXMax=False, outputdir='',
         info='default', verbose=True):
    """Generate figures of empirical cumulative distribution functions.

    Keyword arguments:
    dsList -- list of DataSet instances to process.
    valuesOfInterest -- target function values to be displayed.
    isStoringXMax -- if set to True, the first call geautifyFVD sets the
      globals fmax and maxEvals and all subsequent calls will use these
      values as rightmost xlim in the generated figures.
    outputdir -- output directory (must exist)
    info --- string suffix for output file names.

    Outputs:
    Image files of the empirical cumulative distribution functions.

    """

    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    maxEvalsFactor = max(i.mMaxEvals()/i.dim for i in dsList)
    #maxEvalsFactorCeil = numpy.power(10,
                                     #numpy.ceil(numpy.log10(maxEvalsFactor)))

    if isStoringXMax:
        global evalfmax
    else:
        evalfmax = None

    if not evalfmax:
        evalfmax = maxEvalsFactor

    filename = os.path.join(outputdir,'pprldistr_%s' %(info))
    fig = plt.figure()
    legend = []
    for j in range(len(valuesOfInterest)):
        tmp = plotRLDistr2(dsList, valuesOfInterest[j], evalfmax, rldStyles[j % len(rldStyles)])
        #if not tmp is None:
            #for attr in rldStyles[j % len(rldStyles)]:
                #plt.setp(tmp, attr, rldStyles[j % len(rldStyles)][attr])

    funcs = list(i.funcId for i in dsList)
    text = 'f%s' % (consecutiveNumbers(sorted(funcs)))

    if isBestAlgorithmFound:
        d = set(i.dim for i in dsList).pop() # Get only one element...
        for alg in dictbestalg:
            x = []
            nn = 0
            try:
                tmp = dictbestalg[alg]
                for f in funcs:
                    tmp[f][d] # simply test that they exists
            except KeyError:
                continue

            for f in funcs:
                tmp2 = tmp[f][d][0][1:]
                # [0], because the maximum #evals is also recorded
                # [1:] because the target function value is recorded
                x.append(tmp2[numpy.isnan(tmp2) == False])
                nn += len(tmp2)

            if x:
                x.append([(evalfmax*d) ** 1.05])
                x = numpy.hstack(x)

                plotECDF(x[numpy.isfinite(x)]/float(d), nn,
                         {'color': refcolor, 'ls': '-', 'zorder': -1})

    plt.axvline(x=maxEvalsFactor, color='k')
    beautifyRLD()
    plt.xlim(1.0, evalfmax ** 1.05)
    plt.text(0.5, 0.98, text, horizontalalignment="center",
             verticalalignment="top", transform=plt.gca().transAxes)
             #bbox=dict(ec='k', fill=False), 
    saveFigure(filename, figFormat=figformat, verbose=verbose)

    plt.close(fig)

    filename = os.path.join(outputdir,'ppfvdistr_%s' %(info))
    fig = plt.figure()
    for j in range(len(valuesOfInterest)):
        tmp = plotFVDistr2(dsList, valuesOfInterest[j], evalfmax,
                           rldStyles[j % len(rldStyles)])

    tmp = numpy.floor(numpy.log10(evalfmax))
    # coloring left to right:
    #maxEvalsF = numpy.power(10, numpy.arange(tmp, 0, -1) - 1)
    # coloring right to left:
    maxEvalsF = numpy.power(10, numpy.arange(0, tmp))

    #set_trace()
    for j in range(len(maxEvalsF)):
        tmp = plotFVDistr2(dsList, valuesOfInterest[-1], maxEvalsF[j],
                           rldUnsuccStyles[j % len(rldUnsuccStyles)])

    beautifyFVD(isStoringXMax=isStoringXMax)
    plt.text(0.98, 0.02, text, horizontalalignment="right",
             transform=plt.gca().transAxes)
             #bbox=dict(ec='k', fill=False), 
    saveFigure(filename, figFormat=figformat, verbose=verbose)

    plt.close(fig)

    #plt.rcdefaults()

