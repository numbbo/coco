#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""For generating empirical cumulative distribution function figures.

The outputs show empirical cumulative distribution functions (ECDFs) of
the running times of trials. These ECDFs show on the y-axis the fraction
of cases for which the running time (left subplots) or the df-value
(right subplots) was smaller than the value given on the x-axis. On the
left, ECDFs of the running times from trials are shown for different
target values. Light brown lines in the background show ECDFs for target
value 1e-8 of all algorithms benchmarked during BBOB-2009. On the right,
ECDFs of df-values from all trials are shown for different numbers of
function evaluations.

**Example**

.. plot::
   :width: 75%
   
   import urllib
   import tarfile
   import glob
   from pylab import *
   import bbob_pproc as bb
    
   # Collect and unarchive data (3.4MB)
   dataurl = 'http://coco.lri.fr/BBOB2009/pythondata/BIPOP-CMA-ES.tar.gz'
   filename, headers = urllib.urlretrieve(dataurl)
   archivefile = tarfile.open(filename)
   archivefile.extractall()
    
   # Empirical cumulative distribution function figure
   ds = bb.load(glob.glob('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f0*_20.pickle'))
   figure()
   bb.pprldistr.plot(ds)
   bb.pprldistr.beautify() # resize the window to view whole figure

"""
from __future__ import absolute_import

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
try:
    from matplotlib.transforms import blended_transform_factory as blend
except ImportError:
    # compatibility matplotlib 0.8
    from matplotlib.transforms import blend_xy_sep_transform as blend
from pdb import set_trace
from bbob_pproc import bootstrap
from bbob_pproc.ppfig import consecutiveNumbers, plotUnifLogXMarkers, saveFigure

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


def beautifyECDF():
    """Generic formatting of ECDF figures."""
    plt.ylim(-0.01, 1.01)
    plt.yticks(np.array((0., 0.25, 0.5, 0.75, 1.0)),
               ('0.0', '', '0.5', '', '1.0'))
    plt.grid(True)
    xmin, xmax = plt.xlim()
    c = plt.gca().get_children()
    for i in c:
        try:
            ydata = i.get_ydata()
            ydata = np.hstack((ydata[0], ydata, ydata[-1]))
            xdata = i.get_xdata()
            xdata = np.hstack((xmin, xdata, xmax))
            plt.setp(i, 'xdata', xdata, 'ydata', ydata)
        except (AttributeError, IndexError):
            pass

def beautifyRLD():
    """Format and save the figure of the run length distribution."""
    a = plt.gca()
    a.set_xscale('log')
    #plt.axvline(x=maxEvalsF, color='k')
    a.set_xlabel('log10 of FEvals / DIM')
    a.set_ylabel('proportion of trials')
    xtic = a.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(np.log10(j)))
    a.set_xticklabels(newxtic)
    beautifyECDF()

def beautifyFVD(isStoringXMax=False, ylabel=True):
    """Formats the figure of the run length distribution.

    This function is to be used with :py:func:`plotFVDistr`

    :param bool isStoringMaxF: if set to True, the first call
                               :py:func:`beautifyFVD` sets the global
                               :py:data:`fmax` and all subsequent call
                               will have the same maximum xlim
    :param bool ylabel: if True, y-axis will be labelled.

    """
    a = plt.gca()
    a.set_xscale('log')

    if isStoringXMax:
        global fmax
    else:
        fmax = None

    if not fmax:
        xmin, fmax = plt.xlim()
    plt.xlim(1., fmax)

    #axisHandle.invert_xaxis()
    a.set_xlabel('log10 of Df / Dftarget')
    if ylabel:
        a.set_ylabel('proportion of successful trials')
    xtic = a.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(np.log10(j)))
    a.set_xticklabels(newxtic)
    if not ylabel:
        a.set_yticklabels(())
    beautifyECDF()

def plotECDF(x, n=None, **plotArgs):
    """Plot an empirical cumulative distribution function.

    :param seq x: data
    :param int n: number of samples, if not provided len(x) is used
    :param plotArgs: optional keyword arguments provided to plot.

    :returns: handles of the plot elements.

    """
    if n is None:
        n = len(x)

    nx = len(x)
    if n == 0 or nx == 0:
        res = plt.plot([], [], **plotArgs)
    else:
        x = sorted(x) # do not sort in place
        x = np.hstack((x, x[-1]))
        y = np.hstack((np.arange(0., nx) / n, float(nx)/n))
        res = plotUnifLogXMarkers(x, y, 1, drawstyle='steps', logscale=True,
                                  **plotArgs)
    return res

def plotERTDistr(dsList, fvalueToReach, **plotArgs):
    """Creates estimated run time distributions from a DataSetList.

    :keyword DataSet dsList: Input data sets
    :keyword dict fvalueToReach: target function values
    :keyword plotArgs: keyword arguments to pass to plot command

    :return: resulting plot.

    """
    x = []
    nn = 0
    samplesize = 1000 # samplesize is at least 1000
    percentiles = 0.5 # could be anything...

    for i in dsList:
        #funcs.add(i.funcId)
        for j in i.evals:
            if j[0] <= fvalueToReach[i.funcId]:
                runlengthsucc = j[1:][np.isfinite(j[1:])]
                runlengthunsucc = i.maxevals[np.isnan(j[1:])]
                tmp = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                       percentiles=percentiles,
                                       samplesize=samplesize)
                x.extend(tmp[1])
                break
        nn += samplesize
    res = plotECDF(x, nn, **plotArgs)

    return res

def plotRLDistr(dsList, fvalueToReach, maxEvalsF, **plotArgs):
    """Creates run length distributions from a sequence dataSetList.

    :param DataSetList dsList: Input data sets
    :param dict fvalueToReach: function value to reach.
    :param float maxEvalsF: maximum number of function evaluations.
                            Helps set the rightmost boundary
    :param plotArgs: additional arguments passed to the plot command

    :returns: handles of the resulting plot.

    """
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
                x.extend(tmp[np.isfinite(tmp)]/float(i.dim))
                fsolved.add(i.funcId)
                #TODO: what if j[np.isfinite(j)] is empty
                break
        nn += i.nbRuns()
    kwargs = plotArgs.copy()
    label = ''
    try:
        label += '%+d:' % (np.log10(target))
    except NameError:
        pass
    label += '%d/%d' % (len(fsolved), len(funcs))
    kwargs['label'] = kwargs.setdefault('label', label)

    res = plotECDF(x, nn, **kwargs)
    return res

def plotFVDistr(dsList, fvalueToReach, maxEvalsF, **plotArgs):
    """Creates ECDF of final function values plot from a DataSetList.

    :param dsList: data sets
    :param dict or float fvalueToReach: used for the lower limit of the
                                        plot
    :param float maxEvalsF: indicates which vertical data to display.
    :param plotArgs: additional arguments passed to plot

    :returns: handle

    """
    x = []
    nn = 0
    for i in dsList:
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
    res = plotECDF(x, nn, **plotArgs)
    return res

def comp(dsList0, dsList1, valuesOfInterest, isStoringXMax=False,
         outputdir='', info='default', verbose=True):
    """Generate figures of ECDF for 2 algorithms.

    Dashed lines will correspond to ALG0 and solid lines to ALG1.

    :param DataSetList dsList0: list of DataSet instances for ALG0
    :param DataSetList dsList1: list of DataSet instances for ALG1
    :param seq valuesOfInterest: target function values to be displayed
    :param bool isStoringXMax: if set to True, the first call
                               :py:func:`beautifyFVD` sets the globals
                               :py:data:`fmax` and :py:data:`maxEvals`
                               and all subsequent calls will use these
                               values as rightmost xlim in the generated
                               figures.
    :param string outputdir: output directory (must exist)
    :param string info: string suffix for output file names.
    :param bool verbose: control verbosity

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
        tmp = plotRLDistr(dsList0, valuesOfInterest[j], evalfmax, **kwargs)

        if not tmp is None:
            plt.setp(tmp, 'label', None) # Hack for the legend
            plt.setp(tmp, markersize=20.,
                     markeredgewidth=plt.getp(tmp[-1], 'linewidth'),
                     markeredgecolor=plt.getp(tmp[-1], 'color'),
                     markerfacecolor='none')

        kwargs = rldStyles[j].copy()
        kwargs['marker'] = 'o'
        tmp = plotRLDistr(dsList1, valuesOfInterest[j], evalfmax, **kwargs)

        if not tmp is None:
            ## Hack for the legend.
            plt.setp(tmp[-1], 'marker', '',
                     'label', ('%+d' % (np.log10(valuesOfInterest[j][1]))))
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
                x.append(tmp2[np.isnan(tmp2) == False])
                nn += len(tmp2)

            if x:
                x.append([(evalfmax*d) ** 1.05])
                x = np.hstack(x)

                plotECDF(x[np.isfinite(x)]/d, nn,
                         color=refcolor, ls='-', zorder=-1)

    plt.axvline(max(i.mMaxEvals()/i.dim for i in dsList0), ls='--', color='k')
    plt.axvline(max(i.mMaxEvals()/i.dim for i in dsList1), color='k')
    beautifyRLD()
    plt.legend(loc='best')
    plt.xlim(1.0, evalfmax ** 1.05)
    plt.text(0.5, 0.98, text, horizontalalignment="center",
             verticalalignment="top", transform=plt.gca().transAxes)
             #bbox=dict(ec='k', fill=False), 
    saveFigure(filename, figFormat=figformat, verbose=verbose)

    plt.close(fig)

def beautify():
    """Format the figure of the run length distribution.
    
    Used in conjunction with plot method.

    """
    plt.subplot(121)
    axisHandle = plt.gca()
    axisHandle.set_xscale('log')
    axisHandle.set_xlabel('log10 of FEvals / DIM')
    axisHandle.set_ylabel('proportion of trials')
    # Grid options
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(np.log10(j)))
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
        newxtic.append('%d' % round(np.log10(j)))
    axisHandle.set_xticklabels(newxtic)
    axisHandle.set_yticklabels(())

    plt.gcf().set_size_inches(16.35, 6.175)
#     try:
#         set_trace()
#         plt.setp(plt.gcf(), 'figwidth', 16.35)
#     except AttributeError: # version error?
#         set_trace()
#         plt.setp(plt.gcf(), 'figsize', (16.35, 6.))

def plot(dsList, valuesOfInterest=(10., 1e-1, 1e-4, 1e-8), **kwargs):
    """Plot ECDF of final function values and evaluations."""

    res = []
    plt.subplot(121)
    maxEvalsFactor = max(i.mMaxEvals()/i.dim for i in dsList)
    evalfmax = maxEvalsFactor

    legend = []
    for j in range(len(valuesOfInterest)):
        tmp = plotRLDistr(dsList, valuesOfInterest[j], evalfmax,
                          **rldStyles[j % len(rldStyles)])
    res.extend(tmp)

    funcs = list(i.funcId for i in dsList)
    text = 'f%s' % (consecutiveNumbers(sorted(funcs)))

    plt.axvline(x=maxEvalsFactor, color='k')
    plt.text(0.5, 0.98, text, horizontalalignment="center",
             verticalalignment="top", transform=plt.gca().transAxes)
    plt.xlim(1.0, maxEvalsFactor ** 1.05)

    plt.subplot(122)
    for j in range(len(valuesOfInterest)):
        tmp = plotFVDistr(dsList, valuesOfInterest[j], evalfmax,
                          **rldStyles[j % len(rldStyles)])
    res.extend(tmp)

    tmp = np.floor(np.log10(evalfmax))
    # coloring right to left:
    maxEvalsF = np.power(10, np.arange(0, tmp))

    for j in range(len(maxEvalsF)):
        tmp = plotFVDistr(dsList, valuesOfInterest[-1], maxEvalsF[j],
                          **rldUnsuccStyles[j % len(rldUnsuccStyles)])
    plt.text(0.98, 0.02, text, horizontalalignment="right",
             transform=plt.gca().transAxes)

def main(dsList, valuesOfInterest, isStoringXMax=False, outputdir='',
         info='default', verbose=True):
    """Generate figures of empirical cumulative distribution functions.

    :param DataSetList dsList: list of DataSet instances to process.
    :param seq valuesOfInterest: target function values to be displayed
    :param bool isStoringXMax: if set to True, the first call
                               :py:func:`beautifyFVD` sets the
                               globals :py:data:`fmax` and
                               :py:data:`maxEvals` and all subsequent
                               calls will use these values as rightmost
                               xlim in the generated figures.
    :param string outputdir: output directory (must exist)
    :param string info: string suffix for output file names.
    :param bool verbose: control verbosity
    
    """
    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    maxEvalsFactor = max(i.mMaxEvals()/i.dim for i in dsList)
    #maxEvalsFactorCeil = np.power(10,
                                     #np.ceil(np.log10(maxEvalsFactor)))

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
        tmp = plotRLDistr(dsList, valuesOfInterest[j], evalfmax,
                          **rldStyles[j % len(rldStyles)])
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
                x.append(tmp2[np.isnan(tmp2) == False])
                nn += len(tmp2)

            if x:
                x.append([(evalfmax*d) ** 1.05])
                x = np.hstack(x)

                plotECDF(x[np.isfinite(x)]/float(d), nn,
                         color=refcolor, ls='-', zorder=-1)

    plt.axvline(x=maxEvalsFactor, color='k')
    plt.legend(loc='best')
    plt.xlim(1.0, evalfmax ** 1.05)
    plt.text(0.5, 0.98, text, horizontalalignment="center",
             verticalalignment="top", transform=plt.gca().transAxes)
             #bbox=dict(ec='k', fill=False), 
    beautifyRLD()
    saveFigure(filename, figFormat=figformat, verbose=verbose)

    plt.close(fig)

    filename = os.path.join(outputdir,'ppfvdistr_%s' %(info))
    fig = plt.figure()
    for j in range(len(valuesOfInterest)):
        tmp = plotFVDistr(dsList, valuesOfInterest[j], evalfmax,
                          **rldStyles[j % len(rldStyles)])

    tmp = np.floor(np.log10(evalfmax))
    # coloring left to right:
    #maxEvalsF = np.power(10, np.arange(tmp, 0, -1) - 1)
    # coloring right to left:
    maxEvalsF = np.power(10, np.arange(0, tmp))

    #set_trace()
    for j in range(len(maxEvalsF)):
        tmp = plotFVDistr(dsList, valuesOfInterest[-1], maxEvalsF[j],
                          **rldUnsuccStyles[j % len(rldUnsuccStyles)])

    plt.text(0.98, 0.02, text, horizontalalignment="right",
             transform=plt.gca().transAxes)
             #bbox=dict(ec='k', fill=False), 
    beautifyFVD(isStoringXMax=isStoringXMax, ylabel=False)
    saveFigure(filename, figFormat=figformat, verbose=verbose)

    plt.close(fig)

    #plt.rcdefaults()
