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

CAVEAT: the naming conventions in this module mix up ERT (an estimate 
of the expected running length) and run lengths. 

"""
from __future__ import absolute_import

import os
import numpy as np
import pickle, gzip
import matplotlib.pyplot as plt
from pdb import set_trace
from bbob_pproc import toolsstats, genericsettings, pproc
from bbob_pproc.ppfig import consecutiveNumbers, plotUnifLogXMarkers, saveFigure, logxticks

single_target_values = pproc.TargetValues((10., 1e-1, 1e-4, 1e-8))  # possibly changed in config
single_runlength_factors = [0.5, 1.2, 3, 10] + [10 ** i for i in range(2, 12)]
# TODO: the method names in this module seem to be overly unclear or misleading and should be revised. 
   
refcolor = 'wheat'
nbperdecade = 1  # markers in x-axis decades in ecdfs

runlen_xlimits_max = None  # is possibly manipulated in config
runlen_xlimits_min = 1  # set to 10**-0.5 in runlength case in config
# Used as a global to store the largest xmax and align the FV ECD figures.
fmax = None
evalfmax = runlen_xlimits_max  # is manipulated/stored in this module

# TODO: the target function values and the styles of the line only make sense
# together. Therefore we should either:
# 1. keep the targets as input argument and make rldStyles depend on them or
# 2. remove the targets as input argument and put them here.
rldStyles = ({'color': 'k', 'ls': '-'},
             {'color': 'c'},
             {'color': 'm', 'ls': '-'},
             {'color': 'r', 'linewidth': 3.},
             {'color': 'k'},
             {'color': 'c'},
             {'color': 'm'},
             {'color': 'r'},
             {'color': 'k'},
             {'color': 'c'},
             {'color': 'm'},
             {'color': 'r', 'linewidth': 3.})
rldUnsuccStyles = (
                   {'color': 'c', 'ls': '-'},
                   {'color': 'm', 'ls': '-'},
                   {'color': 'k', 'ls': '-'},
                   {'color': 'c'},
                   {'color': 'm', 'ls': '-'},
                   {'color': 'k', 'ls': '-'},
                   {'color': 'c'},
                   {'color': 'm', 'ls': '-'},
                   {'color': 'k'},
                   {'color': 'c', 'ls': '-'},
                   {'color': 'm'},
                   {'color': 'k'},
                   )  # should not be too short

caption_part_one = r"""%
     Empirical cumulative distribution functions (ECDF), plotting the fraction of 
     trials with an outcome not larger than the respective value on the $x$-axis. 
     #1"""
caption_left_fixed_targets = r"""%
     Left subplots: ECDF of the number of function evaluations (FEvals) divided by search space dimension $D$, 
     to fall below $\fopt+\Df$ with $\Df=10^{k}$, where $k$ is the first value in the legend. 
     The thick red line represents the most difficult target value $\fopt+10^{-8}$. """
caption_left_rlbased_targets = r"""%
     Left subplots: ECDF of number of function evaluations (FEvals) divided by search space dimension $D$, 
     to fall below $\fopt+\Df$ where \Df\ is the 
     target just not reached by the GECCO-BBOB-2009 best algorithm within a budget of
     % largest $\Df$-value $\ge10^{-8}$ for which the best \ERT\ seen in the GECCO-BBOB-2009 was yet above 
     $k\times\DIM$ evaluations, where $k$ is the first value in the legend. """
caption_wrap_up = r"""%
     Legends indicate for each target the number of functions that were solved in at
     least one trial within the displayed budget.
     """
caption_right = r"""%
     Right subplots: ECDF of the 
     best achieved $\Df$ 
     for running times of TO_BE_REPLACED
     function evaluations 
     (from right to left cycling cyan-magenta-black\dots) and final $\Df$-value (red), 
     where \Df\ and \textsf{Df} denote the difference to the optimal function value. 
     Light brown lines in the background show ECDFs for the most difficult target of all 
     algorithms benchmarked during BBOB-2009.
     """
caption_single_fixed = caption_part_one + caption_left_fixed_targets + caption_wrap_up + caption_right 
caption_single_rlbased = caption_part_one + caption_left_rlbased_targets + caption_wrap_up + caption_right


previous_data_filename = 'pprldistr2009_1e-8.pickle.gz'
previous_RLBdata_filename="pprldistr2009_hardestRLB.pickle.gz"
previous_data_filename = os.path.join(os.path.split(__file__)[0], previous_data_filename)
previous_RLBdata_filename =os.path.join(os.path.split(__file__)[0], previous_RLBdata_filename)
previous_data_dict = None
previous_RLBdata_dict = None
def load_previous_data(filename=previous_data_filename, force=False):
    if previous_data_dict and not force:
        return previous_data_dict
    try:
        # cocofy(previous_data_filename)
        f = gzip.open(previous_data_filename, 'r')
        return pickle.load(f)
    except IOError, (errno, strerror):
        print "I/O error(%s): %s" % (errno, strerror)
        previous_algorithm_data_found = False
        print 'Could not find file: ', previous_data_filename
    else:
        f.close()
    return None

def load_previous_RLBdata(filename=previous_RLBdata_filename):
    if previous_RLBdata_dict:
        return previous_RLBdata_dict
    try:
        f = gzip.open(previous_RLBdata_filename, 'r')
        return pickle.load(f)
    except IOError, (errno, strerror):
        print "I/O error(%s): %s" % (errno, strerror)
        print 'Could not find file: ', previous_RLBdata_filename
    else:
        f.close()
    return None
    
    
def caption_single(max_evals_div_dim):
    caption = caption_single_rlbased if genericsettings.runlength_based_targets else caption_single_fixed 
    return caption.replace(r'TO_BE_REPLACED', '$' + 'D, '.join([str(i) for i in single_runlength_factors[:6]]) + 'D,\dots$') 

def beautifyECDF():
    """Generic formatting of ECDF figures."""
    plt.ylim(-0.0, 1.01)  # was plt.ylim(-0.01, 1.01)
    plt.yticks(np.arange(0., 1.001, 0.2))  # , ('0.0', '', '0.5', '', '1.0'))
    plt.grid(True)
    xmin, xmax = plt.xlim()
    # plt.xlim(xmin=xmin*0.90)  # why this?
    c = plt.gca().get_children()
    for i in c:  # TODO: we only want to extend ECDF lines...
        try:
            if i.get_drawstyle() == 'steps' and not i.get_linestyle() in ('', 'None'):
                xdata = i.get_xdata()
                ydata = i.get_ydata()
                if len(xdata) > 0:
                    # if xmin < min(xdata):
                    #    xdata = np.hstack((xmin, xdata))
                    #    ydata = np.hstack((ydata[0], ydata))
                    if xmax > max(xdata):
                        xdata = np.hstack((xdata, xmax))
                        ydata = np.hstack((ydata, ydata[-1]))
                    plt.setp(i, 'xdata', xdata, 'ydata', ydata)
            elif (i.get_drawstyle() == 'steps' and i.get_marker() != '' and
                  i.get_linestyle() in ('', 'None')):
                xdata = i.get_xdata()
                ydata = i.get_ydata()
                if len(xdata) > 0:
                    # if xmin < min(xdata):
                    #    minidx = np.ceil(np.log10(xmin) * nbperdecade)
                    #    maxidx = np.floor(np.log10(xdata[0]) * nbperdecade)
                    #    x = 10. ** (np.arange(minidx, maxidx + 1) / nbperdecade)
                    #    xdata = np.hstack((x, xdata))
                    #    ydata = np.hstack(([ydata[0]] * len(x), ydata))
                    if xmax > max(xdata):
                        minidx = np.ceil(np.log10(xdata[-1]) * nbperdecade)
                        maxidx = np.floor(np.log10(xmax) * nbperdecade)
                        x = 10. ** (np.arange(minidx, maxidx + 1) / nbperdecade)
                        xdata = np.hstack((xdata, x))
                        ydata = np.hstack((ydata, [ydata[-1]] * len(x)))
                    plt.setp(i, 'xdata', xdata, 'ydata', ydata)
        except (AttributeError, IndexError):
            pass

def beautifyRLD(xlimit_max=None):
    """Format and save the figure of the run length distribution.
    
    After calling this function, changing the boundaries of the figure
    will not update the ticks and tick labels.
    
    """
    a = plt.gca()
    a.set_xscale('log')
    a.set_xlabel('log10 of FEvals / DIM')
    a.set_ylabel('proportion of trials')
    logxticks()
    if xlimit_max:
        plt.xlim(xmax=xlimit_max ** 1.0)  # was 1.05
    plt.xlim(xmin=runlen_xlimits_min)
    plt.text(plt.xlim()[0], plt.ylim()[0], single_target_values.short_info, fontsize=14)
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
    plt.xlim(1e-8, fmax)  # 1e-8 was 1.
    # axisHandle.invert_xaxis()
    a.set_xlabel('log10 of Df')  # / Dftarget
    if ylabel:
        a.set_ylabel('proportion of trials')
    logxticks(limits=plt.xlim())
    beautifyECDF()
    if not ylabel:
        a.set_yticklabels(())

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
        x = sorted(x)  # do not sort in place
        x = np.hstack((x, x[-1]))
        y = np.hstack((np.arange(0., nx) / n, float(nx) / n))
        res = plotUnifLogXMarkers(x, y, nbperdecade=nbperdecade,
                                 drawstyle='steps', **plotArgs)
    return res

def _plotERTDistr(dsList, target, **plotArgs):
    """This method is obsolete, should be removed? The replacement for simulated runlengths is in pprldmany? 
    Creates simulated run time distributions (it is not an ERT distribution) from a DataSetList.

    :keyword DataSet dsList: Input data sets
    :keyword dict target: target precision
    :keyword plotArgs: keyword arguments to pass to plot command

    :return: resulting plot.

    Details: calls ``plotECDF``. 
    
    """
    x = []
    nn = 0
    samplesize = genericsettings.simulated_runlength_bootstrap_sample_size  # samplesize should be at least 1000
    percentiles = 0.5  # could be anything...

    for i in dsList:
        # funcs.add(i.funcId)
        for j in i.evals:
            if j[0] <= target[i.funcId]:
                runlengthsucc = j[1:][np.isfinite(j[1:])]
                runlengthunsucc = i.maxevals[np.isnan(j[1:])]
                tmp = toolsstats.drawSP(runlengthsucc, runlengthunsucc,
                                       percentiles=percentiles,
                                       samplesize=samplesize)
                x.extend(tmp[1])
                break
        nn += samplesize
    res = plotECDF(x, nn, **plotArgs)

    return res

def _plotRLDistr_old(dsList, target, **plotArgs):
    """Creates run length distributions from a sequence dataSetList.

    Labels of the line (for the legend) will be set automatically with
    the following format: %+d: %d/%d % (log10()
    

    :param DataSetList dsList: Input data sets
    :param dict or float target: target precision
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
            target = target[i.funcId]  # TODO: this can only work for a single function, generally looks like a bug
            if not genericsettings.test:
                print 'target:', target
                print 'function:', i.funcId
                raise Exception('please check this, it looks like a bug')
        except TypeError:
            target = target
        tmp = i.detEvals((target,))[0] / i.dim
        tmp = tmp[np.isnan(tmp) == False]  # keep only success
        if len(tmp) > 0:
            fsolved.add(i.funcId)
        x.extend(tmp)
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

def plotRLDistr(dsList, target, label='', max_fun_evals=np.inf, **plotArgs):
    """Creates run length distributions from a sequence dataSetList.

    Labels of the line (for the legend) will be appended with the number
    of functions at least solved once. 
    
    :param DataSetList dsList: Input data sets
    :param target: a method that delivers single target values like ``target((fun, dim))``
    :param str label: target value label to be displayed in the legend
    :param plotArgs: additional arguments passed to the plot command

    :returns: handles of the resulting plot.

    Example::
    
        plotRLDistr(dsl, lambda f: 1e-6)

    Details: ``target`` is a function taking a (function_number, dimension) pair 
    as input and returning a ``float``. It can be defined as 
    ``lambda fun_dim: targets(fun_dim)[j]`` returning the j-th element of 
    ``targets(fun_dim)``, where ``targets`` is an instance of 
    ``class pproc.TargetValues`` (see the ``pproc.TargetValues.__call__`` method).  
    
    """
    x = []
    nn = 0
    fsolved = set()
    funcs = set()
    for ds in dsList:  # ds is a DataSet
        funcs.add(ds.funcId)
        tmp = ds.detEvals((target((ds.funcId, ds.dim)),))[0] / ds.dim
        tmp = tmp[np.isnan(tmp) == False]  # keep only success
        if len(tmp) > 0 and sum(tmp <= max_fun_evals):
            fsolved.add(ds.funcId)
        x.extend(tmp)
        nn += ds.nbRuns()
    kwargs = plotArgs.copy()
    label += ': %d/%d' % (len(fsolved), len(funcs))
    kwargs['label'] = kwargs.setdefault('label', label)
    res = plotECDF(x, nn, **kwargs)
    return res

def plotFVDistr(dsList, budget, min_f=1e-8, **plotArgs):
    """Creates ECDF of final function values plot from a DataSetList.
    
    :param dsList: data sets
    :param min_f: used for the left limit of the plot
    :param float budget: maximum evaluations / dimension that "count"
    :param plotArgs: additional arguments passed to plot

    :returns: handle

    """
    x = []
    nn = 0
    for ds in dsList:
        for i, fvals in enumerate(ds.funvals):
            if fvals[0] > budget * ds.dim:
                assert i > 0, 'first entry ' + str(fvals[0]) + 'was smaller than maximal budget ' + str(budget * ds.dim)
                fvals = ds.funvals[i - 1]
                break
        # vals = fvals[1:].copy() / target[i.funcId]
        vals = fvals[1:].copy()
        # replace negative values to prevent problem with log of vals
        vals[vals <= 0] = min(np.append(vals[vals > 0], [min_f]))  # works also when vals[vals > 0] is empty
        if genericsettings.runlength_based_targets:
            NotImplementedError('related function vals with respective budget (e.g. ERT(val)) see pplogloss.generateData()')
        x.extend(vals)
        nn += ds.nbRuns()
    res = plotECDF(x, nn, **plotArgs)
    return res

def comp(dsList0, dsList1, targets, isStoringXMax=False,
         outputdir='', info='default', verbose=True):
    """Generate figures of ECDF that compare 2 algorithms.

    :param DataSetList dsList0: list of DataSet instances for ALG0
    :param DataSetList dsList1: list of DataSet instances for ALG1
    :param seq targets: target function values to be displayed
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
    # plt.rc("axes", labelsize=20, titlesize=24)
    # plt.rc("xtick", labelsize=20)
    # plt.rc("ytick", labelsize=20)
    # plt.rc("font", size=20)
    # plt.rc("legend", fontsize=20)

    targets = pproc.TargetValues.cast(targets)
    dictdim0 = dsList0.dictByDim()
    dictdim1 = dsList1.dictByDim()
    for d in set(dictdim0.keys()) & set(dictdim1.keys()):
        maxEvalsFactor = max(max(i.mMaxEvals() / d for i in dictdim0[d]),
                             max(i.mMaxEvals() / d for i in dictdim1[d]))    
        if isStoringXMax:
            global evalfmax
        else:
            evalfmax = None
        if not evalfmax:
            evalfmax = maxEvalsFactor ** 1.05
        if runlen_xlimits_max is not None:
            evalfmax = runlen_xlimits_max

        filename = os.path.join(outputdir, 'pprldistr_%02dD_%s' % (d, info))
        fig = plt.figure()
        for j in range(len(targets)):
            tmp = plotRLDistr(dictdim0[d], lambda fun_dim: targets(fun_dim)[j], targets.loglabel(j),
                              marker=genericsettings.line_styles[1]['marker'],
                              **rldStyles[j % len(rldStyles)])
            plt.setp(tmp[-1], label=None)  # Remove automatic legend
            # Mods are added after to prevent them from appearing in the legend
            plt.setp(tmp, markersize=20.,
                     markeredgewidth=plt.getp(tmp[-1], 'linewidth'),
                     markeredgecolor=plt.getp(tmp[-1], 'color'),
                     markerfacecolor='none')
    
            tmp = plotRLDistr(dictdim1[d], lambda fun_dim: targets(fun_dim)[j], targets.loglabel(j),
                              marker=genericsettings.line_styles[0]['marker'],
                              **rldStyles[j % len(rldStyles)])
            # modify the automatic legend: remover marker and change text
            plt.setp(tmp[-1], marker='', label=targets.loglabel(j))
            # Mods are added after to prevent them from appearing in the legend
            plt.setp(tmp, markersize=15.,
                     markeredgewidth=plt.getp(tmp[-1], 'linewidth'),
                     markeredgecolor=plt.getp(tmp[-1], 'color'),
                     markerfacecolor='none')
    
        funcs = set(i.funcId for i in dictdim0[d]) | set(i.funcId for i in dictdim1[d])
        text = 'f%s' % (consecutiveNumbers(sorted(funcs)))
        plot_previous_algorithms(d, funcs)
        # plt.axvline(max(i.mMaxEvals()/i.dim for i in dictdim0[d]), ls='--', color='k')
        # plt.axvline(max(i.mMaxEvals()/i.dim for i in dictdim1[d]), color='k')
        plt.axvline(max(i.mMaxEvals() / i.dim for i in dictdim0[d]),
                    marker='+', markersize=20., color='k',
                    markeredgewidth=plt.getp(tmp[-1], 'linewidth',))
        plt.axvline(max(i.mMaxEvals() / i.dim for i in dictdim1[d]),
                    marker='o', markersize=15., color='k', markerfacecolor='None',
                    markeredgewidth=plt.getp(tmp[-1], 'linewidth'))
        plt.legend(loc='best')
        plt.text(0.5, 0.98, text, horizontalalignment="center",
                 verticalalignment="top", transform=plt.gca().transAxes)  # bbox=dict(ec='k', fill=False), 
        beautifyRLD(evalfmax)
        saveFigure(filename, verbose=verbose)
        plt.close(fig)

def beautify():
    """Format the figure of the run length distribution.
    
    Used in conjunction with plot method (obsolete/outdated, see functions ``beautifyFVD`` and ``beautifyRLD``).

    """
    # raise NotImplementedError('this implementation is obsolete')
    plt.subplot(121)
    axisHandle = plt.gca()
    axisHandle.set_xscale('log')
    axisHandle.set_xlabel('log10 of FEvals / DIM')
    axisHandle.set_ylabel('proportion of trials')
    # Grid options
    logxticks()
    beautifyECDF()

    plt.subplot(122)
    axisHandle = plt.gca()
    axisHandle.set_xscale('log')
    xmin, fmax = plt.xlim()
    plt.xlim(1., fmax)
    axisHandle.set_xlabel('log10 of Df / Dftarget')
    beautifyECDF()
    logxticks()
    axisHandle.set_yticklabels(())
    plt.gcf().set_size_inches(16.35, 6.175)
#     try:
#         set_trace()
#         plt.setp(plt.gcf(), 'figwidth', 16.35)
#     except AttributeError: # version error?
#         set_trace()
#         plt.setp(plt.gcf(), 'figsize', (16.35, 6.))

def plot(dsList, targets=single_target_values, **plotArgs):
    """Plot ECDF of evaluations and final function values
    in a single figure for demonstration purposes."""
    # targets = targets()  # TODO: this needs to be rectified
    # targets = targets.target_values
    dsList = pproc.DataSetList(dsList)
    assert len(dsList.dictByDim()) == 1, ('Cannot display different '
                                          'dimensionalities together')
    res = []

    plt.subplot(121)
    maxEvalsFactor = max(i.mMaxEvals() / i.dim for i in dsList)
    evalfmax = maxEvalsFactor
    for j in range(len(targets)):
        tmpplotArgs = dict(plotArgs, **rldStyles[j % len(rldStyles)])
        tmp = plotRLDistr(dsList, lambda fun_dim: targets(fun_dim)[j], **tmpplotArgs)
        res.extend(tmp)
    res.append(plt.axvline(x=maxEvalsFactor, color='k', **plotArgs))
    funcs = list(i.funcId for i in dsList)
    text = 'f%s' % (consecutiveNumbers(sorted(funcs)))
    res.append(plt.text(0.5, 0.98, text, horizontalalignment="center",
                        verticalalignment="top", transform=plt.gca().transAxes))

    plt.subplot(122)
    for j in [range(len(targets))[-1]]:
        tmpplotArgs = dict(plotArgs, **rldStyles[j % len(rldStyles)])
        tmp = plotFVDistr(dsList, evalfmax, lambda fun_dim: targets(fun_dim)[j], **tmpplotArgs)
        res.extend(tmp)
    tmp = np.floor(np.log10(evalfmax))
    # coloring right to left:
    maxEvalsF = np.power(10, np.arange(0, tmp))
    for j in range(len(maxEvalsF)):
        tmpplotArgs = dict(plotArgs, **rldUnsuccStyles[j % len(rldUnsuccStyles)])
        tmp = plotFVDistr(dsList, maxEvalsF[j], lambda fun_dim: targets(fun_dim)[-1], **tmpplotArgs)
        res.extend(tmp)
    res.append(plt.text(0.98, 0.02, text, horizontalalignment="right",
                        transform=plt.gca().transAxes))
    return res

def plot_previous_algorithms(dim, funcs):
    """Display BBOB 2009 data, by default from ``pprldistr.previous_data_filename = 'pprldistr2009_1e-8.pickle.gz'``"""

    global previous_data_dict
    if previous_data_dict is None:
        previous_data_dict = load_previous_data()  # this takes about 6 seconds
    if previous_data_dict is not None:
        for alg in previous_data_dict:
            x = []
            nn = 0
            try:
                tmp = previous_data_dict[alg]
                for f in funcs:
                    tmp[f][dim]  # simply test that they exists
            except KeyError:
                continue

            for f in funcs:
                tmp2 = tmp[f][dim][0][1:]
                # [0], because the maximum #evals is also recorded
                # [1:] because the target function value is recorded
                x.append(tmp2[np.isnan(tmp2) == False])
                nn += len(tmp2)

            if x:
                x = np.hstack(x)
                plotECDF(x[np.isfinite(x)] / float(dim), nn,
                         color=refcolor, ls='-', zorder= -1)



def plotRLB_previous_algorithms(dim, funcs):
    """Display BBOB 2009 data, by default from ``pprldistr.previous_data_filename = 'pprldistr2009_1e-8.pickle.gz'``"""

    global previous_RLBdata_dict
    if previous_RLBdata_dict is None:
        previous_RLBdata_dict = load_previous_RLBdata()
    if previous_RLBdata_dict is not None:
        for alg in previous_RLBdata_dict:
            x = []
            nn = 0
            try:
                tmp = previous_RLBdata_dict[alg]
                for f in funcs:
                    tmp[f][dim]  # simply test that they exists
            except KeyError:
                continue

            for f in funcs:
                tmp2 = np.array(tmp[f][dim][0][1:][0])
                # [0], because the maximum #evals is also recorded
                # [1:] because the target function value is recorded
                x.append(tmp2[np.isnan(tmp2) == False])
                nn += len(tmp2)
                                
            if x:
                x = np.hstack(x)
                plotECDF(x[np.isfinite(x)] / float(dim), nn,
                         color=refcolor, ls='-', zorder= -1)



def main(dsList, isStoringXMax=False, outputdir='',
         info='default', verbose=True):
    """Generate figures of empirical cumulative distribution functions.
    
    This method has a feature which allows to keep the same boundaries
    for the x-axis, if ``isStoringXMax==True``. This makes sense when 
    dealing with different functions or subsets of functions for one 
    given dimension.
    
    CAVE: this is bug-prone, as some data depend on the maximum 
    evaluations and the appearence therefore depends on the 
    calling order. 

    :param DataSetList dsList: list of DataSet instances to process.
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
    # plt.rc("axes", labelsize=20, titlesize=24)
    # plt.rc("xtick", labelsize=20)
    # plt.rc("ytick", labelsize=20)
    # plt.rc("font", size=20)
    # plt.rc("legend", fontsize=20)
    targets = single_target_values  # convenience abbreviation
    for d, dictdim in dsList.dictByDim().iteritems():
        maxEvalsFactor = max(i.mMaxEvals() / d for i in dictdim)
        if isStoringXMax:
            global evalfmax
        else:
            evalfmax = None
        if not evalfmax:
            evalfmax = maxEvalsFactor
        if runlen_xlimits_max is not None:
            evalfmax = runlen_xlimits_max

        # first figure: Run Length Distribution
        filename = os.path.join(outputdir, 'pprldistr_%02dD_%s' % (d, info))
        fig = plt.figure()
        for j in range(len(targets)):
            plotRLDistr(dictdim,
                        lambda fun_dim: targets(fun_dim)[j],
                        targets.label(j) if isinstance(targets, pproc.RunlengthBasedTargetValues) else targets.loglabel(j),
                        evalfmax,  # can be larger maxEvalsFactor with no effect
                        ** rldStyles[j % len(rldStyles)])
    
        funcs = list(i.funcId for i in dictdim)
        text = 'f%s' % (consecutiveNumbers(sorted(funcs)))
        text += ',%d-D' % d
        if(1):
     #   try:
            
            if not isinstance(targets, pproc.RunlengthBasedTargetValues):
            #if targets.target_values[-1] == 1e-8:  # this is a hack
                plot_previous_algorithms(d, funcs)

            else:
                plotRLB_previous_algorithms(d, funcs)
                
    #    except:
     #       pass
            
        plt.axvline(x=maxEvalsFactor, color='k')  # vertical line at maxevals
        plt.legend(loc='best')
        plt.text(0.5, 0.98, text, horizontalalignment="center",
                 verticalalignment="top",
                 transform=plt.gca().transAxes 
                 # bbox=dict(ec='k', fill=False)
                 ) 
        beautifyRLD(evalfmax)
        saveFigure(filename, verbose=verbose)
        plt.close(fig)
        
        # second figure: Function Value Distribution
        filename = os.path.join(outputdir, 'ppfvdistr_%02dD_%s' % (d, info))
        fig = plt.figure()
        plotFVDistr(dictdim, np.inf, 1e-8, **rldStyles[-1])
        # coloring right to left
        for j, max_eval_factor in enumerate(single_runlength_factors):
            if max_eval_factor > maxEvalsFactor:
                break
            plotFVDistr(dictdim, max_eval_factor, 1e-8,
                        **rldUnsuccStyles[j % len(rldUnsuccStyles)])
                
        plt.text(0.98, 0.02, text, horizontalalignment="right",
                 transform=plt.gca().transAxes)  # bbox=dict(ec='k', fill=False), 
        beautifyFVD(isStoringXMax=isStoringXMax, ylabel=False)
        saveFigure(filename, verbose=verbose)
        plt.close(fig)
        # plt.rcdefaults()

