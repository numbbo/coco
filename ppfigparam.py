#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate ERT vs param. figures.

The figures will show the performance in terms of ERT on a log scale
w.r.t. parameter. On the y-axis, data is represented as
a number of function evaluations. Crosses (+) give the median number of
function evaluations for the smallest reached target function value
(also divided by dimension). Crosses (×) give the average number of
overall conducted function evaluations in case the smallest target
function value (1e-8) was not reached.

"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace
from bbob_pproc import bootstrap, bestalg
from bbob_pproc.ppfig import saveFigure, groupByRange

__all__ = ['beautify', 'plot', 'main']

avgstyle = dict(color='r', marker='x', markersize=20)
medmarker = dict(linestyle='', marker='+', markersize=30, markeredgewidth=5,
                 zorder=-1)

colors = ('k', 'b', 'c', 'g', 'y', 'm', 'r', 'k', 'k', 'c', 'r', 'm')  # sort of rainbow style
styles = [{'color': 'k', 'marker': 'o', 'markeredgecolor': 'k'},
          {'color': 'b'},
          {'color': 'c', 'marker': 'v', 'markeredgecolor': 'c'},
          {'color': 'g'},
          {'color': 'y', 'marker': '^', 'markeredgecolor': 'y'},
          {'color': 'm'},
          {'color': 'r', 'marker': 's', 'markeredgecolor': 'r'}] # sort of rainbow style

refcolor = 'wheat'
# should correspond with the colors in pprldistr.

dimsBBOB = (2, 3, 5, 10, 20, 40)

#Get benchmark short infos.
funInfos = {}
isBenchmarkinfosFound = True
infofile = os.path.join(os.path.split(__file__)[0], 'benchmarkshortinfos.txt')
try:
    f = open(infofile,'r')
    for line in f:
        if len(line) == 0 or line.startswith('%') or line.isspace() :
            continue
        funcId, funcInfo = line[0:-1].split(None,1)
        funInfos[int(funcId)] = funcId + ' ' + funcInfo
    f.close()
except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    isBenchmarkinfosFound = False
    print 'Could not find file', infofile, \
          'Function names will not be displayed.'

def beautify():
    """Customize figure presentation."""

    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_xscale("log")
    axisHandle.set_yscale("log")

    # Grid options
    axisHandle.grid(True)

    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()

    # quadratic and cubic "grid"
    #plt.plot((2,200), (1, 1e2), 'k:')
    #plt.plot((2,200), (1, 1e4), 'k:')
    #plt.plot((2,200), (1e3, 1e5), 'k:')  
    #plt.plot((2,200), (1e3, 1e7), 'k:')
    #plt.plot((2,200), (1e6, 1e8), 'k:')  
    #plt.plot((2,200), (1e6, 1e10), 'k:')

    # axes limits
    plt.ylim(ymin=10**-0.2, ymax=ymax) # Set back the previous maximum.

    # ticks on axes
    # axisHandle.invert_xaxis()
    # plt.xlim(1.8, 45)                # TODO should become input arg?
    # dimticklist = (2, 3, 4, 5, 10, 20, 40)  # TODO: should become input arg at some point? 
    # dimannlist = (2, 3, '', 5, 10, 20, 40)  # TODO: should become input arg at some point? 
    # TODO: All these should depend on one given input (xlim, ylim)
    # axisHandle.set_xticks(dimticklist)
    # axisHandle.set_xticklabels([str(n) for n in dimannlist])

    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(np.log10(i)))
    axisHandle.set_yticklabels(tmp2)
    plt.ylabel('Run Lengths')

def plot(dsList, param='dim', targets=(10., 1., 1e-1, 1e-2, 1e-3, 1e-5, 1e-8)):
    """Generate plot of ERT vs param."""

    dictparam = dsList.dictByParam(param)
    params = sorted(dictparam) # sorted because we draw lines

    # generate plot from dsList
    res = []
    # collect data
    rawdata = {}
    for p in params:
        assert len(dictparam[p]) == 1
        rawdata[p] = dictparam[p][0].detEvals(targets)
        # expect dictparam[p] to have only one element

    # plot lines for ERT
    xpltdata = params
    for i, t in enumerate(targets):
        ypltdata = []
        for p in params:
            data = rawdata[p][i]
            unsucc = np.isnan(data)
            assert len(dictparam[p]) == 1
            data[unsucc] = dictparam[p][0].maxevals
            # compute ERT
            ert, srate, succ = bootstrap.sp(data, issuccessful=(unsucc == False))
            ypltdata.append(ert)
        res.extend(plt.plot(xpltdata, ypltdata, markersize=20,
                   zorder=len(targets) - i, **styles[i]))
        # for the legend
        plt.plot([], [], markersize=10,
                 label=' %+d' % (np.log10(targets[i])),
                 **styles[i])

    # plot median of successful runs for hardest target with a success
    for p in params:
        for i, t in enumerate(reversed(targets)): # targets has to be from hardest to easiest
            data = rawdata[p][i]
            data = data[np.isnan(data) == False]
            if len(data) > 0:
                median = bootstrap.prctile(data, 50.)[0]
                res.extend(plt.plot(p, median, styles[i]['color'], **medmarker))
                break

    # plot average number of function evaluations for the hardest target
    xpltdata = []
    ypltdata = []
    for p in params:
        data = rawdata[p][0] # first target
        xpltdata.append(p)
        if (np.isnan(data) == False).all():
            tmpdata = data.copy()
            assert len(dictparam[p]) == 1
            tmpdata[np.isnan(data)] = dictparam[p][0].maxevals[np.isnan(data)]
            tmp = np.mean(tmpdata)
        else:
            tmp = np.nan # Check what happens when plotting NaN
        ypltdata.append(tmp)
    res.extend(plt.plot(xpltdata, ypltdata, **avgstyle))

    # display numbers of successes for hardest target where there is still one success
    for p in params:
        for i, t in enumerate(targets): # targets has to be from hardest to easiest
            data = rawdata[p][i]
            unsucc = np.isnan(data)
            assert len(dictparam[p]) == 1
            data[unsucc] = dictparam[p][0].maxevals
            # compute ERT
            ert, srate, succ = bootstrap.sp(data, issuccessful=(unsucc == False))
            if srate == 1.:
                break
            elif succ > 0:
                res.append(plt.text(p, ert * 1.85, "%d" % succ, axes=plt.gca(),
                                    horizontalalignment="center",
                                    verticalalignment="bottom"))
                break
    return res

def main(dsList, _targets=(10., 1., 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
         param=('dim', 'Dimension'), is_normalized=True, outputdir='.',
         verbose=True):
    """Generates figure of ERT vs. param.

    This script will generate as many figures as there are functions.
    For a given function and a given parameter value there should be
    only **one** data set.
    Crosses (+) give the median number of function evaluations of
    successful trials for the smallest reached target function value.
    Crosses (x) give the average number of overall conducted function
    evaluations in case the smallest target function value (1e-8) was
    not reached.

    :keyword DataSetList dsList: data sets
    :keyword seq _targets: target precisions
    :keyword tuple param: parameter on x-axis. The first element has to
                          be a string corresponding to the name of an
                          attribute common to elements of dsList. The
                          second element has to be a string which will
                          be used as label for the figures. The values
                          of attribute param have to be sortable.
    :keyword bool is_normalized: if True the y values are normalized by
                                 x values
    :keyword string outputdir: name of output directory for the image
                               files
    :keyword bool verbose: controls verbosity
    
    """

    # TODO check input parameter param
    for func, dictfunc in dsList.dictByFunc().iteritems():
        filename = os.path.join(outputdir,'ppfigparam_%s_f%03d' % (param[0], func))

        try:
            targets = list(j[func] for j in _targets)
        except TypeError:
            targets = _targets
        targets = sorted(targets) # from hard to easy

        handles = plot(dictfunc, param[0], targets)

        # # display best 2009
        # if not bestalg.bestalgentries2009:
        #     bestalg.loadBBOB2009()

        # bestalgdata = []
        # for d in dimsBBOB:
        #     entry = bestalg.bestalgentries2009[(d, func)]
        #     tmp = entry.detERT([1e-8])[0]
        #     if not np.isinf(tmp):
        #         bestalgdata.append(tmp/d)
        #     else:
        #         bestalgdata.append(None)

        # plt.plot(dimsBBOB, bestalgdata, color=refcolor, linewidth=10, zorder=-2)
        # plt.plot(dimsBBOB, bestalgdata, ls='', marker='d', markersize=25,
        #          color=refcolor, markeredgecolor=refcolor, zorder=-2)

        a = plt.gca()
        if is_normalized:
            for i in handles:
                try:
                    plt.setp(i, 'ydata', plt.getp(i, 'ydata') / plt.getp(i, 'xdata'))
                except TypeError:
                    pass
            a.relim()
            a.autoscale_view()

        beautify()
        plt.xlabel(param[1])
        if is_normalized:
            plt.setp(plt.gca(), 'ylabel', plt.getp(a, 'ylabel') + ' / ' + param[1])

        if func in (1, 24, 101, 130):
            plt.legend(loc="best")
        if isBenchmarkinfosFound:
            a.set_title(funInfos[func])

        saveFigure(filename, verbose=verbose)
        plt.close()
