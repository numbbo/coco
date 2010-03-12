#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates ERTs and convergence figures for the comparison of 2 algorithms."""

from __future__ import absolute_import

import os
import sys
import warnings
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy
from pdb import set_trace
from bbob_pproc import bootstrap, readalign
from bbob_pproc.bootstrap import ranksums
from bbob_pproc.ppfig import saveFigure
#try:
    #supersede this module own ranksums method
    #from scipy.stats import ranksums as ranksums
#except ImportError:
    #from bbob_pproc.bootstrap import ranksums
    #pass

# colors = ('k', 'b', 'c', 'g', 'y', 'm', 'r', 'k', 'k', 'c', 'r', 'm')
# markers = ('o', 'v', 's', '+', 'x', 'D')
colors = ('c', 'g', 'b', 'k', 'r', 'm', 'k', 'y', 'k', 'c', 'r', 'm')
markers = ('+', 'v', '*', 'o', 's', 'D', 'x')
linewidth = 3
offset = 0.005

figformat = ('eps', 'pdf') # Controls the output when using the main method

#Get benchmark short infos.
funInfos = {}
isBenchmarkinfosFound = True
infofile = os.path.join(os.path.split(__file__)[0], '..',
                        'benchmarkshortinfos.txt')

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
          'Titles in scaling figures will not be displayed.'

def generateData(entry0, entry1):

    def alignData(i0, i1):
        """Returns two arrays of fevals aligned on function evaluations.
        """
    
        res = readalign.alignArrayData(readalign.HArrayMultiReader([i0.evals,
                                                                    i1.evals]))
        idx = 1 + i0.nbRuns()
        data0 = res[:, numpy.r_[0, 1:idx]]
        data1 = res[:, numpy.r_[0, idx:idx+i1.nbRuns()]]
        return data0, data1
    
    def computeERT(hdata, maxevals):
        res = []
        for i in hdata:
            data = i.copy()
            data = data[1:]
            succ = (numpy.isnan(data)==False)
            if any(numpy.isnan(data)):
                data[numpy.isnan(data)] = maxevals[numpy.isnan(data)]
            tmp = [i[0]]
            tmp.extend(bootstrap.sp(data, issuccessful=succ))
            res.append(tmp)
        return numpy.vstack(res)

    tmpdata0, tmpdata1 = alignData(entry0, entry1)
    data0 = computeERT(tmpdata0, entry0.maxevals)
    data1 = computeERT(tmpdata1, entry1.maxevals)

    return data0, data1

def plotERTRatio(data, plotargs={}):
    """Returns handles of line2D plots using input data.
    Input data must be a sequence which first two arguments correspond to
    data of algorithms 0 and 1. The elements of the sequence must be arrays
    which first column is the target function values and the second the 
    ERT.
    """
    
    res = []
    idx = numpy.isfinite(data[0][:, 1]) * numpy.isfinite(data[1][:, 1])
    ydata = data[1][idx, 1]/data[0][idx, 1]
    h = plt.plot(data[0][idx, 0] , ydata, ls='--', **plotargs)
    res.extend(h)
    
    return h

def beautify(xmin=None):
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    ymin, ymax = plt.ylim()
    ybnd = max(1./ymin, ymax)
    plt.ylim(1./ybnd, ybnd)

    # We are setting xmin
    if xmin:
        plt.xlim(xmin=xmin)
    plt.xlim(xmax=1000)
    ax.invert_xaxis()

    # Annotate figure
    ax.set_xlabel('log10(Delta ftarget)')
    ax.set_ylabel('log10(ERT1/ERT0)')

    ax.grid('True')

    #Tick label handling
    xticks = ax.get_xticks()
    tmp = []
    for i in xticks:
        tmp.append('%d' % round(numpy.log10(i)))
    ax.set_xticklabels(tmp)

    yticks = ax.get_yticks()
    tmp = []
    for i in yticks:
        tmp.append('%d' % round(numpy.log10(i)))
    ax.set_yticklabels(tmp)

    # Reverse yticks below 1
    tmp = ax.get_yticks(minor=True)
    tmp[tmp<1] = sorted(1/(tmp[tmp<1]*numpy.power(10, -2*numpy.floor(numpy.log10(tmp[tmp<1]))-1)))
    tmp = tmp[tmp<plt.ylim()[1]]
    tmp = tmp[tmp>plt.ylim()[0]]
    ax.set_yticks(tmp, minor=True)

def annotate(data, dim, minfvalue=1e-8):
    """Display some annotations associated to the graphs generated."""
    
    isEarlyStop = False
    ha = 'left'
    va = 'center'
    if minfvalue < data[0][-1, 0]:
        isEarlyStop = True
        ha = 'center'
        va = 'top'

    if not minfvalue or minfvalue < data[0][-1, 0]:
        minfvalue = data[0][-1, 0]

    # Locate the data corresponding to minfvalue
    line = []
    for d in data:
        line.append(d[d[:, 0] >= minfvalue, :][-1])
    
    # What's the situation?
    txt = '%d-D'
    if line[0][2] == 0 or line[1][2] == 0:
        txt = '%d-D' % dim
    elif line[0][3] > 9 and line[1][3] > 9.:
        txt = '%d-D' % dim    
    else:
        tmp = str(int(line[0][3]))
        tmp2 = str(int(line[1][3]))
        if line[0][3] > 9:
            tmp = '>9'
        elif line[1][3] > 9:
            tmp2 = '>9'
        txt = tmp + '/' + tmp2

    dims = {2:0, 3:1, 5:2, 10:3, 20:4, 40:5}
    ax = plt.gca()
    if line[0][2] > 0 and line[1][2] > 0:
        trans = ax.transData
        annotcoord = [line[0][0], line[1][1]/line[0][1]]
        #plt.text(annotcoord[0], annotcoord[1], txt)
    elif line[0][2] == 0. and line[1][2] == 0.:
        set_trace() # should not occur
    elif line[0][2] == 0.:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        annotcoord = [line[1][0], -line[1][2]/2 + 0.5 + offset*(5-dims[dim])]
        if va == 'top':
            va = 'bottom'
        #plt.text(annotcoord[0], annotcoord[1], txt, transform=trans)
    elif line[1][2] == 0.:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        annotcoord = [line[0][0], line[0][2]/2 + 0.5 - offset*(5-dims[dim])]
        #plt.text(annotcoord[0], annotcoord[1], txt, transform=trans)

    plt.text(annotcoord[0], annotcoord[1], txt, horizontalalignment=ha,
             verticalalignment=va, transform=trans)
    #    if annotcoord[1] > 
    #    annotcoord[0] = 
    # Annotate correspondingly

def main(dsList0, dsList1, minfvalue=1e-8, outputdir='', verbose=True):
    """Returns ERT1/ERT0 comparison figure."""

    plt.rc("axes", labelsize=20, titlesize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("font", size=20)
    plt.rc("legend", fontsize=20)

    dictFun0 = dsList0.dictByFunc()
    dictFun1 = dsList1.dictByFunc()

    for func in set.intersection(set(dictFun0), set(dictFun1)):
        dictDim0 = dictFun0[func].dictByDim()
        dictDim1 = dictFun1[func].dictByDim()

        if isBenchmarkinfosFound:
            title = funInfos[func]
        else:
            title = ''

        filename = os.path.join(outputdir,'ppcmpfig_f%d' % (func))

        dims = sorted(set.intersection(set(dictDim0), set(dictDim1)))

        handles = []
        dataperdim = {}
        fvalueswitch = {}
        for i, dim in enumerate((2, 3, 5, 10, 20, 40)):
            try:
                entry0 = dictDim0[dim][0]
                entry1 = dictDim1[dim][0]
            except KeyError:
                continue

            # generateData:
            data = generateData(entry0, entry1)
            dataperdim[dim] = data

            # TODO: hack, modify slightly so line goes to 'zero'
            if minfvalue:
                for d in data:
                    tmp = d[:, 0]
                    tmp[tmp == 0] = min(min(tmp[tmp > 0]), minfvalue)**2

            # plot
            idx = numpy.isfinite(data[0][:, 1]) * numpy.isfinite(data[1][:, 1])
            ydata = data[1][idx, 1]/data[0][idx, 1]
            plt.plot(data[0][idx, 0], ydata, ls='--', color=colors[i],
                     lw=linewidth)

            # This is one possibility:
            #idx = (data[0][:, 3] >= 5) * (data[1][:, 3] >= 5)
            idx = ((data[0][:, 1] <= 3 * numpy.median(entry0.maxevals))
                   * (data[1][:, 1] <= 3 * numpy.median(entry1.maxevals)))
            #if func==5:
            #    set_trace()
            fvalueswitch[dim] = min(data[0][idx, 0])
            ydata = data[1][idx, 1]/data[0][idx, 1]
            plt.plot(data[0][idx, 0], ydata, color=colors[i], lw=linewidth)
            #h = plotERTRatio(data, plotargs)

        beautify(xmin=minfvalue)
        #beautify()
        ax = plt.gca()
        # Freeze the boundaries
        ax.set_autoscale_on(False)
        #trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        # Plot everything else
        for i, dim in enumerate((2, 3, 5, 10, 20, 40)):

            try:
                data = dataperdim[dim]
            except KeyError:
                continue

            # annotation
            annotate(data, dim, minfvalue)

            tmp0 = numpy.isfinite(data[0][:, 1])
            tmp1 = numpy.isfinite(data[1][:, 1])
            # Determine which algorithm went further
            algstoppedlast = 0
            algstoppedfirst = 1

            if numpy.sum(tmp0) < numpy.sum(tmp1):
                algstoppedlast = 1
                algstoppedfirst = 0

            #marker if an algorithm stopped
            idx = tmp0 * tmp1
            ydata = data[1][idx, 1]/data[0][idx, 1]
            plt.plot((data[0][idx, 0][-1], ), (ydata[-1], ), marker='D',
                     color=colors[i], lw=linewidth, label='%2d-D' % dim,
                     markeredgecolor=colors[i], markerfacecolor='None',
                     markeredgewidth=linewidth, markersize=3*linewidth)
            tmpy = ydata[-1]

            # plot probability of success line
            dataofinterest = data[algstoppedlast]

            tmp = numpy.nonzero(idx)[0][-1] # Why [0]?
            # add the last line for which both algorithm still have a success
            idx = (data[algstoppedfirst][:, 2] == 0.) * (dataofinterest[:, 2] > 0.)
            idx[tmp] = True

            if len(idx) == 0 or not idx.any():
                continue

            ymin, ymax = plt.ylim()
            #orientation = -1
            ybnd = ymin
            if algstoppedlast == 0:
                ybnd = ymax
                #orientation = 1

            #ydata = orientation * dataofinterest[idx, 2] / 2 + 0.5
            ydata = numpy.power(10, numpy.log10(ybnd) * (dataofinterest[idx, 2]
                                                         -offset*(5-i)*numpy.log10(ymax/ymin)/numpy.abs(numpy.log10(ybnd))))

            ls = '-'
            if dataofinterest[idx, 0][0] < fvalueswitch[dim]:
                ls = '--'

            plt.plot([dataofinterest[idx, 0][0]]*2,
                     (tmpy, ydata[0]), ls=ls, lw=linewidth, color=colors[i])

            plt.plot(dataofinterest[idx, 0], ydata, ls='--', lw=linewidth,
                     color=colors[i])

            # marker for when the first algorithm stop
            plt.plot((dataofinterest[idx, 0][0], ), (ydata[0], ), marker='D',
                     color=colors[i], lw=linewidth, markeredgecolor=colors[i],
                     markerfacecolor='None', markeredgewidth=linewidth,
                     markersize=3*linewidth)

            # marker for the other algorithm stop
            plt.plot((dataofinterest[idx, 0][-1], ), (ydata[-1], ), marker='D',
                     color=colors[i], lw=linewidth, markeredgecolor=colors[i],
                     markerfacecolor='None', markeredgewidth=linewidth,
                     markersize=3*linewidth)
            

        if isBenchmarkinfosFound:
            plt.title(funInfos[func])

        if func in (1, 24, 101, 130):
            plt.legend(loc='best')

        # save
        saveFigure(filename, figFormat=figformat, verbose=verbose)
        plt.close()
        #set_trace()

    plt.rcdefaults()

