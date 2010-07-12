#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates ERTs and convergence figures for the comparison of 2 algorithms."""

from __future__ import absolute_import

import os
import sys
import warnings
import matplotlib.pyplot as plt
from pdb import set_trace
try:
    from matplotlib.transforms import blended_transform_factory as blend
except ImportError:
    # compatibility matplotlib 0.8
    from matplotlib.transforms import blend_xy_sep_transform as blend

import numpy

from bbob_pproc import bootstrap, readalign
from bbob_pproc.bootstrap import ranksums
from bbob_pproc.ppfig import saveFigure, plotUnifLogXMarkers
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
styles = [{'color': 'c', 'marker': '+', 'markeredgecolor': 'c',
           'markerfacecolor': 'None'},
          {'color': 'g', 'marker': 'v', 'markeredgecolor': 'g',
           'markerfacecolor': 'None'},
          {'color': 'b', 'marker': '*', 'markeredgecolor': 'b',
           'markerfacecolor': 'None'},
          {'color': 'k', 'marker': 'o', 'markeredgecolor': 'k',
           'markerfacecolor': 'None'},
          {'color': 'r', 'marker': 's', 'markeredgecolor': 'r',
           'markerfacecolor': 'None'},
          {'color': 'm', 'marker': 'D', 'markeredgecolor': 'm',
           'markerfacecolor': 'None'},
          {'color': 'k'},
          {'color': 'y'},
          {'color': 'k'},
          {'color': 'c'},
          {'color': 'r'},
          {'color': 'm'}]
#from scatter
#colors = ('c', 'g', 'b', 'k', 'r', 'm', 'k', 'y', 'k', 'c', 'r', 'm')
#markers = ('+', 'v', '*', 'o', 's', 'D', 'x')
linewidth = 3
offset = 0.005
incrstars = 1.5
fthresh = 1e-8
xmax = 1000

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

def generateData(entry0, entry1, fthresh=None, downsampling=None):

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
    tmpdata0 = tmpdata0[::downsampling] #downsampling
    tmpdata1 = tmpdata1[::downsampling]
    data0 = computeERT(tmpdata0, entry0.maxevals)
    data1 = computeERT(tmpdata1, entry1.maxevals)

    if fthresh and (tmpdata0[:, 0] < fthresh).any():
        if not (tmpdata0[:, 0] == fthresh).any():
            tmp0 = entry0.detEvals([fthresh])[0]
            tmp0 = numpy.reshape(numpy.insert(tmp0, 0, fthresh), (1, -1))
            tmp0 = computeERT(tmp0, entry0.maxevals)
            data0 = numpy.concatenate((data0, tmp0))

            tmp1 = entry1.detEvals([fthresh])[0]
            tmp1 = numpy.reshape(numpy.insert(tmp1, 0, fthresh), (1, -1))
            tmp1 = computeERT(tmp1, entry1.maxevals)
            data1 = numpy.concatenate((data1, tmp1))

        data0 = data0[data0[:, 0] >= fthresh]
        data1 = data1[data1[:, 0] >= fthresh]

    if xmax:
        data0 = data0[data0[:, 0] <= xmax]
        data1 = data1[data1[:, 0] <= xmax]
        # TODO: watch that it does not become empty.
        #set_trace()

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
    yax = ax.get_yaxis()
    ax.set_xscale('log')
    ax.set_yscale('log')

    ymin, ymax = plt.ylim()
    ybnd = max(1./ymin, ymax)
    plt.ylim(1./ybnd, ybnd)
    if ybnd < 100:
        yax.grid(True, which='minor')

    # We are setting xmin
    if xmin:
        plt.xlim(xmin=xmin)
    plt.xlim(xmax=xmax)
    ax.invert_xaxis()

    # Annotate figure
    ax.set_xlabel('log10(Delta ftarget)')
    ax.set_ylabel(r'log10(ERT1/ERT0) or ~#succ')  # TODO: replace hard-coded 15
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
    tmp = ax.get_yticklines()
    tmp.extend(yax.get_minorticklines())
    #set_trace()
    for i in tmp:
        i.set_markeredgewidth(2)

def annotate(entry0, entry1, dim, minfvalue=1e-8, nbtests=1):
    """Display some annotations associated to the graphs generated."""

    isEarlyStop = False
    ha = 'left'
    va = 'center'
    lastfvalue = min(entry0.evals[-1][0], entry1.evals[-1][0])
    if minfvalue < lastfvalue:
        isEarlyStop = True
        #ha = 'center'
        #va = 'top'

    if not minfvalue or minfvalue < lastfvalue:
        minfvalue = lastfvalue

    line = []
    data0 = entry0.detEvals([minfvalue])[0]
    evals0 = data0.copy()
    succ = (numpy.isnan(evals0) == False)
    evals0[numpy.isnan(evals0)] = entry0.maxevals[numpy.isnan(evals0)]
    line.append(bootstrap.sp(evals0, issuccessful=succ))
    data1 = entry1.detEvals([minfvalue])[0]
    evals1 = data1.copy()
    succ = (numpy.isnan(evals1) == False)
    evals1[numpy.isnan(evals1)] = entry1.maxevals[numpy.isnan(evals1)]
    line.append(bootstrap.sp(evals1, issuccessful=succ))

    # What's the situation?
    txt = '%dD' % dim
    if (line[0][2] > 0 and line[1][2] > 0 and line[1][2] < 10):
        tmp = str(int(line[1][2]))
        tmp2 = str(int(line[0][2]))
        txt = tmp + '/' + tmp2

    dims = {2:0, 3:1, 5:2, 10:3, 20:4, 40:5}
    ax = plt.gca()
    assert line[0][2] > 0 or line[1][2] > 0
    signdata = line[1][0] - line[0][0]

    if line[0][2] > 0 and line[1][2] > 0:
        trans = ax.transData
        annotcoord = [minfvalue, line[1][0]/line[0][0]]
    elif line[0][2] == 0:
        trans = blend(ax.transData, ax.transAxes)
        annotcoord = [minfvalue, -line[1][1]/2 + 0.5 + offset*(5-dims[dim])]
        #if va == 'top':
        #    va = 'bottom'
    elif line[1][2] == 0:
        trans = blend(ax.transData, ax.transAxes)
        annotcoord = [minfvalue, line[0][1]/2 + 0.5 - offset*(5-dims[dim])]

    plt.text(annotcoord[0], annotcoord[1], txt, horizontalalignment=ha,
             verticalalignment=va, transform=trans)

    #ranksum test
    line0 = numpy.power(data0, -1.)
    line0[numpy.isnan(line0)] = -entry0.finalfunvals[numpy.isnan(line0)]
    line1 = numpy.power(data1, -1.)
    line1[numpy.isnan(line1)] = -entry1.finalfunvals[numpy.isnan(line1)]
    # one-tailed statistics: scipy.stats.mannwhitneyu, two-tailed statistics: scipy.stats.ranksums
    z, p = ranksums(line0, line1)
    # Set the correct line in data0 and data1
    nbstars = 0
    # sign of z-value and data must agree
    if ((nbtests * p) < 0.05 and (z * signdata) > 0):
        nbstars = -numpy.ceil(numpy.log10(nbtests * p))
    if nbstars > 0:
        xstars = annotcoord[0] * numpy.power(incrstars, numpy.arange(1., 1. + nbstars))
        # the additional slicing [0:int(nbstars)] is due to
        # numpy.arange(1., 1. - 0.1 * nbstars, -0.1) not having the right number
        # of elements due to numerical error
        ystars = [annotcoord[1]] * nbstars

        try:
            h = plt.plot(xstars, ystars, marker='*', ls='', color='w',
                         markersize=5*linewidth, markeredgecolor='k',
                         markerfacecolor='None',
                         zorder=20, markeredgewidth = 0.4 * linewidth,
                         transform=trans, clip_on=False)
        except KeyError:
            #Version problem
            h = plt.plot(xstars, ystars, marker='+', ls='', color='w',
                         markersize=2.5*linewidth, markeredgecolor='k',
                         zorder=20, markeredgewidth = 0.2 * linewidth,
                         transform=trans, clip_on=False)

def main(dsList0, dsList1, minfvalue=1e-8, outputdir='', verbose=True):
    """Returns ERT1/ERT0 comparison figure."""

    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

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
        nbtests = 0
        for i, dim in enumerate((2, 3, 5, 10, 20, 40)):
            try:
                entry0 = dictDim0[dim][0]
                entry1 = dictDim1[dim][0]
            except KeyError:
                continue

            nbtests += 1
            # generateData:
            data = generateData(entry0, entry1, fthresh=fthresh)
            dataperdim[dim] = data

            if len(data[0]) == 0 and len(data[1]) == 0:
                continue

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

            # This is only one possibility:
            #idx = (data[0][:, 3] >= 5) * (data[1][:, 3] >= 5)
            idx = ((data[0][:, 1] <= 3 * numpy.median(entry0.maxevals))
                   * (data[1][:, 1] <= 3 * numpy.median(entry1.maxevals)))

            if not idx.any():
                fvalueswitch[dim] = numpy.inf
                # Hack: fvalueswitch is the smallest value of f where the line
                # was still solid.
                continue

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
                entry0 = dictDim0[dim][0]
                entry1 = dictDim1[dim][0]
                data = dataperdim[dim]
            except KeyError:
                continue

            if len(data[0]) == 0 and len(data[1]) == 0:
                continue

            # annotation
            annotate(entry0, entry1, dim, minfvalue, nbtests=nbtests)

            tmp0 = numpy.isfinite(data[0][:, 1])
            tmp1 = numpy.isfinite(data[1][:, 1])
            idx = tmp0 * tmp1

            if not idx.any():
                continue

            #Do not plot anything else if it happens after minfvalue
            if data[0][idx, 0][-1] <= minfvalue:
                # hack for the legend
                plt.plot((data[0][idx, 0][-1]**2, ), (ydata[-1], ), marker='D',
                         color=colors[i], lw=linewidth, label='%2d-D' % dim,
                         markeredgecolor=colors[i], markerfacecolor='None',
                         markeredgewidth=linewidth, markersize=3*linewidth)
                continue

            # Determine which algorithm went further
            algstoppedlast = 0
            algstoppedfirst = 1

            if numpy.sum(tmp0) < numpy.sum(tmp1):
                algstoppedlast = 1
                algstoppedfirst = 0

            #marker if an algorithm stopped
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

            if numpy.sum(idx) <= 1:#len(idx) == 0 or not idx.any():
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

            #Do not plot anything else if it happens after minfvalue
            if dataofinterest[idx, 0][-1] <= minfvalue:
                continue
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

    #plt.rcdefaults()

def main2(dsList0, dsList1, minfvalue=1e-8, outputdir='', verbose=True):
    """Returns ERT1/ERT0 comparison figure.
    For black and white purpose (symbols, etc.)
    """

    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    dictFun0 = dsList0.dictByFunc()
    dictFun1 = dsList1.dictByFunc()

    for func in set.intersection(set(dictFun0), set(dictFun1)):
        dictDim0 = dictFun0[func].dictByDim()
        dictDim1 = dictFun1[func].dictByDim()

        if isBenchmarkinfosFound:
            title = funInfos[func]
        else:
            title = ''

        filename = os.path.join(outputdir,'ppfig2_f%03d' % (func))

        dims = sorted(set.intersection(set(dictDim0), set(dictDim1)))

        handles = []
        dataperdim = {}
        fvalueswitch = {}
        nbtests = 0
        for i, dim in enumerate((2, 3, 5, 10, 20, 40)):
            try:
                entry0 = dictDim0[dim][0]
                entry1 = dictDim1[dim][0]
            except KeyError:
                continue

            nbtests += 1
            # generateData:
            data = generateData(entry0, entry1, fthresh=fthresh)
            dataperdim[dim] = data

            if len(data[0]) == 0 and len(data[1]) == 0:
                continue

            # TODO: hack, modify slightly so line goes to 'zero'
            if minfvalue:
                for d in data:
                    tmp = d[:, 0]
                    tmp[tmp == 0] = min(min(tmp[tmp > 0]), minfvalue)**2

            # plot
            idx = numpy.isfinite(data[0][:, 1]) * numpy.isfinite(data[1][:, 1])
            ydata = data[1][idx, 1]/data[0][idx, 1]
            kwargs = styles[i].copy()
            kwargs['label'] = '%2d-D' % dim
            tmp = plotUnifLogXMarkers(data[0][idx, 0], ydata, 1, kwargs)
            plt.setp(tmp, markersize=3*linewidth)
            plt.setp(tmp[0], ls='--')

            # This is only one possibility:
            #idx = (data[0][:, 3] >= 5) * (data[1][:, 3] >= 5)
            idx = ((data[0][:, 1] <= 3 * numpy.median(entry0.maxevals))
                   * (data[1][:, 1] <= 3 * numpy.median(entry1.maxevals)))

            if not idx.any():
                fvalueswitch[dim] = numpy.inf
                # Hack: fvalueswitch is the smallest value of f where the line
                # was still solid.
                continue

            fvalueswitch[dim] = min(data[0][idx, 0])
            ydata = data[1][idx, 1]/data[0][idx, 1]
            tmp = plotUnifLogXMarkers(data[0][idx, 0], ydata, 1, styles[i])
            plt.setp(tmp[1], markersize=3*linewidth)

        beautify(xmin=minfvalue)
        #beautify()
        ax = plt.gca()
        # Freeze the boundaries
        ax.set_autoscale_on(False)
        #trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        # Plot everything else
        for i, dim in enumerate((2, 3, 5, 10, 20, 40)):
            try:
                entry0 = dictDim0[dim][0]
                entry1 = dictDim1[dim][0]
                data = dataperdim[dim]
            except KeyError:
                continue

            if len(data[0]) == 0 and len(data[1]) == 0:
                continue

            # annotation
            annotate(entry0, entry1, dim, minfvalue, nbtests=nbtests)

            tmp0 = numpy.isfinite(data[0][:, 1])
            tmp1 = numpy.isfinite(data[1][:, 1])
            idx = tmp0 * tmp1

            if not idx.any():
                continue

            #Do not plot anything else if it happens after minfvalue
            if data[0][idx, 0][-1] <= minfvalue:
                # hack for the legend
                continue

            # Determine which algorithm went further
            algstoppedlast = 0
            algstoppedfirst = 1

            if numpy.sum(tmp0) < numpy.sum(tmp1):
                algstoppedlast = 1
                algstoppedfirst = 0

            #marker if an algorithm stopped
            ydata = data[1][idx, 1]/data[0][idx, 1]
            plt.plot((data[0][idx, 0][-1], ), (ydata[-1], ), marker='D', ls='',
                     color=styles[i]['color'], markeredgecolor=styles[i]['color'],
                     markerfacecolor=styles[i]['color'], markersize=4*linewidth)
            tmpy = ydata[-1]

            # plot probability of success line
            dataofinterest = data[algstoppedlast]

            tmp = numpy.nonzero(idx)[0][-1] # Why [0]?
            # add the last line for which both algorithm still have a success
            idx = (data[algstoppedfirst][:, 2] == 0.) * (dataofinterest[:, 2] > 0.)
            idx[tmp] = True

            if numpy.sum(idx) <= 1:#len(idx) == 0 or not idx.any():
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

            tmp = plt.plot([dataofinterest[idx, 0][0]]*2, (tmpy, ydata[0]),
                           **styles[i])
            plt.setp(tmp, ls=ls, marker='')
            tmp = plt.plot((dataofinterest[idx, 0][0], ), (ydata[0], ), marker='D', ls='',
                     color=styles[i]['color'], markeredgecolor=styles[i]['color'],
                     markerfacecolor=styles[i]['color'], markersize=4*linewidth)

            kwargs = styles[i].copy()
            kwargs['ls'] = ls
            tmp = plotUnifLogXMarkers(dataofinterest[idx, 0], ydata, 1, kwargs)
            plt.setp(tmp, markersize=3*linewidth)

            #Do not plot anything else if it happens after minfvalue
            if dataofinterest[idx, 0][-1] <= minfvalue:
                continue
            #plt.plot((dataofinterest[idx, 0][-1], ), (ydata[-1], ), marker='d',
            #         color=styles[i]['color'], markeredgecolor=styles[i]['color'],
            #         markerfacecolor=styles[i]['color'], markersize=4*linewidth)

        if isBenchmarkinfosFound:
            plt.title(funInfos[func])

        if func in (1, 24, 101, 130):
            plt.legend(loc='best')

        # save
        saveFigure(filename, figFormat=figformat, verbose=verbose)
        plt.close()
        #set_trace()

    #plt.rcdefaults()
