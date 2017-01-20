#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates aRT-ratio comparison figures (ECDF) and convergence figures for the comparison of 2 algorithms.

Scale up figures for two algorithms can be done with compall/ppfigs.py

"""

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

import numpy as np 

from .. import toolsstats, readalign, ppfigparam, testbedsettings, toolsdivers
from ..toolsstats import ranksumtest
from ..ppfig import save_figure, plotUnifLogXMarkers
#try:
    #supersede this module own ranksumtest method
    #from scipy.stats import ranksumtest as ranksumtest
#except ImportError:
    #from cocopp.toolsstats import ranksumtest
    #pass

dimensions = (2, 3, 5, 10, 20, 40)
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
linewidth = 3  # overwritten by config.py 
offset = 0.005
incrstars = 1.5
fthresh = 1e-8
xmax = 1000

dimension_index = dict([(dimensions[i], i) for i in xrange(len(dimensions))])

def _generateData(entry0, entry1, fthresh=None, downsampling=None):

    def alignData(i0, i1):
        """Returns two arrays of fevals aligned on function evaluations.
        """

        res = readalign.alignArrayData(readalign.HArrayMultiReader([i0.evals,
                                                                    i1.evals],
                                                                    i0.isBiobjective()))
        idx = 1 + i0.nbRuns()
        data0 = res[:, np.r_[0, 1:idx]]
        data1 = res[:, np.r_[0, idx:idx+i1.nbRuns()]]
        return data0, data1

    def computeERT(hdata, maxevals):
        res = []
        for i in hdata:
            data = i.copy()
            data = data[1:]
            succ = (np.isnan(data)==False)
            if any(np.isnan(data)):
                data[np.isnan(data)] = maxevals[np.isnan(data)]
            tmp = [i[0]]
            tmp.extend(toolsstats.sp(data, issuccessful=succ))
            res.append(tmp)
        return np.vstack(res)

    tmpdata0, tmpdata1 = alignData(entry0, entry1)
    tmpdata0 = tmpdata0[::downsampling] #downsampling
    tmpdata1 = tmpdata1[::downsampling]
    data0 = computeERT(tmpdata0, entry0.maxevals)
    data1 = computeERT(tmpdata1, entry1.maxevals)

    if fthresh and (tmpdata0[:, 0] < fthresh).any():
        if not (tmpdata0[:, 0] == fthresh).any():
            tmp0 = entry0.detEvals([fthresh])[0]
            tmp0 = np.reshape(np.insert(tmp0, 0, fthresh), (1, -1))
            tmp0 = computeERT(tmp0, entry0.maxevals)
            data0 = np.concatenate((data0, tmp0))

            tmp1 = entry1.detEvals([fthresh])[0]
            tmp1 = np.reshape(np.insert(tmp1, 0, fthresh), (1, -1))
            tmp1 = computeERT(tmp1, entry1.maxevals)
            data1 = np.concatenate((data1, tmp1))

        data0 = data0[data0[:, 0] >= fthresh]
        data1 = data1[data1[:, 0] >= fthresh]

    if xmax:
        data0 = data0[data0[:, 0] <= xmax]
        data1 = data1[data1[:, 0] <= xmax]
        # TODO: watch that it does not become empty.
        #set_trace()

    return data0, data1

def beautify(xmin=None):
    """Format the figure."""
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
    ax.set_ylabel(r'log10(aRT1/aRT0) or ~#succ')  # TODO: replace hard-coded 15
    ax.grid(True)

    #Tick label handling
    xticks = ax.get_xticks()
    tmp = []
    for i in xticks:
        tmp.append('%d' % round(np.log10(i)))
    ax.set_xticklabels(tmp)

    yticks = ax.get_yticks()
    tmp = []
    for i in yticks:
        tmp.append('%d' % round(np.log10(i)))
    ax.set_yticklabels(tmp)

    # Reverse yticks below 1
    tmp = ax.get_yticks(minor=True)
    tmp[tmp<1] = sorted(1/(tmp[tmp<1]*np.power(10, -2*np.floor(np.log10(tmp[tmp<1]))-1)))
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
    succ = (np.isnan(evals0) == False)
    evals0[np.isnan(evals0)] = entry0.maxevals[np.isnan(evals0)]
    line.append(toolsstats.sp(evals0, issuccessful=succ))
    data1 = entry1.detEvals([minfvalue])[0]
    evals1 = data1.copy()
    succ = (np.isnan(evals1) == False)
    evals1[np.isnan(evals1)] = entry1.maxevals[np.isnan(evals1)]
    line.append(toolsstats.sp(evals1, issuccessful=succ))

    # What's the situation?
    txt = '%dD' % dim
    if (line[0][2] > 0 and line[1][2] > 0 and line[1][2] < 10):
        tmp = str(int(line[1][2]))
        tmp2 = str(int(line[0][2]))
        txt = tmp + '/' + tmp2

    dims = dimension_index
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
    line0 = np.power(data0, -1.)
    line0[np.isnan(line0)] = -entry0.finalfunvals[np.isnan(line0)]
    line1 = np.power(data1, -1.)
    line1[np.isnan(line1)] = -entry1.finalfunvals[np.isnan(line1)]
    # one-tailed statistics: scipy.stats.mannwhitneyu, two-tailed statistics: scipy.stats.ranksumtest
    z, p = ranksumtest(line0, line1)
    # Set the correct line in data0 and data1
    nbstars = 0
    # sign of z-value and data must agree
    if ((nbtests * p) < 0.05 and (z * signdata) > 0):
        nbstars = np.min([5, -np.ceil(np.log10(nbtests * p + 1e-99))])
    if nbstars > 0:
        xstars = annotcoord[0] * np.power(incrstars, np.arange(1., 1. + nbstars))
        # the additional slicing [0:int(nbstars)] is due to
        # np.arange(1., 1. - 0.1 * nbstars, -0.1) not having the right number
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

def main(dsList0, dsList1, minfvalue=1e-8, outputdir=''):
    """Returns aRT1/aRT0 comparison figure."""

    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)
    
    # minfvalue = pproc.TargetValues.cast(minfvalue)

    funInfos = ppfigparam.read_fun_infos()    

    dictFun0 = dsList0.dictByFunc()
    dictFun1 = dsList1.dictByFunc()

    for func in set.intersection(set(dictFun0), set(dictFun1)):
        dictDim0 = dictFun0[func].dictByDim()
        dictDim1 = dictFun1[func].dictByDim()

        filename = os.path.join(outputdir,'ppfig2_f%03d' % (func))

        dims = sorted(set.intersection(set(dictDim0), set(dictDim1)))

        handles = []
        dataperdim = {}
        fvalueswitch = {}
        nbtests = 0
        for i, dim in enumerate(dimensions):
            try:
                entry0 = dictDim0[dim][0]
                entry1 = dictDim1[dim][0]
            except KeyError:
                continue

            nbtests += 1
            # generateData:
            data = _generateData(entry0, entry1, fthresh=fthresh)
            dataperdim[dim] = data

            if len(data[0]) == 0 and len(data[1]) == 0:
                continue

            # TODO: hack, modify slightly so line goes to 'zero'
            if minfvalue:
                for d in data:
                    tmp = d[:, 0]
                    tmp[tmp == 0] = min(min(tmp[tmp > 0]), minfvalue)**2

            # plot
            idx = np.isfinite(data[0][:, 1]) * np.isfinite(data[1][:, 1])
            ydata = data[1][idx, 1]/data[0][idx, 1]
            kwargs = styles[i].copy()
            kwargs['label'] = '%2d-D' % dim
            tmp = plotUnifLogXMarkers(data[0][idx, 0], ydata, nbperdecade=1, logscale=True, **kwargs)
            plt.setp(tmp, markersize=3*linewidth)
            plt.setp(tmp[0], ls='--')

            # This is only one possibility:
            #idx = (data[0][:, 3] >= 5) * (data[1][:, 3] >= 5)
            idx = ((data[0][:, 1] <= 3 * np.median(entry0.maxevals))
                   * (data[1][:, 1] <= 3 * np.median(entry1.maxevals)))

            if not idx.any():
                fvalueswitch[dim] = np.inf
                # Hack: fvalueswitch is the smallest value of f where the line
                # was still solid.
                continue

            fvalueswitch[dim] = min(data[0][idx, 0])
            ydata = data[1][idx, 1]/data[0][idx, 1]
            tmp = plotUnifLogXMarkers(data[0][idx, 0], ydata, nbperdecade=1, logscale=True, **styles[i])
            plt.setp(tmp[1], markersize=3*linewidth)

        beautify(xmin=minfvalue)
        #beautify()
        ax = plt.gca()
        # Freeze the boundaries
        ax.set_autoscale_on(False)
        #trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

        # Plot everything else
        for i, dim in enumerate(dimensions):
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

            tmp0 = np.isfinite(data[0][:, 1])
            tmp1 = np.isfinite(data[1][:, 1])
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

            if np.sum(tmp0) < np.sum(tmp1):
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

            tmp = np.nonzero(idx)[0][-1] # Why [0]?
            # add the last line for which both algorithm still have a success
            idx = (data[algstoppedfirst][:, 2] == 0.) * (dataofinterest[:, 2] > 0.)
            idx[tmp] = True

            if np.sum(idx) <= 1:#len(idx) == 0 or not idx.any():
                continue

            ymin, ymax = plt.ylim()
            #orientation = -1
            ybnd = ymin
            if algstoppedlast == 0:
                ybnd = ymax
                #orientation = 1

            #ydata = orientation * dataofinterest[idx, 2] / 2 + 0.5
            ydata = np.power(10, np.log10(ybnd) * (dataofinterest[idx, 2]
                                                         -offset*(5-i)*np.log10(ymax/ymin)/np.abs(np.log10(ybnd))))

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
            tmp = plotUnifLogXMarkers(dataofinterest[idx, 0], ydata, nbperdecade=1, logscale=True, **kwargs)
            plt.setp(tmp, markersize=3*linewidth)

            #Do not plot anything else if it happens after minfvalue
            if dataofinterest[idx, 0][-1] <= minfvalue:
                continue
            #plt.plot((dataofinterest[idx, 0][-1], ), (ydata[-1], ), marker='d',
            #         color=styles[i]['color'], markeredgecolor=styles[i]['color'],
            #         markerfacecolor=styles[i]['color'], markersize=4*linewidth)

        if func in funInfos.keys():
            plt.title(funInfos[func])

        if func in testbedsettings.current_testbed.functions_with_legend:
            toolsdivers.legend(loc='best')

        # save
        save_figure(filename, dsList0[0].algId)
        plt.close()
        #set_trace()

    #plt.rcdefaults()

