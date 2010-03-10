#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Scatter Plot.
For two algorithms, ERTs(given target function value) can also be plotted in a
scatter plot (log(ERT0) vs. log(ERT1)), which results in a very attractive
presentation, see the slides of Frank Hutter at
http://www.msr-inria.inria.fr/events-news/first-search-biology-day. The
advantage is that the absolute values do not get lost. The disadvantage
(in our case minor) is that there is an upper limit of data that can be
displayed.
"""

import os
import numpy
from pdb import set_trace
from matplotlib import pyplot as plt

from bbob_pproc import readalign

figFormat = ('eps', 'pdf')
colors = ('k', 'b', 'c', 'g', 'y', 'm', 'r', 'k', 'k', 'c', 'r', 'm')
markers = ('o', 'v', 's', '+', 'x', 'D')

def beautify():
    a = plt.gca()
    a.set_xscale('log')
    a.set_yscale('log')
    #a.set_xlabel('ERT0')
    #a.set_ylabel('ERT1')
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.plot([max(xmin, ymin), min(xmax, ymax)],
             [max(xmin, ymin), min(xmax, ymax)], ls='--', color='k')
    plt.xlim(max(xmin, ymin), min(xmax, ymax))
    plt.ylim(max(xmin, ymin), min(xmax, ymax))
    a.set_aspect(1./a.get_data_ratio())
    plt.grid(True)
    tmp = a.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    a.set_yticklabels(tmp2)
    a.set_xticklabels(tmp2)

def saveFigure(filename, figFormat=('eps', 'pdf'), verbose=True):

    if isinstance(figFormat, basestring):
        plt.savefig(filename + '.' + figFormat, dpi = 300,
                    format=figFormat)
        if verbose:
            print 'Wrote figure in %s.' %(filename + '.' + figFormat)
    else:
        if not isinstance(figFormat, basestring):
            for entry in figFormat:
                plt.savefig(filename + '.' + entry, dpi = 300,
                            format=entry)
                if verbose:
                    print 'Wrote figure in %s.' %(filename + '.' + entry)

def generateData(ds0, ds1):
    #Align ert arrays on targets
    array0 = numpy.vstack([ds0.target, ds0.ert]).transpose()
    array1 = numpy.vstack([ds1.target, ds1.ert]).transpose()
    data = readalign.alignArrayData(readalign.HArrayMultiReader([array0, array1]))

    #Downsample?
    adata = data[data[:, 0]<=10, :]
    try:
        adata = adata[adata[:, 0]>=1e-8, :]
    except IndexError:
        #empty data
        pass
        #set_trace()

    targets = adata[:, 0]
    ert0 = adata[:, 1]
    ert1 = adata[:, 2]

    return targets, ert0, ert1

def main(dsList0, dsList1, outputdir, verbose=True):

    plt.rc("axes", labelsize=20, titlesize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("font", size=20)
    plt.rc("legend", fontsize=20)

    dictFunc0 = dsList0.dictByFunc()
    dictFunc1 = dsList1.dictByFunc()
    funcs = set(dictFunc0.keys()) & set(dictFunc1.keys())

    for f in funcs:
        dictDim0 = dictFunc0[f].dictByDim()
        dictDim1 = dictFunc1[f].dictByDim()
        dims = set(dictDim0.keys()) & set(dictDim1.keys())
        #set_trace()

        for i, d in enumerate((2, 3, 5, 10, 20, 40)):
            try:
                entry0 = dictDim0[d][0] # should be only one element
                entry1 = dictDim1[d][0] # should be only one element
            except IndexError:
                continue

            targets, xdata, ydata = generateData(entry0, entry1)
            plt.plot(xdata, ydata, ls='', color=colors[i], marker=markers[i],
                     markerfacecolor='None', markeredgecolor=colors[i],
                     markersize=10, markeredgewidth=3)

        beautify()

        filename = os.path.join(outputdir, 'scatter_f%d' % f)
        saveFigure(filename, figFormat=figFormat, verbose=verbose)
        plt.close()

    plt.rcdefaults()
