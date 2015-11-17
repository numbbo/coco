#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Single data set results output module."""

import matplotlib.pyplot as plt
import numpy as np
from bbob_pproc import toolsstats
from pdb import set_trace

lineprops = ('color', 'linestyle', 'linewidth', 'marker', 'markeredgecolor',
             'markeredgewidth', 'markerfacecolor', 'markerfacecoloralt',
             'markersize')

def beautify():
    a = plt.gca()
    a.set_xscale('log')
    a.set_yscale('log')
    a.grid()
    a.set_ylabel('log10 of Df')
    a.set_xlabel('log10 of run lengths')
    tmp = a.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(np.log10(i)))
    a.set_yticklabels(tmp2)
    tmp = a.get_xticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(np.log10(i)))
    a.set_xticklabels(tmp2)
    return a
    # TODO: label size and line width...

def plot(dataset, **kwargs):
    """Plot function values versus function evaluations."""

    res = []
    for i in dataset.funvals[:, 1:].transpose(): # loop over the rows of the transposed array
        h = plt.plot(dataset.funvals[:, 0], i, **kwargs)
        res.extend(h)

    #TODO: make sure each line have the same properties plot properties
    return res

def plot2(dataset, **kwargs):
    """Plot function values versus function evaluations.

    Plus (+) markers for final function values and run lengths and
    dashed lines.
    Lines for the max and min and median.

    Plot markers for all quartiles when:
    * the first run stops
    * the median run stops
    * the maximum number of evaluations is reached
    * at 1/2 of the first run stops

    """

    def _findfrombelow(data, x, log=True):
        if log:
            tmpdata = np.log(data)
            x = np.log(x)
        else:
            tmpdata = data
        return max(data[tmpdata - x <= 0])

    prctiles = np.array((0, 25, 50, 75, 100))
    medidx = 2

    # get data
    data = []
    for i in dataset.funvals[:, 1:]:
        data.append(toolsstats.prctile(i, prctiles))
    data = np.asarray(data)
    xdata = dataset.funvals[:, 0]
    res = []
    # plot
    res.extend(plt.plot(xdata[xdata <= np.median(dataset.maxevals)],
                        data[xdata <= np.median(dataset.maxevals), medidx],
                        **kwargs))
    props = dict((i, plt.getp(res[-1], i)) for i in lineprops)
    res.extend(plt.plot(xdata[xdata <= min(dataset.maxevals)],
                        data[xdata <= min(dataset.maxevals), 0],
                        **props))
    res.extend(plt.plot(xdata, data[:, -1], **props))
    for i in res:
        plt.setp(i, 'marker', '')

    finalpoints = np.vstack((dataset.maxevals, dataset.finalfunvals)).T
    sortedfpoints = np.vstack(sorted(finalpoints,
                                     cmp=lambda x, y: cmp(list(x), list(y))))
    # sort final points like a list of 2-element sequences
    res.extend(plt.plot(sortedfpoints[:, 0], sortedfpoints[:, 1], **props))
    plt.setp(res[-1], marker='+', markeredgewidth=props['linewidth'],
             markersize=5*props['linewidth'], linestyle='')

    xmarkers = (_findfrombelow(xdata, min(dataset.maxevals) ** .5),
                min(dataset.maxevals),
                _findfrombelow(xdata, np.median(dataset.maxevals)),
                max(dataset.maxevals))
    xmarkersprctile = (0, 0, 50, 100)
    for x, p in zip(xmarkers, xmarkersprctile):
        tmp = (p <= prctiles)
        res.extend(plt.plot(np.sum(tmp) * [x], data[xdata == x, tmp],
                            **props))
        plt.setp(res[-1], linestyle='')
    return res

def generatefig(dsList):
    """Plot function values versus function evaluations for multiple sets."""

    for i in dsList:
        plot(i)
        beautify()

def main(dsList, outputdir, verbose=True):
    """Generate output image files of function values vs. function evaluations."""

    generatefigure(dsList)
    # TODO: save, close

