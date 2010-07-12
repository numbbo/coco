#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic routines for figure generation."""

from operator import itemgetter
from itertools import groupby
import numpy
from matplotlib import pyplot as plt
from pdb import set_trace

def saveFigure(filename, figFormat=('eps', 'pdf'), verbose=True):
    """Save figure into a file.
    Need to make sure the file location is available.
    """

    if isinstance(figFormat, basestring):
        try:
            plt.savefig(filename + '.' + figFormat, dpi = 300,
                        format=figFormat)
            if verbose:
                print 'Wrote figure in %s.' % (filename + '.' + figFormat)
        except IOError:
            warnings.warn('%s is not writeable.' % (filename + '.' + figFormat))
    else:
        #if not isinstance(figFormat, basestring):
        for entry in figFormat:
            try:
                plt.savefig(filename + '.' + entry, dpi = 300,
                            format=entry)
                if verbose:
                    print 'Wrote figure in %s.' %(filename + '.' + entry)
            except IOError:
                warnings.warn('%s is not writeable.' % (filename + '.' + entry))

def plotUnifLogXMarkers(x, y, nbperdecade, kwargs={}):
    """Proxy plot function: puts markers regularly spaced on the x-scale.

    This method generates plots with markers regularly spaced on the x-scale
    whereas the matplotlib.pyplot.plot function will put markers on data
    points.

    This method outputs a list of three lines.Line2D objects: the first with
    the line style, the second for the markers and the last for the label.
    """

    res = plt.plot(x, y, **kwargs)

    def downsample(xdata, ydata):
        """Downsample arrays of data, zero-th column elements are evenly spaced."""

        # powers of ten 10**(i/nbperdecade)
        minidx = numpy.ceil(numpy.log10(min(xdata)) * nbperdecade)
        maxidx = numpy.floor(numpy.log10(max(xdata)) * nbperdecade) + 1
        alignmentdata = 10.**(numpy.arange(minidx, maxidx)/nbperdecade)
        # Look in the original data
        res = []
        tmp = numpy.argsort(xdata)
        for i in alignmentdata:
            res.append(ydata[tmp][xdata[tmp] <= i][-1])

        return alignmentdata, res

    if 'marker' in kwargs and len(x) > 0:
        x2, y2 = downsample(x, y)
        res2 = plt.plot(x2, y2, **kwargs)
        for attr in ('linestyle', 'marker', 'markeredgewidth',
                     'markerfacecolor', 'markeredgecolor',
                     'markersize', 'color', 'linewidth', 'markeredgewidth'):
            plt.setp(res2, attr, plt.getp(res[0], attr))
        plt.setp(res2, linestyle='', label='')
        res.extend(res2)

    if 'label' in kwargs:
        res3 = plt.plot([], [], **kwargs)
        for attr in ('linestyle', 'marker', 'markeredgewidth',
                     'markerfacecolor', 'markeredgecolor',
                     'markersize', 'color', 'linewidth', 'markeredgewidth'):
            plt.setp(res3, attr, plt.getp(res[0], attr))
        res.extend(res3)

    plt.setp(res[0], marker='', label='')
    return res

def consecutiveNumbers(data):
    """Find runs of consecutive numbers using groupby.

    The key to the solution is differencing with a range so that consecutive
    numbers all appear in same group.
    Useful for determining ranges of functions.
    """

    res = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        tmp = list(str(i) for i in map(itemgetter(1), g))
        if len(tmp) <= 2 :
            res.append(', '.join(tmp))
        else:
            res.append('-'.join((tmp[0], tmp[-1])))

    return ', '.join(res)
