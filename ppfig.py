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

        assert all(xdata == numpy.sort(xdata)) and all(ydata == numpy.sort(ydata))
        # otherwise xdata and ydata need to be sorted
        # they cannot be sorted individually

        # powers of ten 10**(i/nbperdecade)
        minidx = numpy.ceil(numpy.log10(min(xdata)) * nbperdecade)
        maxidx = numpy.floor(numpy.log10(max(xdata)) * nbperdecade)
        alignmentdata = 10.**(numpy.arange(minidx, maxidx + 1)/nbperdecade)
        # Look in the original data
        res = []
        for i in alignmentdata:
            if (xdata > i).any():
                res.append(ydata[xdata > i][0])
            else:
                res.append(ydata[-1])

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
    """Groups a sequence of integers into ranges of consecutive numbers.
    For instance: [0, 1, 2, 4, 5, 7, 8, 9] -> "0-2, 4, 5, 7-9"

    Range of consecutive numbers is at least 3 (therefore [4, 5] is represented
    as "4, 5".
    """

    # TODO: give reference: it is seemingly in the Python Library Reference

    res = []
    tmp = groupByRange(data)
    for i in tmp:
        tmpstring = list(str(j) for j in i)
        if len(i) <= 2 : # This means length of ranges are at least 3
            res.append(', '.join(tmpstring))
        else:
            res.append('-'.join((tmpstring[0], tmpstring[-1])))

    return ', '.join(res)

def groupByRange(data):
    """Groups a sequence of integers into ranges of consecutive numbers.
    Helper function of consecutiveNumbers(data), returns a list of lists.

    The key to the solution is differencing with a range so that consecutive
    numbers all appear in same group.
    Useful for determining ranges of functions.

    Ref: http://docs.python.org/release/3.0.1/library/itertools.html
    """

    res = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        res.append(list(i for i in map(itemgetter(1), g)))

    return res

