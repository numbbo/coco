#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic routines for figure generation."""

from operator import itemgetter
from itertools import groupby
import warnings
import numpy
from matplotlib import pyplot as plt
from pdb import set_trace
from bbob_pproc import bootstrap

def saveFigure(filename, figFormat=('eps', 'pdf'), verbose=True):
    """Save figure into an image file."""

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

def plotUnifLogXMarkers(x, y, nbperdecade, logscale=True, **kwargs):
    """Proxy plot function: markers are evenly spaced on the log x-scale

    This method generates plots with markers regularly spaced on the
    x-scale whereas the matplotlib.pyplot.plot function will put markers
    on data points.

    This method outputs a list of three lines.Line2D objects: the first
    with the line style, the second for the markers and the last for the
    label.

    """

    res = plt.plot(x, y, **kwargs)

    def downsample(xdata, ydata, logscale=True):
        """Downsample arrays of data."""

        # powers of ten 10**(i/nbperdecade)
        minidx = numpy.ceil(numpy.log10(min(xdata)) * nbperdecade)
        maxidx = numpy.floor(numpy.log10(max(xdata)) * nbperdecade)
        alignmentdata = 10.**(numpy.arange(minidx, maxidx + 1)/nbperdecade)
        xdataarray = numpy.array(xdata)

        # Look in the original data
        res = []
        for i in alignmentdata:
            diffarray = xdataarray - i
            assert (diffarray >= 0.).any() and (diffarray <= 0.).any()
            if (diffarray == 0.).any():
                idx = (diffarray == 0.)
                if logscale:
                    y = 10.**((numpy.log10(max(ydata[idx])) + numpy.log10(min(ydata[idx])))/2.)
                else:
                    y = (max(ydata[idx]) + min(ydata[idx]))/2.
            else:
                # find the indices of the element that is the closest greater than i
                idx1 = numpy.nonzero((diffarray == min(diffarray[diffarray >= 0.])))[0]
                # find the indices of the element that is the closest smaller than i
                idx2 = numpy.nonzero((diffarray == max(diffarray[diffarray <= 0.])))[0]
                x1 = xdata[idx1][0]
                x2 = xdata[idx2][0]
                if abs(min(idx1)-max(idx2)) > abs(max(idx1)-min(idx2)):
                    y1 = ydata[max(idx1)]
                    y2 = ydata[min(idx2)]
                else:
                    y1 = ydata[min(idx1)]
                    y2 = ydata[max(idx2)]
                # log interpolation / semi-log
                if logscale:
                    y = 10.**(numpy.log10(y1) + (numpy.log10(i) - numpy.log10(x1)) * (numpy.log10(y2) - numpy.log10(y1))/(numpy.log10(x2) - numpy.log10(x1)))
                else:
                    y = y1 + (numpy.log10(i) - numpy.log10(x1)) * (y2 - y1)/(numpy.log10(x2) - numpy.log10(x1))
            res.append(y)

        return alignmentdata, res

    if 'marker' in kwargs and len(x) > 0:
        x2, y2 = downsample(x, y, logscale=logscale)
        try:
            res2 = plt.plot(x2, y2, **kwargs)
        except ValueError:
            raise # TODO
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

    Example::

      >>> import bbob_pproc as bb
      >>> bb.ppfig.consecutiveNumbers([0, 1, 2, 4, 5, 7, 8, 9])
      '0-2, 4, 5, 7-9'

    Range of consecutive numbers is at least 3 (therefore [4, 5] is
    represented as "4, 5").

    """
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
    The key to the solution is differencing with a range so that
    consecutive numbers all appear in same group.
    Useful for determining ranges of functions.
    Ref: http://docs.python.org/release/3.0.1/library/itertools.html

    """
    res = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        res.append(list(i for i in map(itemgetter(1), g)))

    return res

def beautify():
    """ Customize a figure by adding a legend, axis label, etc."""
    # TODO: what is this function for?
    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_yscale("log")

    # Grid options
    axisHandle.grid(True)

    ymin, ymax = plt.ylim()
    plt.ylim(ymin=10**-0.2, ymax=ymax) # Set back the default maximum.

    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    axisHandle.set_yticklabels(tmp2)
    axisHandle.set_ylabel('log10 of ERT')

def generateData(dataSet, targetFuncValue):
    """Returns an array of results to be plotted.

    1st column is ert, 2nd is  the number of success, 3rd the success
    rate, 4th the sum of the number of  function evaluations, and
    finally the median on successful runs.

    """
    res = []
    data = []

    it = iter(reversed(dataSet.evals))
    i = it.next()
    prev = numpy.array([numpy.nan] * len(i))

    while i[0] <= targetFuncValue:
        prev = i
        try:
            i = it.next()
        except StopIteration:
            break

    data = prev[1:].copy() # keep only the number of function evaluations.
    succ = (numpy.isnan(data) == False)
    if succ.any():
        med = bootstrap.prctile(data[succ], 50)[0]
        #Line above was modified at rev 3050 to make sure that we consider only
        #successful trials in the median
    else:
        med = numpy.nan

    data[numpy.isnan(data)] = dataSet.maxevals[numpy.isnan(data)]

    res = []
    res.extend(bootstrap.sp(data, issuccessful=succ, allowinf=False))
    res.append(numpy.mean(data)) #mean(FE)
    res.append(med)

    return numpy.array(res)

def plot(dsList, _valuesOfInterest=(10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
         isbyinstance=True, kwargs={}):
    """From a DataSetList, plot a graph."""

    #set_trace()
    res = []

    valuesOfInterest = list(_valuesOfInterest)
    valuesOfInterest.sort(reverse=True)

    def transform(dsList):
        """Create dictionary of instances."""

        class StrippedUpDS():
            """Data Set stripped up of everything."""

            pass

        res = {}
        for i in dsList:
            dictinstance = i.createDictInstance()
            for j, idx in dictinstance.iteritems():
                tmp = StrippedUpDS()
                idxs = list(k + 1 for k in idx)
                idxs.insert(0, 0)
                tmp.evals = i.evals[:, numpy.r_[idxs]].copy()
                tmp.maxevals = i.maxevals[numpy.ix_(idx)].copy()
                res.setdefault(j, [])
                res.get(j).append(tmp)
        return res
    
    for i in range(len(valuesOfInterest)):

        succ = []
        unsucc = []
        displaynumber = []
        data = []

        dictX = transform(dsList)
        for x in sorted(dictX.keys()):
            dsListByX = dictX[x]
            for j in dsListByX:
                tmp = generateData(j, valuesOfInterest[i])
                if tmp[2] > 0: #Number of success is larger than 0
                    succ.append(numpy.append(x, tmp))
                    if tmp[2] < j.nbRuns():
                        displaynumber.append((x, tmp[0], tmp[2]))
                else:
                    unsucc.append(numpy.append(x, tmp))

        if succ:
            tmp = numpy.vstack(succ)
            #ERT
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))
            #median
            tmp2 = plt.plot(tmp[:, 0], tmp[:, -1], **kwargs)
            plt.setp(tmp2, linestyle='', marker='+', markersize=30, markeredgewidth=5)
            #, color=colors[i], linestyle='', marker='+', markersize=30, markeredgewidth=5))
            res.extend(tmp2)

        # To have the legend displayed whatever happens with the data.
        tmp = plt.plot([], [], **kwargs)
        plt.setp(tmp, label=' %+d' % (numpy.log10(valuesOfInterest[i])))
        res.extend(tmp)

        #Only for the last target function value
        if unsucc:
            tmp = numpy.vstack(unsucc) # tmp[:, 0] needs to be sorted!
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))

    if displaynumber: #displayed only for the smallest valuesOfInterest
        for j in displaynumber:
            t = plt.text(j[0], j[1]*1.85, "%.0f" % j[2],
                         horizontalalignment="center",
                         verticalalignment="bottom")
            res.append(t)

    return res
