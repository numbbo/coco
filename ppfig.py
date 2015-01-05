#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic routines for figure generation."""
from __future__ import absolute_import
import os
from operator import itemgetter
from itertools import groupby
import warnings
import numpy as np
from matplotlib import pyplot as plt
# from pdb import set_trace
from . import genericsettings, toolsstats  # absolute_import => . refers to where ppfig resides in the package


bbox_inches_choices = {  # do we also need pad_inches = 0?
    'svg': 'tight',
}


def saveFigure(filename, figFormat=genericsettings.fig_formats,
               verbose=True):
    """Save figure into an image file.

    `figFormat` can be a string or a list of strings, like
    ``('pdf', 'svg')``

    """
    if isinstance(figFormat, basestring):
        figFormat = (figFormat, )
    for format in figFormat:
        # a hack for making smaller figures for browser display 
        if format == 'svg':
            svg_downsize_factor = 0.8
            # plt.rcParams['font.size'] *= 0.7
            # plt.plot(plt.xlim()[0], plt.ylim()[0], '.')
            # pretty desperate way to get a smaller figure
            plt.gcf().set_size_inches([svg_downsize_factor * v for v in
                                       plt.gcf().get_size_inches()])
        try:
            plt.savefig(filename + '.' + format,
                        dpi = 60 if genericsettings.in_a_hurry else 300,
                        format=format,
                        bbox_inches=bbox_inches_choices.get(format, None)
            )
            if verbose:
                print 'Wrote figure in %s.' %(filename + '.' + format)
        except IOError:
            warnings.warn('%s is not writeable.' % (filename + '.' + format))
        if format == 'svg':
            plt.gcf().set_size_inches([v / svg_downsize_factor for v in
                                       plt.gcf().get_size_inches()])

html_header = """<HTML>
<HEAD>
   <META NAME="description" CONTENT="COCO/BBOB figures by function">
   <META NAME="keywords" CONTENT="COCO, BBOB">
   <META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=iso-8859-1">
   <TITLE> %s </TITLE>
</HEAD>
<BODY>
<H1> %s
</H1>
"""


def next_dimension_str(s):
    try:
        dim = int(s.strip().strip('_').rstrip('D'))
        return s.replace('%02d' % dim, '%02d' % next_dimension(dim))
    except:
        warnings.warn('next_dimension_str failed on "%s"' % s)
        print(s)
        raise


def next_dimension(dim):
    """next dimension when clicking single function html pages"""
    if dim == 2:
        return 3
    if dim == 3:
        return 5
    if dim == 40:
        return 2
    return 2 * dim


def save_single_functions_html(filename, algname='', extension='svg',
                               add_to_names = ''):
    name = filename.split(os.sep)[-1]
    with open(filename + add_to_names + '.html', 'w') as f:
        header_title = algname + ' ' + name + add_to_names
        f.write(html_header % (header_title.strip().replace(' ', ', '),
            algname))
        if add_to_names.endswith('D'):
            name_for_click = next_dimension_str(add_to_names)
            f.write('<A HREF="%s">\n' % (filename.split(os.sep)[-1] + name_for_click  + '.html'))
        for ifun in range(1, 25):
            f.write('<IMG SRC="' + name + '_f%03d' % (ifun)
                    + add_to_names + '.%s">' % (extension))
        if add_to_names.endswith('D'):
            f.write('"\n</A>\n')
        f.write("\n</BODY>\n</HTML>")


def discretize_limits(limits, smaller_steps_limit=3.1):
    """return new limits with discrete values in k * 10**i with k in [1, 3].

    `limits` has len 2 and the new lower limit is always ``10**-0.2``. 

    if `limits[1] / limits[0] < 10**smaller_steps_limits`, k == 3 is an 
    additional choice.
    """
    ymin, ymax = limits
    ymin=np.max((ymin, 10**-0.2))
    ymax=int(ymax + 1)

    ymax_new = 10**np.ceil(np.log10(ymax)) * (1 + 1e-6)
    if 3. * ymax_new / 10 > ymax and np.log10(ymax / ymin) < smaller_steps_limit:
        ymax_new *= 3. / 10
    ymin_new = 10**np.floor(np.log10(ymin)) / (1 + 1e-6)
    if 11 < 3 and 3 * ymin_new < ymin and np.log10(ymax / ymin) < 1.1:
        ymin_new *= 3

    if ymin_new < 1.1:
        ymin_new = 10**-0.2
    ymin_new = 10**-0.2
    return ymin_new, ymax_new


def marker_positions(xdata, ydata, nbperdecade, maxnb,
                     ax_limits=None, y_transformation=None):
    """return randomized marker positions

    replacement for downsample, could be improved by becoming independent
    of axis limits?
    """
    if ax_limits is None:  # use current axis limits
        ax_limits = plt.axis()
    tfy = y_transformation
    if tfy is None:
        tfy = lambda x: x  # identity

    xdatarange = np.log10(max([max(xdata), ax_limits[0], ax_limits[1]]) + 0.5) - \
                 np.log10(min([min(xdata), ax_limits[0], ax_limits[1]]) + 0.5)  #np.log10(xdata[-1]) - np.log10(xdata[0])
    ydatarange = tfy(max([max(ydata), ax_limits[2], ax_limits[3]]) + 0.5) - \
                 tfy(min([min(ydata), ax_limits[2], ax_limits[3]]) + 0.5)  # tfy(ydata[-1]) - tfy(ydata[0])
    nbmarkers = np.min([maxnb, nbperdecade +
                               np.ceil(nbperdecade * (1e-99 + np.abs(np.log10(max(xdata)) - np.log10(min(xdata)))))])
    probs = np.abs(np.diff(np.log10(xdata))) / xdatarange + \
            np.abs(np.diff(tfy(ydata))) / ydatarange
    xpos = []
    ypos= []
    if sum(probs) > 0:
        xoff = np.random.rand() / nbmarkers
        probs /= sum(probs)
        cum = np.cumsum(probs)
        for xact in np.arange(0, 1, 1./nbmarkers):
            pos = xoff + xact + (1./nbmarkers) * (0.3 + 0.4 * np.random.rand())
            idx = np.abs(cum - pos).argmin()  # index of closest value
            xpos.append(xdata[idx])
            ypos.append(ydata[idx])
    xpos.append(xdata[-1])
    ypos.append(ydata[-1])
    return xpos, ypos


def plotUnifLogXMarkers(x, y, nbperdecade, logscale=False, **kwargs):
    """Proxy plot function: markers are evenly spaced on the log x-scale

    Remark/TODO: should be called plot_with_unif_markers!? Here is where
    the ECDF plot "done in pprldmany" actually happens.

    This method generates plots with markers regularly spaced on the
    x-scale whereas the matplotlib.pyplot.plot function will put markers
    on data points.

    This method outputs a list of three lines.Line2D objects: the first
    with the line style, the second for the markers and the last for the
    label.

    This function only works with monotonous graph.
    """
    res = plt.plot(x, y, **kwargs)  # shouldn't this be done in the calling code?

    if 'marker' in kwargs and len(x) > 0:
        # x2, y2 = downsample(x, y)
        x2, y2 = marker_positions(x, y, nbperdecade, 19, plt.axis(),
                                  np.log10 if logscale else None)
        res2 = plt.plot(x2, y2)
        for i in res2:
            i.update_from(res[0]) # copy all attributes of res
        plt.setp(res2, linestyle='', label='')
        res.extend(res2)

    if 'label' in kwargs:
        res3 = plt.plot([], [], **kwargs)
        for i in res3:
            i.update_from(res[0]) # copy all attributes of res
        res.extend(res3)

    plt.setp(res[0], marker='', label='')
    return res

def consecutiveNumbers(data):
    """Groups a sequence of integers into ranges of consecutive numbers.

    Example::
      >>> import os
      >>> os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
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
    for _k, g in groupby(enumerate(data), lambda (i,x):i-x):
        res.append(list(i for i in map(itemgetter(1), g)))

    return res

def logxticks(limits=[-np.inf, np.inf]):
    """Modify log-scale figure xticks from 10^i to i for values with the
    ``limits`` and (re-)sets the current xlim() thereby turning autoscale
    off (if it was on).
    
    This is to have xticks that are more visible.
    Modifying the x-limits of the figure after calling this method will
    not update the ticks.
    Please make sure the xlabel is changed accordingly.
    """
    _xticks = plt.xticks()
    xlims = plt.xlim()
    newxticks = []
    for j in _xticks[0]:
        if j > limits[0] and j < limits[1]: # tick annotations only within the limits
            newxticks.append('%d' % round(np.log10(j)))
        else:
            newxticks.append('')
    plt.xticks(_xticks[0], newxticks)  # this changes the limits (only in newer versions of mpl?)
    plt.xlim(xlims[0], xlims[1])
    # TODO: check the xlabel is changed accordingly?

def beautify():
    """ Customize a figure by adding a legend, axis label, etc."""
    # TODO: what is this function for?
    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_yscale("log")

    # Grid options
    axisHandle.grid(True)

    _ymin, ymax = plt.ylim()
    plt.ylim(ymin=10**-0.2, ymax=ymax) # Set back the default maximum.

    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(np.log10(i)))
    axisHandle.set_yticklabels(tmp2)
    axisHandle.set_ylabel('log10 of ERT')

def generateData(dataSet, targetFuncValue):
    """Returns an array of results to be plotted.

    1st column is ert, 2nd is  the number of success, 3rd the success
    rate, 4th the sum of the number of function evaluations, and
    finally the median on successful runs.
    """
    it = iter(reversed(dataSet.evals))
    i = it.next()
    prev = np.array([np.nan] * len(i))

    while i[0] <= targetFuncValue:
        prev = i
        try:
            i = it.next()
        except StopIteration:
            break

    data = prev[1:].copy() # keep only the number of function evaluations.
    # was up to rev4997: succ = (np.isnan(data) == False)  # better: ~np.isnan(data)
    succ = np.isfinite(data)
    if succ.any():
        med = toolsstats.prctile(data[succ], 50)[0]
        #Line above was modified at rev 3050 to make sure that we consider only
        #successful trials in the median
    else:
        med = np.nan

    # prepare to compute runlengths / ERT with restarts (AKA SP1)
    data[np.isnan(data)] = dataSet.maxevals[np.isnan(data)]

    res = []
    res.extend(toolsstats.sp(data, issuccessful=succ, allowinf=False))
    res.append(np.mean(data)) #mean(FE)
    res.append(med)

    return np.array(res)

def plot(dsList, _valuesOfInterest=(10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
         isbyinstance=True, kwargs={}):
    """From a DataSetList, plot a graph. Not in use and superseeded by ppfigdim.main!?"""

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
                tmp.evals = i.evals[:, np.r_[idxs]].copy()
                tmp.maxevals = i.maxevals[np.ix_(idx)].copy()
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
                    succ.append(np.append(x, tmp))
                    if tmp[2] < j.nbRuns():
                        displaynumber.append((x, tmp[0], tmp[2]))
                else:
                    unsucc.append(np.append(x, tmp))

        if succ:
            tmp = np.vstack(succ)
            #ERT
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))
            #median
            tmp2 = plt.plot(tmp[:, 0], tmp[:, -1], **kwargs)
            plt.setp(tmp2, linestyle='', marker='+', markersize=30, markeredgewidth=5)
            #, color=colors[i], linestyle='', marker='+', markersize=30, markeredgewidth=5))
            res.extend(tmp2)

        # To have the legend displayed whatever happens with the data.
        tmp = plt.plot([], [], **kwargs)
        plt.setp(tmp, label=' %+d' % (np.log10(valuesOfInterest[i])))
        res.extend(tmp)

        #Only for the last target function value
        if unsucc:
            tmp = np.vstack(unsucc) # tmp[:, 0] needs to be sorted!
            res.extend(plt.plot(tmp[:, 0], tmp[:, 1], **kwargs))

    if displaynumber: # displayed only for the smallest valuesOfInterest
        for j in displaynumber:
            t = plt.text(j[0], j[1]*1.85, "%.0f" % j[2],
                         horizontalalignment="center",
                         verticalalignment="bottom")
            res.append(t)

    return res
