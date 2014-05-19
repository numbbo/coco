#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for computing ERT loss ratio

This module outputs figures and tables showing ERT loss ratios.
Comparisons are based on computing the ratio between an ERT value and a
reference (best) ERT value (or the inverse)

"""

from __future__ import absolute_import

import os
import warnings
import cPickle as pickle
import gzip
from pdb import set_trace
import numpy as np
from matplotlib import pyplot as plt
try:
    from matplotlib.transforms import blended_transform_factory as blend
except ImportError:
    # compatibility matplotlib 0.8
    from matplotlib.transforms import blend_xy_sep_transform as blend
from matplotlib import mlab as mlab

from bbob_pproc import toolsstats, bestalg
from bbob_pproc.pptex import writeFEvals2
from bbob_pproc.ppfig import saveFigure

"""
ERT loss ratio of an algorithm A for comparison to BBOB-best2009. This works
only as comparison to a set of algorithms that reach at least the same
target values. Let f=f_A(EVALS) be the smallest target value such that the
expected running time of algorithm A was smaller than or equal to EVALS.
Let ERT_A=EVALS, if ERT_best(next difficult f) < EVALS and
ERT_A=ERT_A(f_A(EVALS)) otherwise (we have ERT_A(f_A(EVALS)) <= EVALS).
The ERT loss ratio for algorithm A is defined as:
    Loss_A = stat_fcts(exp(CrE_A) * ERT_A / ERT_best(f))

    + where f is a function of EVALS and stat_fcts is the desired statistics
      over the values from all functions (or a subgroup of functions), for
      example the geometric mean, min, max or any quantile. More specific: we
      plot versus 'the budget EVALS' the geometric mean (line) and Box-Whisker
      error bars at EVALS=2*D, 10*D, 100*D,...: a box between 25% and 75% with
      the median as additional symbol, a line with "T" as end-marker between
      10% and 90% (the box covers the line) and a single point for min, max.
      For a function subgroup the Box-Whisker is replaced with the four or five
      actual points with the function number written.
      Caption: ERT loss ratio: expected running time, ERT (measured in number
      of function evaluations), divided by the best ERT seen in BBOB-best2009 for
      the respectively same function and target function value, plotted versus
      number of function evaluations for the functions $f_1$--$f_{24}$ in
      dimension $D=XXX$, corrected by the parameter-crafting-effort
      $\exp(CrE)==YYY$. Line: geometric mean over all functions. Box-Whisker
      error bars: 25-75\%-percentile range with median (box),
      10-90\%-percentile range (line), and minimum and maximum ERT loss ratio
      (points). Alternative Box-Whisker sentence: Points: ERT loss ratio for
      each function
    + The problem: how to find out CrE_A? Possible solution: ask for input in
      the script and put the given number into the caption and put exp(CrE_A)
      as small symbol on the y-axis of the figure for cross-checking.
    + This should make a collection of graphs for all functions and all
      subgroups which gives an additional page in the 'single algorithm'
      template. Respective tables could be side-by-side the graphs.
    + Example for how to read the graph: a loss ratio of 4 for ERT=20D means,
      that the function value reached with ERT=20D could be reached with the
      respective best algorithm in ERT_best=5D function evaluations on average.
      Therefore, given a budget of 20*D function evaluations, the best
      algorithm could have further improved the function value using the
      remaining 15*D ($75\%=1-1/4$) function evaluations.
      
Details: if ERT_A = ERT_A(f_A(EVALS)) always, the x-axis of plots between 
different algorithms becomes incomparable. Also could ERT_A < ERT_best, 
even though ERT_best reaches a better f-value for the given EVALS. 

"""

"""OLD STUFF:
ERT loss ratio: expected running time, ERT (measured in number
      of function evaluations), divided by the best ERT seen in BBOB-best2009 for
      the respectively same function and target function value, plotted versus
      number of function evaluations for the functions $f_1$--$f_{24}$ in
      dimension $D=XXX$, corrected by the parameter-crafting-effort
      $\exp(CrE)==YYY$. Line: geometric mean over all functions. Box-Whisker
      error bars: 25-75\%-percentile range with median (box),
      10-90\%-percentile range (line), and minimum and maximum ERT loss ratio
      (points). 
Table: 
\ERT\ loss ratio (see also Figure~\ref{fig:ERTgraphs}) vs.\ a given budget
$\FEvals$. Each cross ({\color{blue}$+$}) represents a single function. The
target value \ftarget\ used for a given \FEvals\ is the smallest (best) recorded
function value such that $\ERT(\ftarget)\le\FEvals$ for the presented algorithm.
Shown is \FEvals\ divided by the respective best $\ERT(\ftarget)$ from BBOB-2009
for functions $f_1$--$f_{24}$ in 5-D and 20-D. Line: geometric mean. Box-Whisker
error bar: 25-75\%-ile with median (box), 10-90\%-ile (caps), and minimum and
maximum \ERT\ loss ratio (points). The vertical line gives the maximal number of
function evaluations in a single trial in this function subset.

\ERT\ loss ratio. The ERT of the considered algorithm, the budget, is shown in
the first column. For the loss ratio the budget is divided by the ERT for the
respective best result from BBOB-2009 (see also Table~\ref{tab:ERTloss}).
The last row $\text{RL}_{\text{US}}/\text{D}$ gives the number of function
evaluations in unsuccessful runs divided by dimension. Shown are the smallest,
10\%-ile, 25\%-ile, 50\%-ile, 75\%-ile and 90\%-ile value (smaller values are
better). The ERT Loss ratio equals to one for the respective best algorithm from
BBOB-2009. Typical median values are between ten and hundred.

\ERT\ loss ratio. The ERT of the considered algorithm, the budget, is shown in the first column. 
For the loss ratio the budget is divided by the ERT for the respective best result from BBOB-2009 (see also Figure~\ref{fig:ERTlogloss}). The last row $\text{RL}_{\text{US}}/\text{D}$ gives the number of function evaluations in 
unsuccessful runs divided by dimension. Shown are the smallest, 10\%-ile, 25\%-ile, 
50\%-ile, 75\%-ile and 90\%-ile value (smaller values are better).
The ERT Loss ratio equals to one for the respective best algorithm from BBOB-2009.
Typical median values are between ten and hundred. 

such that $\ERT(\ftarget)\le\FEvals$ for the 
    Shown is \FEvals\ divided by the respective best $\ERT(\ftarget)$ from BBOB-2009 
    %
    for functions $f_1$--$f_{24}$ in 5-D and 20-D. 
    %
    % Each \ERT\ is multiplied by $\exp(\CrE)$ correcting for the parameter crafting effort. 
"""

table_caption = r"""% \bbobloglosstablecaption{}
    \ERT\ loss ratio versus the budget (both in number of $f$-evaluations
    divided by dimension). The target value \ftarget\ for a given budget
    \FEvals\ is the best target $f$-value reached within the budget by the given
    algorithm. Shown is the \ERT\ of the given algorithm divided by best \ERT\
    seen in GECCO-BBOB-2009 for the target \ftarget, or, if the best algorithm
    reached a better target within the budget, the budget divided by the best
    \ERT. Line: geometric mean. Box-Whisker error bar: 25-75\%-ile with median
    (box), 10-90\%-ile (caps), and minimum and maximum \ERT\ loss ratio
    (points). The vertical line gives the maximal number of function evaluations
    in a single trial in this function subset. See also
    Figure~\ref{fig:ERTlogloss} for results on each function subgroup.
    """
figure_caption = r"""%
    \ERT\ loss ratios (see Figure~\ref{tab:ERTloss} for details).  
    Each cross ({\color{blue}$+$}) represents a single function, the line is the geometric mean.
    """  # \bbobloglossfigurecaption{}

evalf = None
f_thresh = 1.e-8
whiskerscolor = 'b'
boxescolor = 'b'
medianscolor = 'r'
capscolor = 'k'
flierscolor = 'b'

def detERT(entry, funvals):
    # could be more efficient given that funvals is sorted...
    res = []
    for f in funvals:
        idx = (entry.target<=f)
        try:
            res.append(entry.ert[idx][0])
        except IndexError:
            res.append(np.inf)
    return res

def detf(entry, evals):
    """Determines a function value given a number of evaluations.

    Let A be the algorithm considered. Let f=f_A(evals) be the smallest
    target value such that the expected running time of algorithm A was
    smaller than or equal to evals.

    :keyword DataSet entry: data set
    :keyword list evals: numbers of function evaluations considered

    :Returns: list of the target function values

    """
    res = []
    for fevals in evals:
        tmp = (entry.ert <= fevals)
        #set_trace()
        #if len(entry.target[tmp]) == 0:
            #set_trace()
        idx = np.argmin(entry.target[tmp])
        res.append(max(entry.target[idx], f_thresh))
        #res2.append(entry.ert[dix])
        #TODO np.min(empty)
    return res

def generateData(dsList, evals, CrE_A):
    res = {}

    D = set(i.dim for i in dsList).pop() # should have only one element
    #if D == 3:
       #set_trace()
    if not bestalg.bestalgentries2009:
        bestalg.loadBBOB2009()

    for fun, tmpdsList in dsList.dictByFunc().iteritems():
        assert len(tmpdsList) == 1
        entry = tmpdsList[0]

        bestalgentry = bestalg.bestalgentries2009[(D, fun)]

        #ERT_A
        f_A = detf(entry, evals)

        ERT_best = detERT(bestalgentry, f_A)
        ERT_A = detERT(entry, f_A)
        nextbestf = []
        for i in f_A:
            if i == 0.:
                nextbestf.append(0.)
            else:
                tmp = bestalgentry.target[bestalgentry.target < i]
                try:
                    nextbestf.append(tmp[0])
                except IndexError:
                    nextbestf.append(i * 10.**(-0.2)) # TODO: this is a hack

        ERT_best_nextbestf = detERT(bestalgentry, nextbestf)

        for i in range(len(ERT_A)):
            # nextbestf[i] >= f_thresh: this is tested because if it is not true
            # ERT_best_nextbestf[i] is supposed to be infinite.
            if nextbestf[i] >= f_thresh and ERT_best_nextbestf[i] < evals[i]: # is different from the specification...
                ERT_A[i] = evals[i]

        # For test purpose:
        #if fun % 10 == 0:
        #    ERT_A[-2] = 1.
        #    ERT_best[-2] = np.inf
        ERT_A = np.array(ERT_A)
        ERT_best = np.array(ERT_best)
        loss_A = np.exp(CrE_A) * ERT_A / ERT_best
        assert (np.isnan(loss_A) == False).all()
        #set_trace()
        #if np.isnan(loss_A).any() or np.isinf(loss_A).any() or (loss_A == 0.).any():
        #    txt = 'Problem with entry %s' % str(entry)
        #    warnings.warn(txt)
        #    #set_trace()
        res[fun] = loss_A

    return res

def boxplot(x, notch=0, sym='b+', positions=None, widths=None):
    """Makes a box and whisker plot.

    Adapted from matplotlib.axes 0.98.5.2
    Modified such that the caps are set to the 10th and 90th
    percentiles, and to have some control on the colors.

    call signature::

      boxplot(x, notch=0, sym='+', positions=None, widths=None)

    Make a box and whisker plot for each column of *x* or each
    vector in sequence *x*.  The box extends from the lower to
    upper quartile values of the data, with a line at the median.
    The whiskers extend from the box to show the range of the
    data.  Flier points are those past the end of the whiskers.

    - *notch* = 0 (default) produces a rectangular box plot.
    - *notch* = 1 will produce a notched box plot

    *sym* (default 'b+') is the default symbol for flier points.
    Enter an empty string ('') if you don't want to show fliers.

    *whis* (default 1.5) defines the length of the whiskers as
    a function of the inner quartile range.  They extend to the
    most extreme data point within ( ``whis*(75%-25%)`` ) data range.

    *positions* (default 1,2,...,n) sets the horizontal positions of
    the boxes. The ticks and limits are automatically set to match
    the positions.

    *widths* is either a scalar or a vector and sets the width of
    each box. The default is 0.5, or ``0.15*(distance between extreme
    positions)`` if that is smaller.

    *x* is an array or a sequence of vectors.

    Returns a dictionary mapping each component of the boxplot
    to a list of the :class:`matplotlib.lines.Line2D`
    instances created.
    
    Copyright (c) 2002-2009 John D. Hunter; All Rights Reserved
    """
    whiskers, caps, boxes, medians, fliers = [], [], [], [], []

    # convert x to a list of vectors
    if hasattr(x, 'shape'):
        if len(x.shape) == 1:
            if hasattr(x[0], 'shape'):
                x = list(x)
            else:
                x = [x,]
        elif len(x.shape) == 2:
            nr, nc = x.shape
            if nr == 1:
                x = [x]
            elif nc == 1:
                x = [x.ravel()]
            else:
                x = [x[:,i] for i in xrange(nc)]
        else:
            raise ValueError, "input x can have no more than 2 dimensions"
    if not hasattr(x[0], '__len__'):
        x = [x]
    col = len(x)

    # get some plot info
    if positions is None:
        positions = range(1, col + 1)
    if widths is None:
        distance = max(positions) - min(positions)
        widths = min(0.15*max(distance,1.0), 0.5)
    if isinstance(widths, float) or isinstance(widths, int):
        widths = np.ones((col,), float) * widths

    # loop through columns, adding each to plot
    for i,pos in enumerate(positions):
        d = np.ravel(x[i])
        row = len(d)
        # get median and quartiles
        wisk_lo, q1, med, q3, wisk_hi = mlab.prctile(d,[10,25,50,75,90])
        # get high extreme
        #iq = q3 - q1
        #hi_val = q3 + whis*iq
        #wisk_hi = np.compress( d <= hi_val , d )
        #if len(wisk_hi) == 0:
            #wisk_hi = q3
        #else:
            #wisk_hi = max(wisk_hi)
        ## get low extreme
        #lo_val = q1 - whis*iq
        #wisk_lo = np.compress( d >= lo_val, d )
        #if len(wisk_lo) == 0:
            #wisk_lo = q1
        #else:
            #wisk_lo = min(wisk_lo)
        # get fliers - if we are showing them
        flier_hi = []
        flier_lo = []
        flier_hi_x = []
        flier_lo_x = []
        if len(sym) != 0:
            flier_hi = np.compress( d > wisk_hi, d )
            flier_lo = np.compress( d < wisk_lo, d )
            flier_hi_x = np.ones(flier_hi.shape[0]) * pos
            flier_lo_x = np.ones(flier_lo.shape[0]) * pos

        # get x locations for fliers, whisker, whisker cap and box sides
        box_x_min = pos - widths[i] * 0.5
        box_x_max = pos + widths[i] * 0.5

        wisk_x = np.ones(2) * pos

        cap_x_min = pos - widths[i] * 0.25
        cap_x_max = pos + widths[i] * 0.25
        cap_x = [cap_x_min, cap_x_max]

        # get y location for median
        med_y = [med, med]

        # calculate 'regular' plot
        if notch == 0:
            # make our box vectors
            box_x = [box_x_min, box_x_max, box_x_max, box_x_min, box_x_min ]
            box_y = [q1, q1, q3, q3, q1 ]
            # make our median line vectors
            med_x = [box_x_min, box_x_max]
        # calculate 'notch' plot
        else:
            raise NotImplementedError
            notch_max = med #+ 1.57*iq/np.sqrt(row)
            notch_min = med #- 1.57*iq/np.sqrt(row)
            if notch_max > q3:
                notch_max = q3
            if notch_min < q1:
                notch_min = q1
            # make our notched box vectors
            box_x = [box_x_min, box_x_max, box_x_max, cap_x_max, box_x_max,
                     box_x_max, box_x_min, box_x_min, cap_x_min, box_x_min,
                     box_x_min ]
            box_y = [q1, q1, notch_min, med, notch_max, q3, q3, notch_max,
                     med, notch_min, q1]
            # make our median line vectors
            med_x = [cap_x_min, cap_x_max]
            med_y = [med, med]

        doplot = plt.plot
        whiskers.extend(doplot(wisk_x, [q1, wisk_lo], color=whiskerscolor, linestyle='--'))
        whiskers.extend(doplot(wisk_x, [q3, wisk_hi], color=whiskerscolor, linestyle='--'))
        caps.extend(doplot(cap_x, [wisk_hi, wisk_hi], color=capscolor, linestyle='-'))
        caps.extend(doplot(cap_x, [wisk_lo, wisk_lo], color=capscolor, linestyle='-'))
        boxes.extend(doplot(box_x, box_y, color=boxescolor, linestyle='-'))
        medians.extend(doplot(med_x, med_y, color=medianscolor, linestyle='-'))
        fliers.extend(doplot(flier_hi_x, flier_hi, sym,
                             flier_lo_x, flier_lo, sym))

    # fix our axes/ticks up a little
    newlimits = min(positions)-0.5, max(positions)+0.5
    plt.gca().set_xlim(newlimits)
    plt.gca().set_xticks(positions)

    return dict(whiskers=whiskers, caps=caps, boxes=boxes,
                medians=medians, fliers=fliers)

def plot(xdata, ydata):
    """Plot the ERT log loss figures.

    Two cases: box-whisker plot is used for representing the data of all
    functions, otherwise all data is represented using crosses.

    """
    res = []

    tmp = list(10**np.mean(i[np.isfinite(i)]) for i in ydata)
    res.extend(plt.plot(xdata, tmp, ls='-', color='k', lw=3, #marker='+',
                        markersize=20, markeredgewidth=3))

    if max(len(i) for i in ydata) < 20: # TODO: subgroups of function, hopefully.
        for i, y in enumerate(ydata):
            # plot all single data points
            if (np.isfinite(y)==False).any():
                assert not (np.isinf(y) * y > 0.).any()
                assert not np.isnan(y).any()

                ax = plt.gca()
                trans = blend(ax.transData, ax.transAxes)
                res.extend(plt.plot((xdata[i], ), (0., ),
                                    marker='+', color=flierscolor,
                                    ls='', markersize=20, markeredgewidth=3,
                                    transform=trans, clip_on=False))
                res.append(plt.text(xdata[i], 0.02, '%d' % len(y[np.isinf(y)]),
                                    transform=trans, horizontalalignment='left',
                                    verticalalignment='bottom'))
                y = y[np.isfinite(y)]
                if len(y) == 0:
                    continue

            res.extend(plt.plot([xdata[i]]*len(y), 10**np.array(y),
                                marker='+', color=flierscolor,
                                ls='', markersize=20, markeredgewidth=3))

            # plot dashed vertical line between min and max 
            plt.plot([xdata[i]]*2, 10**np.array([min(y), max(y)]),
                                color='k',  # marker='+', 
                                ls='--', linewidth=2) #, markersize=20, markeredgewidth=3)
            # plot min and max with different symbol
            #plt.plot([xdata[i]], 10**min(np.array(y)),
            #                    marker='+', color='k',
            #                    ls='', markersize=20, markeredgewidth=3)
            #plt.plot([xdata[i]], 10**max(np.array(y)),
            #                    marker='+', color='k',
            #                    ls='', markersize=20, markeredgewidth=3)
    else:
        for i, y in enumerate(ydata):
            # plot all single data points
            if (np.isfinite(y)==False).any():
                assert not (np.isinf(y) * y > 0.).any()
                assert not np.isnan(y).any()

                ax = plt.gca()
                trans = blend(ax.transData, ax.transAxes)
                res.extend(plt.plot((xdata[i], ), (0, ),
                                    marker='.', color='k',
                                    ls='', markersize=20, markeredgewidth=3,
                                    transform=trans, clip_on=False))
                res.append(plt.text(xdata[i], 0.02, '%d' % len(y[np.isinf(y)]),
                                    transform=trans, horizontalalignment='left',
                                    verticalalignment='bottom'))
                y = y[np.isfinite(y)]

        dictboxwhisker = boxplot(list(10**np.array(i) for i in ydata),
                                 sym='', notch=0, widths=None,
                                 positions=xdata)
        #'medians', 'fliers', 'whiskers', 'boxes', 'caps'
        plt.setp(dictboxwhisker['medians'], lw=3)
        plt.setp(dictboxwhisker['boxes'], lw=3)
        plt.setp(dictboxwhisker['caps'], lw=3)
        plt.setp(dictboxwhisker['whiskers'], lw=3)
        for i in dictboxwhisker.values():
            res.extend(i)
        res.extend(plt.plot(xdata, list(10**min(i) for i in ydata), marker='.',
                            markersize=20, color='k', ls=''))
        res.extend(plt.plot(xdata, list(10**max(i) for i in ydata), marker='.',
                             markersize=20, color='k', ls=''))

    return res

def beautify():
    """Format the figure."""

    a = plt.gca()
    a.set_yscale('log')
    ymin = 1e-2
    ymax = 1e4
    plt.ylim(ymin=ymin, ymax=ymax)
    ydata = np.power(10., np.arange(np.log10(ymin), np.log10(ymax)+1))
    yticklabels = list(str(i) for i in range(int(np.log10(ymin)),
                                             int(np.log10(ymax)+1)))
    plt.yticks(ydata, yticklabels)

    plt.xlabel('log10 of FEvals / dimension')
    plt.ylabel('log10 of ERT loss ratio')
    #a.yaxis.grid(True, which='minor')
    a.yaxis.grid(True, which='major')

def generateTable(dsList, CrE=0., outputdir='.', info='default', verbose=True):
    """Generates ERT loss ratio tables.

    :param DataSetList dsList: input data set
    :param float CrE: crafting effort (see COCO documentation)
    :param string outputdir: output folder (must exist)
    :param string info: string suffix for output file names
    :param bool verbose: controls verbosity

    """

    #Set variables
    prcOfInterest = [0, 10, 25, 50, 75, 90]
    for d, dsdim in dsList.dictByDim().iteritems():
        res = []
        maxevals = []
        funcs = []
        mFE = []

        for i in dsdim:
            maxevals.append(max(i.ert[np.isinf(i.ert)==False]))
            funcs.append(i.funcId)
            mFE.append(max(i.maxevals))

        maxevals = max(maxevals)
        mFE = max(mFE)
        EVALS = [2.*d]
        EVALS.extend(10.**(np.arange(1, np.log10(1e-9 + maxevals * 1./d))) * d)
        #Set variables: Done
        data = generateData(dsList, EVALS, CrE)
    
        tmp = "\\textbf{\\textit{f}\\raisebox{-0.35ex}{%d}--\\textit{f}\\raisebox{-0.35ex}{%d} in %d-D}, maxFE/D=%s" \
            % (min(funcs), max(funcs), d, writeFEvals2(int(mFE/d), maxdigits=6))
    
        res.append(r" & \multicolumn{" + str(len(prcOfInterest)) + "}{|c}{" + tmp + "}")
    
        header = ["\\#FEs/D"]
        for i in prcOfInterest:
            if i == 0:
                tmp = "best"
            elif i == 50:
                tmp = "\\textbf{med}"
            else:
                tmp = "%d\\%%" % i
            header.append(tmp)
    
        #set_trace()
        res.append(" & ".join(header))
        for i in range(len(EVALS)):
            tmpdata = list(data[f][i] for f in data)
            #set_trace()
            tmpdata = toolsstats.prctile(tmpdata, prcOfInterest)
            # format entries
            #tmp = [writeFEvals(EVALS[i]/d, '.0')]
            if EVALS[i]/d < 200:
                tmp = [writeFEvals2(EVALS[i]/d, 3)]
            else:
                tmp = [writeFEvals2(EVALS[i]/d, 1)]
            for j in tmpdata:
                # tmp.append(writeFEvals(j, '.2'))
                # tmp.append(writeFEvals2(j, 2))
                if j == 0.:
                    tmp.append("~\\,0")
                elif j < 1:
                    tmp.append("~\\,%1.2f" % j)
                elif j < 10:
                    tmp.append("\\hspace*{1ex}%1.1f" % j)
                elif j < 100:
                    tmp.append("%2.0f" % j)
                else:
                    ar = ("%1.1e" % j).split('e')
                    tmp.append(ar[0] + 'e' + str(int(ar[1])))
                # print tmp[-1]
            res.append(" & ".join(tmp))
    
        # add last line: runlength distribution for which 1e-8 was not reached.
        tmp = [r"$\text{RL}_{\text{US}}$/D"]
        tmpdata = []
        for i in dsList:
            it = reversed(i.evals)
            curline = None
            nextline = it.next()
            while nextline[0] <= f_thresh:
                curline = nextline[1:]
                nextline = it.next()
            if curline is None:
                tmpdata.extend(i.maxevals)
            else:
                tmpdata.extend(i.maxevals[np.isnan(curline)])
    
        #set_trace()
        if tmpdata: # if it is not empty
            tmpdata = toolsstats.prctile(tmpdata, prcOfInterest)
            for j in tmpdata:
               tmp.append(writeFEvals2(j/d, 1))
            res.append(" & ".join(tmp))
    
        res = (r"\\"+ "\n").join(res)
        res = r"\begin{tabular}{c|" + len(prcOfInterest) * "l" +"}\n" + res
        #res = r"\begin{tabular}{ccccc}" + "\n" + res
        res = res + "\n" + r"\end{tabular}" + "\n"
    
        filename = os.path.join(outputdir, 'pploglosstable_%02dD_%s.tex' % (d, info))
        f = open(filename, 'w')
        f.write(res)
        f.close()
        if verbose:
            print "Wrote ERT loss ratio table in %s." % filename

def generateFigure(dsList, CrE=0., isStoringXRange=True, outputdir='.',
                   info='default', verbose=True):
    """Generates ERT loss ratio figures.

    :param DataSetList dsList: input data set
    :param float CrE: crafting effort (see COCO documentation)
    :param bool isStoringXRange: if set to True, the first call to this
                                 function sets the global
                                 :py:data:`evalf` and all subsequent
                                 calls will use this value as boundaries
                                 in the generated figures.
    :param string outputdir: output folder (must exist)
    :param string info: string suffix for output file names
    :param bool verbose: controls verbosity

    """

    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    if isStoringXRange:
        global evalf
    else:
        evalf = None

    # do not aggregate over dimensions
    for d, dsdim in dsList.dictByDim().iteritems():
        maxevals = max(max(i.ert[np.isinf(i.ert)==False]) for i in dsdim)
        EVALS = [2.*d]
        EVALS.extend(10.**(np.arange(1, np.ceil(1e-9 + np.log10(maxevals * 1./d))))*d)
        if not evalf:
            evalf = (np.log10(EVALS[0]/d), np.log10(EVALS[-1]/d))
    
        data = generateData(dsdim, EVALS, CrE)
        ydata = []
        for i in range(len(EVALS)):
            #Aggregate over functions.
            ydata.append(np.log10(list(data[f][i] for f in data)))
    
        xdata = np.log10(np.array(EVALS)/d)
        xticklabels = ['']
        xticklabels.extend('%d' % i for i in xdata[1:])
        plot(xdata, ydata)
    
        filename = os.path.join(outputdir, 'pplogloss_%02dD_%s' % (d, info))
        plt.xticks(xdata, xticklabels)
        #Is there an upper bound?
    
        if CrE > 0 and len(set(dsdim.dictByFunc().keys())) >= 20:
            #TODO: hopefully this means we are not considering function groups.
            plt.text(0.01, 0.98, 'CrE = %5g' % CrE, fontsize=20,
                     horizontalalignment='left', verticalalignment='top',
                     transform = plt.gca().transAxes,
                     bbox=dict(facecolor='w'))
    
        plt.axhline(1., color='k', ls='-', zorder=-1)
        plt.axvline(x=np.log10(max(i.mMaxEvals()/d for i in dsdim)), color='k')
        funcs = set(i.funcId for i in dsdim)
        if len(funcs) > 1:
            text = 'f%d-%d' %(min(funcs), max(funcs))
        else:
            text = 'f%d' %(funcs.pop())
        plt.text(0.5, 0.93, text, horizontalalignment="center",
                 transform=plt.gca().transAxes)
        beautify()
        if evalf:
            plt.xlim(xmin=evalf[0]-0.5, xmax=evalf[1]+0.5)

        saveFigure(filename, verbose=verbose)
    
        #plt.show()
        plt.close()
    
        #plt.rcdefaults()

def main(dsList, CrE=0., isStoringXRange=True, outputdir='.', info='default',
         verbose=True):
    """Generates ERT loss ratio boxplot figures.
    
    Calls method generateFigure.

    """
    generateFigure(dsList, CrE, isStoringXRange, outputdir, info, verbose)
