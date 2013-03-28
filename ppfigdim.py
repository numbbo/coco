#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate performance scaling figures.

The figures show the scaling of the performance in terms of ERT w.r.t.
dimensionality on a log-log scale. On the y-axis, data is represented as
a number of function evaluations divided by dimension, this is in order
to compare at a glance with a linear scaling for which ERT is
proportional to the dimension and would therefore be represented by a
horizontal line in the figure.

Crosses (+) give the median number of function evaluations of successful
trials divided by dimension for the smallest *reached* target function
value.
Numbers indicate the number of successful runs for the smallest
*reached* target.
If the smallest target function value (1e-8) is not reached for a given
dimension, crosses (x) give the average number of overall conducted
function evaluations divided by the dimension.

Horizontal lines indicate linear scaling with the dimension, additional
grid lines show quadratic and cubic scaling.
The thick light line with diamond markers shows the single best results
from BBOB-2009 for df = 1e-8.

**Example**

.. plot::
    :width: 50%
    
    import urllib
    import tarfile
    import glob
    from pylab import *
    
    import bbob_pproc as bb
    
    # Collect and unarchive data (3.4MB)
    dataurl = 'http://coco.lri.fr/BBOB2009/pythondata/BIPOP-CMA-ES.tar.gz'
    filename, headers = urllib.urlretrieve(dataurl)
    archivefile = tarfile.open(filename)
    archivefile.extractall()
    
    # Scaling figure
    ds = bb.load(glob.glob('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f002_*.pickle'))
    figure()
    bb.ppfigdim.plot(ds)
    bb.ppfigdim.beautify()
    bb.ppfigdim.plot_previous_algorithms(2) # plot BBOB 2009 best algorithm on fun 2

"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace
from bbob_pproc import genericsettings, toolsstats, bestalg, pproc
from bbob_pproc.ppfig import saveFigure, groupByRange

values_of_interest = pproc.TargetValues((10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8))  # to rename!?
xlim_max = None
ynormalize_by_dimension = True  # not at all tested yet

styles = [  # sort of rainbow style, most difficult (red) first
          {'color': 'r', 'marker': 'o', 'markeredgecolor': 'k', 'markeredgewidth': 2, 'linewidth': 4},
          {'color': 'm', 'marker': '.', 'linewidth': 4},
          {'color': 'y', 'marker': '^', 'markeredgecolor': 'k', 'markeredgewidth': 2, 'linewidth': 4},
          {'color': 'g', 'marker': '.', 'linewidth': 4},
          {'color': 'c', 'marker': 'v', 'markeredgecolor': 'k', 'markeredgewidth': 2, 'linewidth': 4},
          {'color': 'b', 'marker': '.', 'linewidth': 4},
          {'color': 'k', 'marker': 'o', 'markeredgecolor': 'k', 'markeredgewidth': 2, 'linewidth': 4},
        ] 

refcolor = 'wheat'

caption_part_one = r"""%
    Expected number of $f$-evaluations (\ERT, lines) to reach $\fopt+\Df$;
    median number of $f$-evaluations (+) to reach the most difficult
    target that was reached not always but at least once; maximum number of
    $f$-evaluations in any trial ({\color{red}$\times$}); """ + (r"""interquartile 
    range with median (notched boxes) of simulated runlengths
    to reach $\fopt+\Df$;""" if genericsettings.scaling_figures_with_boxes 
    else "") + """ all values are """ + ("""divided by dimension and """ if ynormalize_by_dimension else "") + """plotted as 
    $\log_{10}$ values versus dimension. %
    """
    
#""" .replace('REPLACE_THIS', r"interquartile range with median (notched boxes) of simulated runlengths to reach $\fopt+\Df$;" 
#                if genericsettings.scaling_figures_with_boxes else '')
#    # r"(the exponent is given in the legend of #1). " + 
#    "For each function and dimension, $\\ERT(\\Df)$ equals to $\\nbFEs(\\Df)$ " +
#    "divided by the number of successful trials, where a trial is " +
#    "successful if $\\fopt+\\Df$ was surpassed. The " +
#    "$\\nbFEs(\\Df)$ are the total number (the sum) of $f$-evaluations while " +
#    "$\\fopt+\\Df$ was not surpassed in a trial, from all " +  
#    "(successful and unsuccessful) trials, and \\fopt\\ is the optimal " +
#    "function value.  " +
scaling_figure_caption_fixed = caption_part_one + r"""%
    Shown are $\Df = 10^{\{values_of_interest\}}$.  
    Numbers above \ERT-symbols (if appearing) indicate the number of trials reaching the respective target. 
    The light thick line with diamonds indicates the respective best result from BBOB-2009 for $\Df=10^{-8}$. 
    Horizontal lines mean linear scaling, slanted grid lines depict quadratic scaling.  
    """
scaling_figure_caption_rlbased = caption_part_one + r"""%
    Shown is the \ERT\ for 
    targets just not reached by
%    the largest $\Df$-values $\ge10^{-8}$ for which the \ERT\ of 
    the GECCO-BBOB-2009 best algorithm  
    within the given budget $k\DIM$, where $k$ is shown in the legend.
%    was above $\{values_of_interest\}\times\DIM$ evaluations. 
    Numbers above \ERT-symbols indicate the number of trials reaching the respective target.  
    Slanted grid lines indicate a scaling with ${\cal O}(\DIM)$ compared to ${\cal O}(1)$  
    when using the respective 2009 best algorithm. 
    """
    # r"Shown is the \ERT\ for the smallest $\Df$-values $\ge10^{-8}$ for which the \ERT\ of the GECCO-BBOB-2009 best algorithm " + 
    # r"was below $10^{\{values_of_interest\}}\times\DIM$ evaluations. " + 

def scaling_figure_caption():
    if genericsettings.runlength_based_targets:
        return scaling_figure_caption_rlbased.replace('values_of_interest', 
                                        ', '.join(values_of_interest.labels()))
    else:
        return scaling_figure_caption_fixed.replace('values_of_interest', 
                                        ', '.join(values_of_interest.loglabels()))

# should correspond with the colors in pprldistr.
dimensions = genericsettings.dimensions_to_display
functions_with_legend = (1, 24, 101, 130)

# Get benchmark short infos.
funInfos = {}
isBenchmarkinfosFound = True
infofile = os.path.join(os.path.split(__file__)[0], 'benchmarkshortinfos.txt')

try:
    f = open(infofile, 'r')
    for line in f:
        if len(line) == 0 or line.startswith('%') or line.isspace() :
            continue
        funcId, funcInfo = line[0:-1].split(None, 1)
        funInfos[int(funcId)] = funcId + ' ' + funcInfo
    f.close()
except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    isBenchmarkinfosFound = False
    print 'Could not find file', infofile, \
          'Titles in figures will not be displayed.'

def beautify(axesLabel=True):
    """Customize figure presentation.
    
    Uses information from :file:`benchmarkshortinfos.txt` for figure
    title. 
    
    """

    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_xscale("log")
    axisHandle.set_yscale("log")

    # Grid options
    axisHandle.xaxis.grid(False, which='major')
    # axisHandle.grid(True, which='major')
    axisHandle.grid(False, which='minor')
    # axisHandle.xaxis.grid(True, linewidth=0, which='major')
    ymin, ymax = plt.ylim()

    if isinstance(values_of_interest, pproc.RunlengthBasedTargetValues):
        axisHandle.yaxis.grid(False, which='major')
        expon = values_of_interest.times_dimension - ynormalize_by_dimension
        for (i, y) in enumerate(reversed(values_of_interest.run_lengths)):
            plt.plot((1, 200), [y, y], 'k:', linewidth=0.2)
            if i / 2. == i // 2:
                plt.plot((1, 200), [y, y * 200**expon], styles[i]['color'] + '-', linewidth=0.2)
    else:
        axisHandle.yaxis.grid(True, which='major')
    # quadratic slanted "grid"
    for i in xrange(-2, 7, 1 if ymax <= 1e3 else 2):
        plt.plot((0.2, 200), (10**i, 10**(i + 3)), 'k:', linewidth=0.5)  # TODO: this should be done before the real lines are plotted? 

    # for x in dimensions:
    #     plt.plot(2 * [x], [0.1, 1e11], 'k:', linewidth=0.5)
    # ticks on axes
    # axisHandle.invert_xaxis()
    dimticklist = dimensions 
    dimannlist = dimensions 
    # TODO: All these should depend on one given input (xlim, ylim)

    axisHandle.set_xticks(dimticklist)
    axisHandle.set_xticklabels([str(n) for n in dimannlist])
    logyend = 11  # int(1 + np.log10(plt.ylim()[1]))
    axisHandle.set_yticks([10.**i for i in xrange(0, logyend)])
    axisHandle.set_yticklabels(range(0, logyend))
    if 11 < 3:
        tmp = axisHandle.get_yticks()
        tmp2 = []
        for i in tmp:
            tmp2.append('%d' % round(np.log10(i)))
        axisHandle.set_yticklabels(tmp2)
    if 11 < 3:
        # ticklabels = 10**np.arange(int(np.log10(plt.ylim()[0])), int(np.log10(1 + plt.ylim()[1])))
        ticks = []
        for i in xrange(int(np.log10(plt.ylim()[0])), int(np.log10(1 + plt.ylim()[1]))):
            ticks += [10 ** i, 2 * 10 ** i, 5 * 10 ** i]
        axisHandle.set_yticks(ticks)
        # axisHandle.set_yticklabels(ticklabels)
    # axes limites
    plt.xlim(0.9 * dimensions[0], 1.125 * dimensions[-1]) 
    plt.ylim(ymin=np.min((ymin, 10**-0.2)), ymax=int(ymax + 1))  # Set back the default maximum.
    if xlim_max is not None:
        if isinstance(values_of_interest, pproc.RunlengthBasedTargetValues):
            plt.ylim(0.3, xlim_max)  # set in config 
        else:
            pass  # TODO: xlim_max seems to be not None even when not desired
            # plt.ylim(1, xlim_max)
        if 11 < 3:
            title = plt.gca().get_title()  # works not not as expected
            if title.startswith('1 ') or title.startswith('5 '):
                plt.ylim(0.5, 1e2)
            if title.startswith('19 ') or title.startswith('20 '):
                plt.ylim(0.5, 1e4)


    if axesLabel:
        plt.xlabel('Dimension')
        if ynormalize_by_dimension:
            plt.ylabel('Run Lengths / Dimension')
        else:
            plt.ylabel('Run Lengths')
            

def generateData(dataSet, targetFuncValue):
    """Computes an array of results to be plotted.
    
    :returns: (ert, success rate, number of success, total number of
               function evaluations, median of successful runs).

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

    data = prev[1:].copy()  # keep only the number of function evaluations.
    succ = (np.isnan(data) == False)
    if succ.any():
        med = toolsstats.prctile(data[succ], 50)[0]
        # Line above was modified at rev 3050 to make sure that we consider only
        # successful trials in the median
    else:
        med = np.nan

    data[np.isnan(data)] = dataSet.maxevals[np.isnan(data)]

    res = []
    res.extend(toolsstats.sp(data, issuccessful=succ, allowinf=False))
    res.append(np.mean(data))  # mean(FE)
    res.append(med)

    return np.array(res)  


def plot(dsList, valuesOfInterest=values_of_interest, styles=styles):
    """From a DataSetList, plot a figure of ERT/dim vs dim.
    
    There will be one set of graphs per function represented in the
    input data sets. Most usually the data sets of different functions
    will be represented separately.
    
    :param DataSetList dsList: data sets
    :param seq valuesOfInterest: 
        target precisions via class TargetValues, there might 
        be as many graphs as there are elements in
        this input. Can be different for each
        function (a dictionary indexed by ifun). 
    
    :returns: handles

    """
    styles = list(reversed(styles[:len(valuesOfInterest)]))
    dictFunc = dsList.dictByFunc()
    res = []

    for func in dictFunc:
        dictFunc[func] = dictFunc[func].dictByDim()
        dimensions = sorted(dictFunc[func])

        # legend = []
        line = []
        mediandata = {}
        displaynumber = {}
        for i_target in range(len(valuesOfInterest)):
            succ = []
            unsucc = []
            # data = []
            maxevals = np.ones(len(dimensions))
            maxevals_succ = np.ones(len(dimensions)) 
            # Collect data that have the same function and different dimension.
            for idim, dim in enumerate(dimensions):
                assert len(dictFunc[func][dim]) == 1
                # (ert, success rate, number of success, total number of
                #        function evaluations, median of successful runs)
                tmp = generateData(dictFunc[func][dim][0], valuesOfInterest((func, dim))[i_target])
                maxevals[idim] = max(dictFunc[func][dim][0].maxevals)
                # data.append(np.append(dim, tmp))
                if tmp[2] > 0:  # Number of success is larger than 0
                    succ.append(np.append(dim, tmp))
                    if tmp[2] < dictFunc[func][dim][0].nbRuns():
                        displaynumber[dim] = ((dim, tmp[0], tmp[2]))
                    mediandata[dim] = (i_target, tmp[-1])
                    unsucc.append(np.append(dim, np.nan))
                else:
                    unsucc.append(np.append(dim, tmp[-2]))  # total number of fevals

            if len(succ) > 0:
                tmp = np.vstack(succ)
                # ERT
                if genericsettings.scaling_figures_with_boxes:
                    for dim in dimensions: 
                        # to find finite simulated runlengths we need to have at least one successful run
                        if dictFunc[func][dim][0].detSuccesses([valuesOfInterest((func, dim))[i_target]])[0]:
                            # make a box-plot
                            y = toolsstats.drawSP_from_dataset(
                                                dictFunc[func][dim][0],
                                                valuesOfInterest((func, dim))[i_target],
                                                [25, 50, 75], 
                                                genericsettings.simulated_runlength_bootstrap_sample_size)[0]
                            rec_width = 1.1 # box ("rectangle") width
                            rec_taille_fac = 0.3  # notch width parameter
                            r = rec_width ** ((1. + i_target / 3.) / 4)  # more difficult targets get a wider box
                            styles2 = {}
                            for s in styles[i_target]:
                                styles2[s] = styles[i_target][s]
                            styles2['linewidth'] = 1
                            styles2['markeredgecolor'] = styles2['color'] 
                            x = [dim / r, r * dim]
                            xm = [dim / (r**rec_taille_fac), dim * (r**rec_taille_fac)]
                            y = np.array(y) / dim
                            plt.plot([x[0], xm[0], x[0], x[1], xm[1], x[1], x[0]],
                                     [y[0], y[1],  y[2], y[2], y[1],  y[0], y[0]],
                                     markersize=0, **styles2)
                            styles2['linewidth'] = 0
                            plt.plot([x[0], x[1], x[1], x[0], x[0]],
                                     [y[0], y[0], y[2], y[2], y[0]],
                                     **styles2)
                            styles2['linewidth'] = 2  # median
                            plt.plot([x[0], x[1]], [y[1], y[1]],
                                     markersize=0, **styles2)
                # plot lines, we have to be smart to connect only adjacent dimensions
                for i, n in enumerate(tmp[:, 0]):
                    j = list(dimensions).index(n)
                    if i == len(tmp[:, 0]) - 1 or j == len(dimensions) - 1: 
                        break
                    if dimensions[j+1] == tmp[i+1, 0]:
                        res.extend(plt.plot(tmp[i:i+2, 0], tmp[i:i+2, 1] / tmp[i:i+2, 0]**ynormalize_by_dimension,
                                            markersize=0, clip_on=True, **styles[i_target]))
                # plot only marker
                lw = styles[i_target].get('linewidth', None) 
                styles[i_target]['linewidth'] = 0
                res.extend(plt.plot(tmp[:, 0], tmp[:, 1] / tmp[:, 0]**ynormalize_by_dimension,
                           markersize=20, clip_on=True, **styles[i_target]))
                # restore linewidth
                if lw:
                    styles[i_target]['linewidth'] = lw
                else:
                    del styles[i_target]['linewidth']

        # To have the legend displayed whatever happens with the data.
        for i in reversed(range(len(valuesOfInterest))):
            res.extend(plt.plot([], [], markersize=10,
                                label=valuesOfInterest.label(i) if isinstance(valuesOfInterest, pproc.RunlengthBasedTargetValues) else valuesOfInterest.loglabel(i),
                                **styles[i]))
        # Only for the last target function value
        if unsucc:  # obsolete
            tmp = np.vstack(unsucc)  # tmp[:, 0] needs to be sorted!
            # res.extend(plt.plot(tmp[:, 0], tmp[:, 1]/tmp[:, 0],
            #            color=styles[len(valuesOfInterest)-1]['color'],
            #            marker='x', markersize=20))
        if 1 < 3: # maxevals
            ylim = plt.ylim()
            res.extend(plt.plot(tmp[:, 0], maxevals / tmp[:, 0]**ynormalize_by_dimension,
                       color=styles[len(valuesOfInterest) - 1]['color'],
                       ls='', marker='x', markersize=20))
            plt.ylim(ylim)
        # median
        if mediandata:
            # for i, tm in mediandata.iteritems():
            for i in displaynumber:  # display median where success prob is smaller than one
                tm = mediandata[i]
                plt.plot((i,), (tm[1] / i**ynormalize_by_dimension,), 
                         color=styles[tm[0]]['color'],
                         linestyle='', marker='+', markersize=30,
                         markeredgewidth=5, zorder= -1)

        a = plt.gca()
        # the displaynumber is emptied for each new target precision
        # therefore the displaynumber displayed below correspond to the
        # last target (must be the hardest)
        if displaynumber:  # displayed only for the smallest valuesOfInterest
            for _k, j in displaynumber.iteritems():
                # the 1.5 factor is a shift up for the digits 
                plt.text(j[0], 1.5 * j[1] / j[0]**ynormalize_by_dimension, 
                         "%.0f" % j[2], axes=a,
                         horizontalalignment="center",
                         verticalalignment="bottom", fontsize=plt.rcParams['font.size'] * 0.85)
        # if later the ylim[0] becomes >> 1, this might be a problem
    return res

def plot_previous_algorithms(func, target=lambda x: [1e-8]):
    """Add graph of the BBOB-2009 virtual best algorithm."""
    if 11 < 3 and isinstance(values_of_interest, pproc.RunlengthBasedTargetValues):
        return None
    if not bestalg.bestalgentries2009:
        bestalg.loadBBOB2009()
    bestalgdata = []
    for d in dimensions:
        entry = bestalg.bestalgentries2009[(d, func)]
        tmp = entry.detERT([target((func, d))[-1]])[0]
        if not np.isinf(tmp):
            bestalgdata.append(tmp / d)
        else:
            bestalgdata.append(None)
    res = plt.plot(dimensions, bestalgdata, color=refcolor, linewidth=10,
                   marker='d', markersize=25, markeredgecolor='k',
                   zorder= -2)
    return res

def main(dsList, _valuesOfInterest, outputdir, verbose=True):
    """From a DataSetList, returns a convergence and ERT/dim figure vs dim.
    
    Uses data of BBOB 2009 (:py:mod:`bbob_pproc.bestalg`).
    
    :param DataSetList dsList: data sets
    :param seq _valuesOfInterest: target precisions, there might be as
                                  many graphs as there are elements in
                                  this input
    :param string outputdir: output directory
    :param bool verbose: controls verbosity
    
    """

    # plt.rc("axes", labelsize=20, titlesize=24)
    # plt.rc("xtick", labelsize=20)
    # plt.rc("ytick", labelsize=20)
    # plt.rc("font", size=20)
    # plt.rc("legend", fontsize=20)

    if not bestalg.bestalgentries2009:
        bestalg.loadBBOB2009()

    dictFunc = dsList.dictByFunc()

    for func in dictFunc:
        plot(dictFunc[func], _valuesOfInterest, styles=styles)  # styles might have changed via config
        beautify(axesLabel=False)
        plt.text(plt.xlim()[0], plt.ylim()[0], _valuesOfInterest.short_info, fontsize=14)
        if func in functions_with_legend:
            plt.legend(loc="best")
        if isBenchmarkinfosFound:
            plt.gca().set_title(funInfos[func])
        plot_previous_algorithms(func, _valuesOfInterest)
        filename = os.path.join(outputdir, 'ppfigdim_f%03d' % (func))
        saveFigure(filename, verbose=verbose)
        plt.close()

