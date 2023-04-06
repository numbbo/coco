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
The thick light line with diamond markers shows the results of the
specified reference algorithm for df = 1e-8 or a runlength-based
target (if in the expensive/runlength-based targets setting).

**Example**

.. plot::
    :width: 50%
    
    import urllib
    import tarfile
    import glob
    from pylab import *
    
    import cocopp
    
    # Collect and unarchive data (3.4MB)
    dataurl = 'http://coco.lri.fr/BBOB2009/pythondata/BIPOP-CMA-ES.tar.gz'
    filename, headers = urllib.urlretrieve(dataurl)
    archivefile = tarfile.open(filename)
    archivefile.extractall()
    
    # Scaling figure
    ds = cocopp.load(glob.glob('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f002_*.pickle'))
    figure()
    cocopp.ppfigdim.plot(ds)
    cocopp.ppfigdim.beautify()
    cocopp.ppfigdim.plot_previous_algorithms(2, False) # plot BBOB 2009 best algorithm on fun 2

"""
from __future__ import absolute_import

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from six import advance_iterator

from . import genericsettings, toolsstats, bestalg, pproc, ppfig, ppfigparam, htmldesc, toolsdivers
from . import testbedsettings
from . import captions

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


# should correspond with the colors in pprldistr.

def scaling_figure_caption():
    """Provides a figure caption with the help of captions.py for
       replacing common texts, abbreviations, etc.
    """

    caption_text = (r"""%
        Scaling of runtime with dimension to reach certain target values"""
        + (" !!DF!!." if not testbedsettings.current_testbed.has_constraints else ".")
        + r"""
        Lines: expected runtime (\ERT);
        Cross (+): median runtime of successful runs to reach the most difficult
        target that was reached at least once (but not always);
        Cross ({\color{red}$\times$}): maximum number of
        """
        + testbedsettings.current_testbed.string_evals
        + r""" in any trial. !!NOTCHED-BOXES!!
        All values are !!DIVIDED-BY-DIMENSION!! 
        plotted as $\log_{10}$ values versus dimension. %
        """)
    
    caption_part_absolute_targets = (r"""%
        Shown is the \ERT\ for fixed target precision values of $10^k$ with $k$ given
        in the legend.
        Numbers above \ERT-symbols (if appearing) indicate the number of trials
        reaching the respective target. """ + # TODO: add here "(out of XYZ trials)"
        r"""!!LIGHT-THICK-LINE!! Horizontal lines mean linear scaling, slanted
        grid lines depict quadratic scaling.""")

    caption_part_rlbased_targets = r"""%
        Shown is the \ERT\ for targets just not reached by !!THE-REF-ALG!!
        within the given budget $k\times\DIM$, where $k$ is shown in the
        legend. Numbers above \ERT-symbols (if appearing) indicate the number
        of trials reaching the respective target. !!LIGHT-THICK-LINE!! Slanted
        grid lines indicate a scaling with $\mathcal O$$(\DIM)$ compared to
        $\mathcal O$$(1)$ when using the respective reference algorithm.
        """

    if genericsettings.runlength_based_targets:
        figure_caption = captions.replace(caption_text + caption_part_rlbased_targets)
    else:
        figure_caption = captions.replace(caption_text + caption_part_absolute_targets)
        
    return figure_caption

def beautify(axesLabel=True):
    """Customize figure presentation.
    
    Uses information from the appropriate benchmark short infos file 
    for figure title. 
    
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

    values_of_interest = testbedsettings.current_testbed.ppfigdim_target_values

    # horizontal grid
    if isinstance(values_of_interest, pproc.RunlengthBasedTargetValues):
        axisHandle.yaxis.grid(False, which='major')
        expon = values_of_interest.times_dimension - ynormalize_by_dimension
        for (i, y) in enumerate(reversed(values_of_interest.run_lengths)):
            plt.plot((1, 200), [y, y], 'k:', linewidth=0.2)
            if i / 2. == i // 2:
                plt.plot((1, 200), [y, y * 200**expon],
                         styles[i]['color'] + '-', linewidth=0.2)
    else:
        # TODO: none of this is visible in svg format!
        axisHandle.yaxis.grid(True, which='major')
        # for i in range(0, 11):
        #     plt.plot((0.2, 20000), 2 * [10**i], 'k:', linewidth=0.5)

    # quadratic slanted "grid"
    for i in range(-2, 7, 1 if ymax < 1e5 else 2):
        plt.plot((0.2, 20000), (10**i, 10**(i + 5)), 'k:', linewidth=0.5)
        # TODO: this should be done before the real lines are plotted?


    # for x in dimensions:
    #     plt.plot(2 * [x], [0.1, 1e11], 'k:', linewidth=0.5)

    # Ticks on axes
    # axisHandle.invert_xaxis()
    
    dimensions = testbedsettings.current_testbed.dimensions_to_display
    dimticklist = dimensions
    dimannlist = dimensions 
    # TODO: All these should depend on one given input (xlim, ylim)

    axisHandle.set_xticks(dimticklist)
    axisHandle.set_xticklabels([str(n) for n in dimannlist])
    logyend = 11  # int(1 + np.log10(plt.ylim()[1]))
    axisHandle.set_yticks([10.**i for i in range(0, logyend)])
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
        for i in range(int(np.log10(plt.ylim()[0])), int(np.log10(1 + plt.ylim()[1]))):
            ticks += [10 ** i, 2 * 10 ** i, 5 * 10 ** i]
        axisHandle.set_yticks(ticks)
        # axisHandle.set_yticklabels(ticklabels)
    # axes limites
    plt.xlim(0.9 * dimensions[0], 1.125 * dimensions[-1]) 
    if xlim_max is not None:
        if isinstance(values_of_interest, pproc.RunlengthBasedTargetValues):
            plt.ylim(0.3, xlim_max)  # set in config 
        else:
            # pass  # TODO: xlim_max seems to be not None even when not desired
            plt.ylim(None, min([plot.ylim()[1], xlim_max]))
    plt.ylim(ppfig.discretize_limits((ymin, ymax)))

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
    i = advance_iterator(it)
    prev = np.array([np.nan] * len(i))

    while i[0] <= targetFuncValue:
        prev = i
        try:
            i = advance_iterator(it)
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

def plot_a_bar(x, y,
               plot_cmd=plt.loglog,
               rec_width=0.1, # box ("rectangle") width, log scale 
               rec_taille_fac=0.3,  # notch width parameter
               styles={'color': 'b'},
               linewidth=1,
               fill_color=None, # None means no fill
               fill_transparency=0.7  # 1 should be invisible
               ):
    """plot/draw a notched error bar, x is the x-position,
    y[0,1,2] are lower, median and upper percentile respectively. 
    
    hold(True) to see everything.

    TODO: with linewidth=0, inf is not visible
    
    """
    if not np.isfinite(y[2]):
        y[2] = y[1] + 100 * (y[1] - y[0])
        if plot_cmd in (plt.loglog, plt.semilogy):
            y[2] = (1 + y[1]) * (1 + y[1] / y[0])**10
    if not np.isfinite(y[0]):
        y[0] = y[1] - 100 * (y[2] - y[1])
        if plot_cmd in (plt.loglog, plt.semilogy):
            y[0] = y[1] / (1 + y[2] / y[1])**10
    styles2 = {}
    for s in styles:
        styles2[s] = styles[s]
    styles2['linewidth'] = linewidth
    styles2['markeredgecolor'] = styles2['color']
    dim = 1  # to remove error
    x0 = x
    if plot_cmd in (plt.loglog, plt.semilogx):
        r = np.exp(rec_width) # ** ((1. + i_target / 3.) / 4)  # more difficult targets get a wider box
        x = [x0 * dim / r, x0 * r * dim] # assumes log-scale of x-axis
        xm = [x0 * dim / (r**rec_taille_fac), x0 * dim * (r**rec_taille_fac)]
    else:
        r = rec_width
        x = [x0 * dim - r, x0 * dim + r]
        xm = [x0 * dim - (r * rec_taille_fac), x0 * dim + (r * rec_taille_fac)]

    y = np.array(y) / dim
    if fill_color is not None:
        plt.fill_between([x[0], xm[0], x[0], x[1], xm[1], x[1], x[0]],
                         [y[0], y[1],  y[2], y[2], y[1],  y[0], y[0]], 
                         color=fill_color, alpha=1-fill_transparency)
    plot_cmd([x[0], xm[0], x[0], x[1], xm[1], x[1], x[0]],
             [y[0], y[1],  y[2], y[2], y[1],  y[0], y[0]],
             markersize=0, **styles2)
    styles2['linewidth'] = 0
    plot_cmd([x[0], x[1], x[1], x[0], x[0]],
             [y[0], y[0], y[2], y[2], y[0]],
             **styles2)
    styles2['linewidth'] = 2  # median
    plot_cmd([x[0], x[1]], [y[1], y[1]],
             markersize=0, **styles2)

    
def plot(dsList, valuesOfInterest=None, styles=styles):
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
    if not valuesOfInterest:
        valuesOfInterest = testbedsettings.current_testbed.ppfigdim_target_values

    valuesOfInterest = pproc.TargetValues.cast(valuesOfInterest)
    styles = list(reversed(styles[:len(valuesOfInterest)]))
    dsList = pproc.DataSetList(dsList)
    dictFunc = dsList.dictByFunc()
    res = []

    for func in dictFunc:
        dictFunc[func] = dictFunc[func].dictByDim()
        dimensions = sorted(dictFunc[func])

        mediandata = {}
        displaynumber = {}
        no_target_reached = 1;
        for i_target in range(len(valuesOfInterest)):
            succ = []
            unsucc = []
            # data = []
            maxevals = np.ones(len(dimensions))
            # Collect data that have the same function and different dimension.
            for idim, dim in enumerate(dimensions):
                if len(dictFunc[func][dim]) > 1:
                    raise ppfig.Usage('\nFound more than one algorithm inside one data folder. '
                                      'Specify a separate data folder for each algorithm.')
                elif len(dictFunc[func][dim]) < 1:
                    raise ppfig.Usage('\nNo data for function %s and dimension %d.' % (func, dim))
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
                    no_target_reached = 0
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
            if no_target_reached:
                ylim_maxevals = max(maxevals / tmp[:, 0]**ynormalize_by_dimension)
                ylim = (ylim[0], ylim_maxevals)
                plt.annotate('no target reached', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=14,
                            horizontalalignment='center', verticalalignment='center')
            plt.ylim(ylim)
        # median
        if mediandata:
            # for i, tm in mediandata.items():
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
            for _k, j in displaynumber.items():
                # the 1.5 factor is a shift up for the digits 
                plt.text(j[0], 1.5 * j[1] / j[0]**ynormalize_by_dimension, 
                         "%.0f" % j[2], axes=a,
                         horizontalalignment="center",
                         verticalalignment="bottom",
                         fontsize=genericsettings.rcfont['size'] * 0.85)
        # if later the ylim[0] becomes >> 1, this might be a problem
    return res

def plot_previous_algorithms(func, target=None):  # lambda x: [1e-8]):
    """Add graph of the reference algorithm, specified in
    testbedsettings.current_testbed using the
    last, most difficult target in ``target``."""
    
    testbed = testbedsettings.current_testbed    
    
    if not target:
        target = testbed.ppfigdim_target_values
        
    target = pproc.TargetValues.cast(target)

    refalgentries = bestalg.load_reference_algorithm(testbed.reference_algorithm_filename)
    
    if not refalgentries:
        return None

    refalgdata = []
    dimensions = testbedsettings.current_testbed.dimensions_to_display
    for d in dimensions:
        try:
            entry = refalgentries[(d, func)]
            tmp = entry.detERT([target((func, d))[-1]])[0]
            if not np.isinf(tmp):
                refalgdata.append(tmp / d)
            else:
                refalgdata.append(np.inf)
        except KeyError: # dimension not in refalg
            refalgdata.append(np.inf)  # None/nan give a runtime warning
        
    res = plt.plot(dimensions, refalgdata, color=refcolor, linewidth=10,
                   marker='d', markersize=25, markeredgecolor='k',
                   zorder= -2)
    return res

def main(dsList, _valuesOfInterest, outputdir):
    """From a DataSetList, returns a convergence and ERT/dim figure vs dim.
    
    If available, uses data of a reference algorithm as specified in 
    ``:py:genericsettings.py``.
    
    :param DataSetList dsList: data sets
    :param seq _valuesOfInterest: target precisions, either as list or as
                                  ``pproc.TargetValues`` class instance. 
                                  There will be as many graphs as there are 
                                  elements in this input. 
    :param string outputdir: output directory
    
    """

    _valuesOfInterest = pproc.TargetValues.cast(_valuesOfInterest)

    dictFunc = dsList.dictByFunc()
    values_of_interest = testbedsettings.current_testbed.ppfigdim_target_values

    key = 'bbobppfigdimlegend' + testbedsettings.current_testbed.scenario
    joined_values_of_interest = ', '.join(values_of_interest.labels()) if genericsettings.runlength_based_targets else ', '.join(values_of_interest.loglabels())
    caption = htmldesc.getValue('##' + key + '##').replace('valuesofinterest', joined_values_of_interest)
    header = 'Scaling of run "time" with problem dimension'

    ppfig.save_single_functions_html(
        os.path.join(outputdir, 'ppfigdim'),
        htmlPage = ppfig.HtmlPage.NON_SPECIFIED,
        parentFileName=genericsettings.single_algorithm_file_name,
        header = header,
        caption = caption)

    ppfig.save_single_functions_html(
        os.path.join(outputdir, 'pprldistr'),
        htmlPage = ppfig.HtmlPage.PPRLDISTR,
        function_groups= dsList.getFuncGroups(),
        parentFileName=genericsettings.single_algorithm_file_name)

    if not testbedsettings.current_testbed.reference_algorithm_filename == '':
        ppfig.save_single_functions_html(
            os.path.join(outputdir, 'pplogloss'),
            htmlPage = ppfig.HtmlPage.PPLOGLOSS,
            function_groups= dsList.getFuncGroups(),
            parentFileName=genericsettings.single_algorithm_file_name)

    ppfig.copy_js_files(outputdir)

    funInfos = ppfigparam.read_fun_infos()
    fontSize = ppfig.getFontSize(funInfos.values())

    for func in dictFunc:
        plot(dictFunc[func], _valuesOfInterest, styles=styles)  # styles might have changed via config
        beautify(axesLabel=False)
        
        # display number of instances in data and used targets type:
        if all(set(d.instancenumbers) == set(dictFunc[func][0].instancenumbers)
               for d in dictFunc[func]): # all the same?
            display_text = '%d instances\n' % len(((dictFunc[func][0]).instancenumbers))
        else:
            display_text = 'instances %s' % [d.instancenumbers for d in dictFunc[func]]
        display_text += _valuesOfInterest.short_info
        plt.text(plt.xlim()[0], plt.ylim()[0],
                 display_text, fontsize=14, horizontalalignment="left",
                 verticalalignment="bottom")

        if func in testbedsettings.current_testbed.functions_with_legend:
            toolsdivers.legend(loc="best", fontsize=16)
        if func in funInfos.keys():
            funcName = funInfos[func]
            plt.gca().set_title(funcName, fontsize=fontSize)

        if genericsettings.scaling_plots_with_axis_labels:
            plt.xlabel('dimension')
            plt.ylabel('log10(# f-evals / dimension)')

        plot_previous_algorithms(func, _valuesOfInterest)
        filename = os.path.join(outputdir, 'ppfigdim_f%03d' % (func))
        with warnings.catch_warnings(record=True) as ws:
            ppfig.save_figure(filename, dsList[0].algId)
            if len(ws):
                for w in ws:
                    print(w)
                print('while saving figure to "' + filename + '" (at ppfigdim.py:595)')

        plt.close()
