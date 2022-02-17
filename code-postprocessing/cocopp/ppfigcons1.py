#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate performance scaling figures wrt constraints.
See also:
    - ppfigdim for the unconstrained case
    - compall/ppfigcons for the same plot but with a single target and 2 or more algorithms

TODOs:
    - add legend: on first figure ?
    - review the caption and html caption
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
from .compall.ppfigcons import ylim_upperbound
from .compall.pprldmany import text_infigure_if_constraints

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
        Scaling of runtime with number of constraints to reach certain target values.
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

def beautify(dim, axesLabel=True):
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

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    # quadratic slanted "grid"
    for i in range(-2, 7, 1 if ymax < 1e5 else 2):
        plt.plot((0.2, 20000), (10**i, 10**(i + 5)), 'k:', linewidth=0.5)
        # TODO: this should be done before the real lines are plotted?
    plt.xlim(.8, xmax)

    # for x in dimensions:
    #     plt.plot(2 * [x], [0.1, 1e11], 'k:', linewidth=0.5)

    # Ticks on axes
    # axisHandle.invert_xaxis()
    # TODO: All these should depend on one given input (xlim, ylim)

    # X ticks
    xtick_locs = [[n, 3 * n] for n in axisHandle.get_xticks()]
    xtick_locs = [n for sublist in xtick_locs for n in sublist]  # flatten
    xtick_locs = [n for n in xtick_locs if n > plt.xlim()[0] and n < plt.xlim()[1]]  # filter
    xtick_labels = ['%d' % int(n) if n < 1e10  # assure 1 digit for uniform figure sizes
                   else '' for n in xtick_locs]
    axisHandle.set_xticks(xtick_locs)
    axisHandle.set_xticklabels(xtick_labels)

    logyend = int(1 + np.log10(plt.ylim()[1]))
    axisHandle.set_yticks([10.**i for i in range(0, logyend)])
    axisHandle.set_yticklabels(range(0, logyend))
    
    if xlim_max is not None:
        if isinstance(values_of_interest, pproc.RunlengthBasedTargetValues):
            plt.ylim(0.3, xlim_max)  # set in config 
        else:
            # pass  # TODO: xlim_max seems to be not None even when not desired
            plt.ylim(None, min([plot.ylim()[1], xlim_max]))
    plt.ylim(ppfig.discretize_limits((ymin, ymax)))

    if axesLabel:
        plt.xlabel('Number of constraints')
        if ynormalize_by_dimension:
            plt.ylabel('Run Lengths / Dimension')
        else:
            plt.ylabel('Run Lengths')

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    upper_bound = ylim_upperbound(ymax)

    if upper_bound < 1e3:
        plt.grid(True, axis="y", which="both")

    plt.ylim(10**-0.2, upper_bound) # Set back the default maximum.
    # Vertical bar where dimension = number of active constraints
    plt.vlines(dim, 0, upper_bound, lw=2, linestyles="--", alpha=.5, colors=["blue"],
               label="m = dim")
    plt.vlines(int(dim * 1.5), 0, upper_bound, lw=2, linestyles="--", alpha=.5,
               colors=["green"], label="m(active) = dim")
            

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
    """From a DataSetList, plot a figure of ERT/dim vs number of constraints.
    
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
    res = []
    mediandata = {}
    displaynumber = {}
    no_target_reached = 1;
    for i_target in range(len(valuesOfInterest)):
        succ = []
        unsucc = []
        constraints = []
        maxevals = []
        for icons, ds in enumerate(dsList):
            cons = ds.number_of_constraints
            maxevals.append(max(ds.maxevals))
            constraints.append(cons)
            # (ert, success rate, number of success, total number of
            #        function evaluations, median of successful runs)
            data = generateData(ds, valuesOfInterest()[i_target])
            
            # data.append(np.append(dim, data))
            if data[2] > 0:  # Number of success is larger than 0
                succ.append(np.append(cons, data))
                if data[2] < ds.nbRuns():
                    displaynumber[cons] = ((cons, data[0], data[2]))
                mediandata[cons] = (i_target, data[-1])
                unsucc.append(np.append(cons, np.nan))
                no_target_reached = 0
            else:
                unsucc.append(np.append(cons, data[-2]))  # total number of fevals

        if len(succ) > 0:
            data = np.vstack(succ)
            # ERT
            if genericsettings.scaling_figures_with_boxes:
                for icons, cons in enumerate(constraints): 
                    # to find finite simulated runlengths we need to have at least one successful run
                    if dsList[icons].detSuccesses([valuesOfInterest()[i_target]])[0]:
                        # make a box-plot
                        y = toolsstats.drawSP_from_dataset(
                                            dsList[icons],
                                            valuesOfInterest()[i_target],
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
                        x = [cons / r, r * cons]
                        xm = [cons / (r**rec_taille_fac), cons * (r**rec_taille_fac)]
                        y = np.array(y) / ds.dim**ynormalize_by_dimension
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
            for i, n in enumerate(data[:, 0]):
                j = list(constraints).index(n)
                if i == len(data[:, 0]) - 1 or j == len(constraints) - 1: 
                    break
                if constraints[j+1] == data[i+1, 0]:
                    res.extend(plt.plot(data[i:i+2, 0], data[i:i+2, 1] / ds.dim**ynormalize_by_dimension,
                                        markersize=0, clip_on=True, **styles[i_target]))
            # plot only marker
            lw = styles[i_target].get('linewidth', None) 
            styles[i_target]['linewidth'] = 0
            res.extend(plt.plot(data[:, 0], data[:, 1] / ds.dim**ynormalize_by_dimension,
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
        data = np.vstack(unsucc)  # data[:, 0] needs to be sorted!
        # res.extend(plt.plot(data[:, 0], data[:, 1]/data[:, 0],
        #            color=styles[len(valuesOfInterest)-1]['color'],
        #            marker='x', markersize=20))
    if 1 < 3: # maxevals
        maxevals = np.asarray(maxevals)
        ylim = plt.ylim()
        res.extend(plt.plot(data[:, 0], maxevals / ds.dim**ynormalize_by_dimension,
                   color=styles[len(valuesOfInterest) - 1]['color'],
                   ls='', marker='x', markersize=20))
        if no_target_reached:
            ylim_maxevals = max(maxevals / ds.dim**ynormalize_by_dimension)
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
                plt.text(j[0], 1.5 * j[1] / ds.dim**ynormalize_by_dimension, 
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
            data = entry.detERT([target((func, d))[-1]])[0]
            if not np.isinf(data):
                refalgdata.append(data / d)
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

    dictFuncCons = dsList.dictByFuncCons()
    dimensions = list(dsList.dictByDim().keys())
    values_of_interest = testbedsettings.current_testbed.ppfigdim_target_values
    key = 'bbobppfigconsonelegend' + testbedsettings.current_testbed.scenario
    joined_values_of_interest = ', '.join(values_of_interest.labels()) if genericsettings.runlength_based_targets else ', '.join(values_of_interest.loglabels())
    caption = htmldesc.getValue('##' + key + '##').replace('valuesofinterest', joined_values_of_interest)

    ppfig.save_single_functions_html(
        os.path.join(outputdir, 'ppfigcons1'),
        htmlPage = ppfig.HtmlPage.PPFIGCONS1,
        parentFileName=genericsettings.single_algorithm_file_name,
        dimensions=dimensions,
        caption=caption)

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

    for ifunc, func in enumerate(dictFuncCons):
        dd = dictFuncCons[func].dictByDim()
        for idim, dim in enumerate(dd.keys()):
            plot(dd[dim], _valuesOfInterest, styles=styles)  # styles might have changed via config
            beautify(dim)
            plt.title("%s, %s-D" % (func, dim))
            # display number of instances in data and used targets type:
            if all(set(d.instancenumbers) == set(dd[dim][0].instancenumbers) for d in dd[dim]):
                display_text = '%d instances\n' % len(((dd[dim][0]).instancenumbers))
            else:
                display_text = 'instances %s' % [d.instancenumbers for d in dd[dim]]
            display_text += _valuesOfInterest.short_info
            display_text += text_infigure_if_constraints()
            if ifunc == 0 and idim == 0:
                # because of legend, put annotation somewhere else
                toolsdivers.legend(loc="lower left", fontsize=12, ncol=3)
                loc = (plt.xlim()[0], plt.ylim()[1])
                align = ("left", "top")
            else:
                loc = (plt.xlim()[0], plt.ylim()[0])
                align = ("left", "bottom")

            plt.text(loc[0], loc[1], display_text,
                     fontsize=12, horizontalalignment=align[0],
                     verticalalignment=align[1])

            if genericsettings.scaling_plots_with_axis_labels:
                plt.xlabel('Number of constraints')
                plt.ylabel('log10(%s / dimension)' % testbedsettings.current_testbed.string_evals_legend)

            plot_previous_algorithms(func, _valuesOfInterest)
            filename = os.path.join(outputdir, 'ppfigcons1_%s_d%d' % (func, dim))
            with warnings.catch_warnings(record=True) as ws:
                ppfig.save_figure(filename, dsList[0].algId)
                if len(ws):
                    for w in ws:
                        print(w)
                    print('while saving figure in "' + filename +
                            '" (in ppfigcons1.py:551)')

            plt.close()