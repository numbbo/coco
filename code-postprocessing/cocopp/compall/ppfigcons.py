#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate performance scaling figures for the constrained testbed.

The figures show the scaling of the performance in terms of ERT w.r.t.
number of constraints on a log-log scale.

TODO:
- docstring
- example

Comments:
(Paul) for now the strategy is to keep the same interface as in ppfigdim.py
and implement an additional call to ppfigcons if data is from the constrained test suite
"""

from __future__ import absolute_import, print_function
import os
import matplotlib.pyplot as plt
import numpy
import warnings
from pdb import set_trace
from .. import toolsdivers, toolsstats, bestalg, pproc, genericsettings, htmldesc, ppfigparam, ppfig
from .. import testbedsettings
from .. import captions
from ..ppfig import save_figure, get_plotting_styles, getFontSize
from ..pptex import color_to_latex, marker_to_latex, marker_to_html, writeLabels
from .ppfigs import generateData
from .pprldmany import text_infigure_if_constraints

show_significance = 0.01  # for zero nothing is shown
ratio_nbsucc_to_display = .5  # annotate symbol if ratio of success < this
refcolor = 'wheat'

show_algorithms = []
fontsize = 14.0
legend_text_max_len = 14
legend = False


def legend_fontsize_scaler(number_of_entries=None):
    """return a fontsize scaling factor depending on the number of entries
    in the legend.

    Works currently well with fontsize 14, where a legend with up to ~30
    entries will still fit into the figure.
    """
    if not number_of_entries:
        number_of_entries = len(plt.gca().get_legend_handles_labels()[1])
    return 2.55 / (number_of_entries + 1.5)**0.5


def fix_styles(plotting_styles, line_styles):
    """a short hack to fix length of styles"""
    m = len(line_styles)
    while len(line_styles) < len(plotting_styles.algorithm_list):
        line_styles.append(line_styles[len(line_styles) % m])
    for i in range(len(line_styles)):
        if plotting_styles.in_background:
            line_styles[i].update(plotting_styles.ppfigs_styles)
        else:
            line_styles[i].update({'linewidth': 4 - min([2, i / 3.0]),  # thinner lines over thicker lines
                                   'markeredgewidth': 3 - min([2, i / 2.0]),
                                   'markersize': int(line_styles[i]['markersize'] / 2),
                                   'markerfacecolor': 'None'})

def prepare_scaling_figure_caption():

    scaling_figure_caption_start_fixed = (r"""Expected running time (\ERT\ in number of %s
                    as $\log_{10}$ value), divided by dimension for target function value $!!PPFIGS-FTARGET!!$
                    versus number of constraints. Slanted grid lines indicate quadratic scaling with the dimension. """ % testbedsettings.current_testbed.string_evals
                                          )

    scaling_figure_caption_start_rlbased = (r"""Expected running time (\ERT\ in number of %s
                        as $\log_{10}$ value) divided by dimension versus number of constraints. The target function value
                        is chosen such that !!THE-REF-ALG!! just failed to achieve
                        an \ERT\ of $!!PPFIGS-FTARGET!!\times\DIM$. """ % testbedsettings.current_testbed.string_evals
                                            )

    scaling_figure_caption_end = (
        r"Different symbols " +
        r"correspond to different algorithms given in the legend of #1. " +
        r"If the success ratio is < %d/%d " % ratio_nbsucc_to_display.as_integer_ratio() +
        r"and > 0 the symbol is annotated with the number of successes. " +
        r"Light symbols give the maximum number of evaluations from the longest trial " +
        r"divided by dimension. " +
        (r"Black stars (if present) indicate a statistically better result compared to all other algorithms " +
         r"with $p<0.01$ and Bonferroni correction number of constraints (six).  ")
        if show_significance else ''
    )

    scaling_figure_caption_fixed = scaling_figure_caption_start_fixed + scaling_figure_caption_end
    scaling_figure_caption_rlbased = scaling_figure_caption_start_rlbased + scaling_figure_caption_end

    if testbedsettings.current_testbed.name in [testbedsettings.suite_name_bi_ext,
                                                testbedsettings.suite_name_cons,
                                                testbedsettings.suite_name_ls,
                                                testbedsettings.suite_name_mixint,
                                                testbedsettings.suite_name_bi_mixint]:
        # NOTE: no runlength-based targets supported yet
        figure_caption = scaling_figure_caption_fixed
    elif testbedsettings.current_testbed.name in [testbedsettings.suite_name_single,
                                                  testbedsettings.suite_name_single_noisy,
                                                  testbedsettings.suite_name_bi]:
        if genericsettings.runlength_based_targets:
            figure_caption = scaling_figure_caption_rlbased
        else:
            figure_caption = scaling_figure_caption_fixed
    else:
        warnings.warn("Current settings do not support ppfigdim caption.")

    return figure_caption


def scaling_figure_caption(for_html=False):


    if for_html:
        figure_caption = htmldesc.getValue('##bbobppfigconslegend' +
                                           testbedsettings.current_testbed.scenario + '##')
    else:
        figure_caption = prepare_scaling_figure_caption()

    return captions.replace(figure_caption, html=for_html)


def prepare_ecdfs_figure_caption():
    testbed = testbedsettings.current_testbed
    refalgtext = (
                  r"As reference algorithm, !!THE-REF-ALG!! " +
                  r"is shown as light " +
                  r"thick line with diamond markers."
                 )

    ecdfs_figure_caption_standard = (
                r"!!BOOTSTRAPPED-BEGINNING!!mpirical cumulative distribution of the number " +
                r"of %s divided by dimension " % testbedsettings.current_testbed.string_evals +
                r"(%s/DIM) for $!!NUM-OF-TARGETS-IN-ECDF!!$ " % testbedsettings.current_testbed.string_evals_short +
                r"targets with target precision in !!TARGET-RANGES-IN-ECDF!! " +
                r"for all functions and subgroups in #1-D. "
                )
    ecdfs_figure_caption_rlbased = (
                r"!!BOOTSTRAPPED-BEGINNING!!mpirical cumulative distribution of the number " +
                r"of %s divided by dimension " % testbedsettings.current_testbed.string_evals +
                r"(%s/DIM) for all functions and subgroups in #1-D. " % testbedsettings.current_testbed.string_evals_short +
                r"The targets are chosen from !!TARGET-RANGES-IN-ECDF!! " +
                r"such that !!THE-REF-ALG!! just " +
                r"not reached them within a given budget of $k$ $\times$ DIM, " +
                r"with $!!NUM-OF-TARGETS-IN-ECDF!!$ different values of $k$ " +
                r"chosen equidistant in logscale within the interval " +
                r"$\{0.5, \dots, 50\}$. "
                )

    if testbed.name in [testbedsettings.suite_name_bi_ext,
                        testbedsettings.suite_name_cons,
                        testbedsettings.suite_name_ls,
                        testbedsettings.suite_name_mixint,
                        testbedsettings.suite_name_bi_mixint]:
        # NOTE: no runlength-based targets supported yet
        figure_caption = ecdfs_figure_caption_standard
    elif testbed.name in [testbedsettings.suite_name_single,
                          testbedsettings.suite_name_single_noisy,
                          testbedsettings.suite_name_bi]:
        if genericsettings.runlength_based_targets:
            figure_caption = ecdfs_figure_caption_rlbased + refalgtext
        else:
            figure_caption = ecdfs_figure_caption_standard + refalgtext
    else:
        warnings.warn("Current settings do not support ppfigdim caption.")

    return captions.replace(figure_caption)


def ecdfs_figure_caption(for_html=False, dimension=0):

    if for_html:
        key = '##bbobECDFslegend%s##' % testbedsettings.current_testbed.scenario
        caption = htmldesc.getValue(key)
    else:
        caption = prepare_ecdfs_figure_caption()   

    return captions.replace(caption, html=for_html)


def get_ecdfs_single_fcts_caption():
    ''' Returns figure caption for single function ECDF plots. '''

    if genericsettings.runlength_based_targets:
        s = (r"""Empirical cumulative distribution of !!SIMULATED-BOOTSTRAP!! runtimes in number
             of %s divided by dimension (%s/DIM) for  
             targets in !!TARGET-RANGES-IN-ECDF!! that have just not
             been reached by !!THE-REF-ALG!!
             in a given budget of $k$ $\times$ DIM, with $!!NUM-OF-TARGETS-IN-ECDF!!$ 
             different values of $k$ chosen equidistant in logscale within the interval $\{0.5, \dots, 50\}$.
             Shown are functions $f_{#1}$ to $f_{#2}$ and all dimensions. """ % (
                testbedsettings.current_testbed.string_evals,
                testbedsettings.current_testbed.string_evals_short)
             )
    else:
        s = (r"""Empirical cumulative distribution of !!SIMULATED-BOOTSTRAP!! runtimes in number
             of %s divided by dimension (%s/DIM) for the 
             $!!NUM-OF-TARGETS-IN-ECDF!!$ targets !!TARGET-RANGES-IN-ECDF!!
             for functions $f_{#1}$ to $f_{#2}$ and all dimensions. """ % (
                testbedsettings.current_testbed.string_evals,
                testbedsettings.current_testbed.string_evals_short)
             )

    return captions.replace(s)


def get_ecdfs_all_groups_caption():
    ''' Returns figure caption for ECDF plots aggregating over function groups. '''
    
    if genericsettings.runlength_based_targets:
        s = (r"""Empirical cumulative distribution of !!SIMULATED-BOOTSTRAP!!
             runtimes, measured in number of %s 
             divided by dimension (%s/DIM) for all function groups and all 
             dimensions and for those targets in
             !!TARGET-RANGES-IN-ECDF!! that have just not been reached by 
             !!THE-REF-ALG!! in a given budget of $k$ $\times$ DIM, with 
             $!!NUM-OF-TARGETS-IN-ECDF!!$ different values of $k$ chosen 
             equidistant in logscale within the interval $\{0.5, \dots, 50\}$.             
             The aggregation over all !!TOTAL-NUM-OF-FUNCTIONS!! 
             functions is shown in the last plot.""" % (
                testbedsettings.current_testbed.string_evals,
                testbedsettings.current_testbed.string_evals_short)
             )
    else:
        s = (r"""Empirical cumulative distribution of !!SIMULATED-BOOTSTRAP!!
             runtimes, measured in number of %s,
             divided by dimension (%s/DIM) for the $!!NUM-OF-TARGETS-IN-ECDF!!$ 
             targets !!TARGET-RANGES-IN-ECDF!! for all function groups and all 
             dimensions. The aggregation over all !!TOTAL-NUM-OF-FUNCTIONS!! 
             functions is shown in the last plot.""" % (
                testbedsettings.current_testbed.string_evals,
                testbedsettings.current_testbed.string_evals_short)
             )
    return captions.replace(s)


def get_ecdfs_single_functions_single_dim_caption():
    ''' Returns figure caption for single function ECDF plots
        showing the results of 2+ algorithms in a single dimension. '''
    
    if genericsettings.runlength_based_targets:
        s = (r"""Empirical cumulative distribution of !!SIMULATED-BOOTSTRAP!!
             runtimes, measured in number of %s 
             divided by dimension (%s/DIM) in 
             dimension #1 and for those targets in
             !!TARGET-RANGES-IN-ECDF!! that have just not been reached by 
             !!THE-REF-ALG!! in a given budget of $k$ $\times$ DIM, with 
             $!!NUM-OF-TARGETS-IN-ECDF!!$ different values of $k$ chosen 
             equidistant in logscale within the interval $\{0.5, \dots, 50\}$.""" % (
                testbedsettings.current_testbed.string_evals,
                testbedsettings.current_testbed.string_evals_short)
             )
    else:
        s = (r"""Empirical cumulative distribution of !!SIMULATED-BOOTSTRAP!!
             runtimes, measured in number of %s,
             divided by dimension (%s/DIM) for the $!!NUM-OF-TARGETS-IN-ECDF!!$ 
             targets !!TARGET-RANGES-IN-ECDF!! in dimension #1.""" % (
                testbedsettings.current_testbed.string_evals,
                testbedsettings.current_testbed.string_evals_short)
             )
    return captions.replace(s)


def plotLegend(handles, maxval=None):
    """Display right-side legend.
    
    Sorted from smaller to larger y-coordinate values.
    
    """
    ys = {}
    lh = 0 # Number of labels to display on the right
    if not maxval:
        maxval = []
        for h in handles:
            x2 = []
            y2 = []
            for i in h:
                x2.append(plt.getp(i, "xdata"))
            x2 = numpy.sort(numpy.hstack(x2))
            maxval.append(max(x2))
        maxval = max(maxval)

    for h in handles:
        x2 = []
        y2 = []

        for i in h:
            x2.append(plt.getp(i, "xdata"))
            y2.append(plt.getp(i, "ydata"))

        x2 = numpy.array(numpy.hstack(x2))
        y2 = numpy.array(numpy.hstack(y2))
        tmp = numpy.argsort(x2)
        x2 = x2[tmp]
        y2 = y2[tmp]
        h = h[-1]

        # ybis is used to sort in case of ties
        try:
            tmp = x2 <= maxval
            y = y2[tmp][-1]
            ybis = y2[tmp][y2[tmp] < y]
            if len(ybis) > 0:
                ybis = ybis[-1]
            else:
                ybis = y2[tmp][-2]
            ys.setdefault(y, {}).setdefault(ybis, []).append(h)
            lh += 1
        except IndexError:
            pass

    if len(show_algorithms) > 0:
        lh = min(lh, len(show_algorithms))

    if lh <= 1:
        lh = 2

    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()

    i = 0 # loop over the elements of ys
    for j in sorted(ys.keys()):
        for k in sorted(ys[j].keys()):
            # enforce that a "best" algorithm comes first in case of equality
            tmp = []
            for h in ys[j][k]:
                if 'best' in plt.getp(h, 'label'):
                    tmp.insert(0, h)
                else:
                    tmp.append(h)
            #tmp.reverse()
            ys[j][k] = tmp

            for h in ys[j][k]:
                if (not plt.getp(h, 'label').startswith('_line') and
                    (len(show_algorithms) == 0 or
                     plt.getp(h, 'label') in show_algorithms)):
                    y = 0.02 + i * 0.96/(lh-1)
                    # transform y in the axis coordinates
                    #inv = plt.gca().transLimits.inverted()
                    #legx, ydat = inv.transform((.9, y))
                    #leglabx, ydat = inv.transform((.92, y))
                    #set_trace()

                    ydat = 10**(y * numpy.log10(ymax/ymin)) * ymin
                    legx = 10**(.85 * numpy.log10(xmax/xmin)) * xmin
                    leglabx = 10**(.87 * numpy.log10(xmax/xmin)) * xmin
                    tmp = {}
                    for attr in ('lw', 'linestyle', 'marker',
                                 'markeredgewidth', 'markerfacecolor',
                                 'markeredgecolor', 'markersize', 'zorder'):
                        tmp[attr] = plt.getp(h, attr)
                    plt.plot((maxval, legx), (j, ydat),
                             color=plt.getp(h, 'markeredgecolor'), **tmp)

                    plt.text(leglabx, ydat,
                             plt.getp(h, 'label'), horizontalalignment="left",
                             verticalalignment="center", size=fontsize)
                    i += 1

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    if maxval:
        plt.axvline(maxval, color='k')


def ylim_upperbound(ymax):
    upper_bound = 1.1e2
    while True:
        if ymax < upper_bound:
            return upper_bound
        upper_bound *= 100


def beautify(dim, legend=False, rightlegend=False):
    """Customize figure format.

    adding a legend, axis label, etc

    :param bool legend: if True, display a box legend
    :param bool rightlegend: if True, makes some space on the right for
                             legend

    """
    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_xscale("log")
    try:
        axisHandle.set_yscale("log")
    except OverflowError:
        set_trace()

    # Grid options
    axisHandle.yaxis.grid(True)

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()

    # quadratic slanted "grid"
    for i in range(-2, 7, 1 if ymax < 1e5 else 2):
        plt.plot((0.2, 20000), (10**i, 10**(i + 5)), 'k:',
                 linewidth=0.5)  # grid should be on top

    plt.xlim(.8, xmax)
    upper_bound = ylim_upperbound(ymax)
    plt.ylim(10**-0.2, upper_bound) # Set back the default maximum.

    # Vertical bar where dimension = number of active constraints
    plt.vlines(dim, 0, upper_bound, lw=2, ls="--", alpha=.5, colors=["blue"],
               label="m = dim")
    plt.vlines(int(dim * 1.5), 0, upper_bound, lw=2, ls="--", alpha=.5,
               colors=["green"], label="m(active) = dim")

    # X ticks
    xtick_locs = [[n, 3 * n] for n in axisHandle.get_xticks()]
    xtick_locs = [n for sublist in xtick_locs for n in sublist]  # flatten
    xtick_locs = [n for n in xtick_locs if n > plt.xlim()[0] and n < plt.xlim()[1]]  # filter
    xtick_labels = ['%d' % int(n) if n < 1e10  # assure 1 digit for uniform figure sizes
                   else '' for n in xtick_locs]
    axisHandle.set_xticks(xtick_locs)
    axisHandle.set_xticklabels(xtick_labels)

    # Y ticks
    ytick_locs = [n for n in axisHandle.get_yticks()
                  if n > plt.ylim()[0] and n < plt.ylim()[1]]
    ytick_labels = ['%d' % round(numpy.log10(n)) if n < 1e10  # assure 1 digit for uniform figure sizes
                   else '' for n in ytick_locs]
    axisHandle.set_yticks(ytick_locs)
    axisHandle.set_yticklabels(ytick_labels)
    if upper_bound < 1e3:
        plt.grid(True, axis="y", which="both")

    if legend:
        toolsdivers.legend(loc=0, numpoints=1,
                           fontsize=fontsize * legend_fontsize_scaler())

    if genericsettings.scaling_plots_with_axis_labels:
        plt.xlabel('number of constraints')
        plt.ylabel('log10(# f-evals / dimension)')


def main(dictAlg, html_file_prefix, sorted_algorithms=None, output_dir='ppdata', latex_commands_file=''):
    """From a DataSetList, returns figures showing the scaling: ERT/dim vs dim.
    
    One function and one target per figure.
    
    ``target`` can be a scalar, a list with one element or a 
    ``pproc.TargetValues`` instance with one target.
    
    ``sortedAlgs`` is a list of string-identifies (folder names)
    
    """
    # target becomes a TargetValues "list" with one element
    # for now: use the same target as in ppfigs, could be changed
    target = testbedsettings.current_testbed.ppfigs_ftarget
    target = pproc.TargetValues.cast([target] if numpy.isscalar(target) else target)
    assert isinstance(target, pproc.TargetValues)
    if len(target) != 1:
        raise ValueError('only a single target can be managed in ppfigcons, ' + str(len(target)) + ' targets were given')
    
    funInfos = ppfigparam.read_fun_infos()    

    dictFunc = pproc.dictAlgByFun(dictAlg, agg_cons=True)
    if sorted_algorithms is None:
        sorted_algorithms = sorted(dictAlg.keys())

    plotting_style_list = get_plotting_styles(sorted_algorithms)
    styles = [d.copy() for d in genericsettings.line_styles]  # deep copy
    default_styles = [d.copy() for d in genericsettings.line_styles]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for fname, d in dictFunc.items():
        handles = []
        dictDim = pproc.dictAlgByDim(d)
        for dim, dd in dictDim.items():
            filename = os.path.join(output_dir, 'ppfigcons_%s_d%s' % (fname, dim))
            for plotting_style in plotting_style_list:
                algorithm_list = plotting_style.algorithm_list
                line_styles = [d.copy() for d in default_styles]
                fix_styles(plotting_style, line_styles)  #
                for i, alg in enumerate(algorithm_list):
                    dsl = dd[alg]
                    #Collect data
                    consert = []
                    ert = []
                    consnbsucc = []
                    ynbsucc = []
                    nbsucc = []
                    consmaxevals = []
                    maxevals = []
                    consmedian = []
                    medianfes = []
                    consticks = []
                    successes = []
                    previous_success = False
                    function_ids = []
                    for ds in dsl:
                        assert isinstance(ds, pproc.DataSet)
                        cons = ds.number_of_constraints
                        consticks.append(cons)
                        function_ids.append(ds.funcId)
                        data = generateData(ds, 1e-6)  # TODO: target code is broken, hardcoded for now
                        if data[2] == 0:  # No success
                            previous_success = False
                            consmaxevals.append(cons)
                            maxevals.append(float(data[3]) / dim)
                        if data[2] > 0:
                            if previous_success:
                                successes.append(True)
                            else:
                                successes.append(False)
                            previous_success = True
                            consmedian.append(cons)
                            medianfes.append(data[4] / dim)
                            consert.append(cons)
                            ert.append(float(data[0]) / dim)
                            if data[1] < ratio_nbsucc_to_display:
                                consnbsucc.append(cons)
                                ynbsucc.append(float(data[0]) / dim)
                                nbsucc.append('%d' % data[2])
                    # Draw lines
                    successes.append(False)
                    for icons in range(len(consert)):
                        # plot line only if the current data point in consert
                        # comes just after the next, as monitored by successes
                        if icons < len(consert) - 1 and successes[icons + 1]:
                            tmp = plt.plot(
                                consert[icons:icons + 2],
                                ert[icons:icons + 2], **line_styles[i])
                        else:  # plot remaining single points (some twice)
                            tmp = plt.plot(
                                consert[icons], ert[icons], **line_styles[i])
                        plt.setp(tmp[0], markeredgecolor=plt.getp(tmp[0], 'color'))
                    if consmaxevals:
                        tmp = plt.plot(consmaxevals, maxevals, **line_styles[i])
                        plt.setp(tmp[0], markersize=20, #label=alg,
                                 markeredgecolor=plt.getp(tmp[0], 'color'),
                                 markeredgewidth=1,
                                 markerfacecolor='None', linestyle='None')

                    for i, n in enumerate(nbsucc):
                        plt.text(consnbsucc[i], numpy.array(ynbsucc[i])*1.85, n,
                                 verticalalignment='bottom',
                                 horizontalalignment='center')

                    if not plotting_style.in_background:
                        handles.append(tmp)
                        sorted_algorithms = plotting_style.algorithm_list
                        styles = line_styles

                    # For legend
                    # tmp = plt.plot([], [], label=alg.replace('..' + os.sep, '').strip(os.sep), **line_styles[i])
                    algorithm_name = toolsdivers.str_to_latex(toolsdivers.strip_pathname1(alg))
                    if plotting_style.in_background:
                        algorithm_name = '_' + algorithm_name
                    tmp = plt.plot([], [], label=algorithm_name[:legend_text_max_len], **line_styles[i])
                    plt.setp(tmp[0], markersize=12.,
                             markeredgecolor=plt.getp(tmp[0], 'color'))
                    functions_with_legend = testbedsettings.current_testbed.functions_with_legend
                    isLegend = False
                    if legend:
                        plotLegend(handles)
                    elif set(function_ids).intersection(set(functions_with_legend)) and len(sorted_algorithms) < 1e6:  # 6 elements at most in the boxed legend
                        if (functions_with_legend[0] in function_ids and dim == 2) or (functions_with_legend[-1] in function_ids and dim == 40):
                            # plot legend for the first plot of each objective
                            isLegend = True

            beautify(dim, legend=isLegend, rightlegend=legend)


            refalgentries = bestalg.load_reference_algorithm(testbedsettings.current_testbed.reference_algorithm_filename)

            if refalgentries:        
                refalgdata = []
                dimrefalg = list(df[0] for df in refalgentries if df[1] == f)
                dimrefalg.sort()
                dimrefalg2 = []
                for d in dimrefalg:
                    entry = refalgentries[(d, f)]
                    tmp = entry.detERT(target((f, d)))[0]
                    if numpy.isfinite(tmp):
                        refalgdata.append(float(tmp)/d)
                        dimrefalg2.append(d)
        
                tmp = plt.plot(dimrefalg2, refalgdata, color=refcolor, linewidth=10,
                               marker='d', markersize=25, markeredgecolor=refcolor, zorder=-1
                               #label='best 2009', 
                               )
                handles.append(tmp)
            
            # TODO: is broken
            if 11 < 3 and show_significance: # plot significance-stars
                xstar, ystar = [], []
                dims = sorted(pproc.dictAlgByDim(dictFunc[f]))
                for i, dim in enumerate(dims):
                    datasets = pproc.dictAlgByDim(dictFunc[f])[dim]
                    assert all([len(datasets[ialg]) == 1 for ialg in sorted_algorithms if datasets[ialg]])
                    dsetlist =  [datasets[ialg][0] for ialg in sorted_algorithms if datasets[ialg]]
                    if len(dsetlist) > 1:
                        arzp, arialg = toolsstats.significance_all_best_vs_other(dsetlist, target((f, dim)))
                        if arzp[0][1] * len(dims) < show_significance:
                            ert = dsetlist[arialg[0]].detERT(target((f, dim)))[0]
                            if ert < numpy.inf: 
                                xstar.append(dim)
                                ystar.append(ert/dim)

                plt.plot(xstar, ystar, '*',
                         markerfacecolor='k',  # visible over light colors
                         markeredgecolor='red',  # visible over dark colors
                         markeredgewidth=0.7,
                         markersize=styles[0]['markersize'])
            
            fontSize = getFontSize(funInfos.values())
            plt.gca().set_title('%s, %s-D' % (fname, dim), fontsize=0.9*fontSize)

            
            # bottom labels with #instances and type of targets:
            infotext = ''
            algorithms_with_data = [a for a in dictAlg.keys() if dictAlg[a] != []]

            num_of_instances = []
            for alg in algorithms_with_data:
                try:
                    # display number of instances in data and used targets type:
                    if all(d.instancenumbers == (dictFunc[fname][alg])[0].instancenumbers
                           for d in dictFunc[fname][alg]):  # all the same?
                        num_of_instances.append(len((dictFunc[fname][alg])[0].instancenumbers))
                    else:
                        for d in dictFunc[fname][alg]:
                            num_of_instances.append(len(d.instancenumbers))
                except IndexError:
                    pass
            # issue a warning if number of instances is inconsistant, otherwise
            # display only the present number of instances, i.e. remove copies
            if len(set(num_of_instances)) > 1 and genericsettings.warning_level >= 5:
                warnings.warn('Number of instances inconsistent over all algorithms.')
            num_of_instances = set(num_of_instances)
            for n in num_of_instances:
                infotext += '%d, ' % n

            infotext = infotext.rstrip(', ')
            infotext += ' instances\n'
            infotext += 'target ' + target.label_name() + ': ' + target.label(0)
            infotext += text_infigure_if_constraints()
            plt.text(plt.xlim()[0], plt.ylim()[0],
                     infotext, fontsize=fontsize, horizontalalignment="left",
                     verticalalignment="bottom")

            save_figure(filename, dictAlg[algorithms_with_data[0]][0].algId)

            plt.close()

    htmlFile = os.path.join(output_dir, html_file_prefix + '.html')
    # generate commands in tex file:
    try:
        abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        alg_definitions = []
        alg_definitions_html = ''
        for i in range(len(sorted_algorithms)):
            symb = r'{%s%s}' % (color_to_latex(styles[i]['color']),
                                marker_to_latex(styles[i]['marker']))
            symb_html = '<span style="color:%s;">%s</span>' % (styles[i]['color'], marker_to_html(styles[i]['marker']))
            
            alg_definitions.append((', ' if i > 0 else '') + '%s: %s' % (symb, '\\algorithm' + abc[i % len(abc)]))
            alg_definitions_html += (', ' if i > 0 else '') + '%s: %s' % (symb_html, toolsdivers.str_to_latex(toolsdivers.strip_pathname1(sorted_algorithms[i])))
        toolsdivers.prepend_to_file(latex_commands_file,
                [providecolorsforlatex()]) # needed since the latest change in ACM template
        toolsdivers.prepend_to_file(latex_commands_file,
                                    [  # '\\providecommand{\\bbobppfigsftarget}{\\ensuremath{10^{%s}}}'
                                        #       % target.loglabel(0), # int(numpy.round(numpy.log10(target))),
                                        '\\providecommand{\\bbobppfigconslegend}[1]{',
                                        scaling_figure_caption(),
                                        'Legend: '] + alg_definitions + ['}']
                                    )
        toolsdivers.prepend_to_file(latex_commands_file,
                                    ['\\providecommand{\\bbobECDFslegend}[1]{',
                                     ecdfs_figure_caption(), '}']
                                    )
        
        toolsdivers.replace_in_file(htmlFile, '##bbobppfigconslegend##', scaling_figure_caption(True) + 'Legend: ' + alg_definitions_html)

        if genericsettings.verbose:
            print('Wrote commands and legend to %s' % filename)

        # this is obsolete (however check templates)
        filename = os.path.join(output_dir, 'ppfigcons.tex')
        f = open(filename, 'w')
        f.write('% Do not modify this file: calls to post-processing software'
                + ' will overwrite any modification.\n')
        f.write('Legend: ')
        
        for i in range(0, len(sorted_algorithms)):
            symb = r'{%s%s}' % (color_to_latex(styles[i]['color']),
                                marker_to_latex(styles[i]['marker']))
            f.write((', ' if i > 0 else '') + '%s:%s' % (symb, writeLabels(sorted_algorithms[i])))
        f.close()    
        if genericsettings.verbose:
            print('(obsolete) Wrote legend in %s' % filename)
    except IOError:
        raise


        handles.append(tmp)

        if f in funInfos.keys():
            plt.gca().set_title(funInfos[f], fontsize=fontSize)

        beautify(rightlegend=legend)

        if legend:
            plotLegend(handles)
        else:
            if f in functions_with_legend:
                toolsdivers.legend()

        save_figure(filename, dictAlg[algorithms_with_data[0]][0].algId)

        plt.close()

def providecolorsforlatex():
    """ Provides the dvipsnames colors in pure LaTeX.
    
    Used when the xcolor option of the same name is not available, e.g.
    within the new ACM LaTeX templates.
    
    """
    return r"""% define some COCO/dvipsnames colors because
% ACM style does not allow to use them directly
\definecolor{NavyBlue}{HTML}{000080}
\definecolor{Magenta}{HTML}{FF00FF}
\definecolor{Orange}{HTML}{FFA500}
\definecolor{CornflowerBlue}{HTML}{6495ED}
\definecolor{YellowGreen}{HTML}{9ACD32}
\definecolor{Gray}{HTML}{BEBEBE}
\definecolor{Yellow}{HTML}{FFFF00}
\definecolor{GreenYellow}{HTML}{ADFF2F}
\definecolor{ForestGreen}{HTML}{228B22}
\definecolor{Lavender}{HTML}{FFC0CB}
\definecolor{SkyBlue}{HTML}{87CEEB}
\definecolor{NavyBlue}{HTML}{000080}
\definecolor{Goldenrod}{HTML}{DDF700}
\definecolor{VioletRed}{HTML}{D02090}
\definecolor{CornflowerBlue}{HTML}{6495ED}
\definecolor{LimeGreen}{HTML}{32CD32}
"""
