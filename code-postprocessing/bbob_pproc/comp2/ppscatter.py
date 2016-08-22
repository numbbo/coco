#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Scatter Plots.

For two algorithms, this generates the scatter plot of log(aRT1(df)) vs.
log(aRT0(df)), where aRT0(df) is the aRT of the reference algorithm,
aRT1(df) is the aRT of the algorithm of concern, both for target
precision df.

Different symbols are used for different dimension (see
:py:data:`markers` for the order of the markers, :py:data:`colors` for
the corresponding colors).
The target precisions considered are in :py:data:`targets`: by 
default 46 targets are uniformly spread on the log-scale in
10**[-8:2].

Boxes correspond to the maximum numbers of function evaluations for
each algorithm in each dimension.

"""
from __future__ import absolute_import

"""For two algorithms, aRTs(given target function value) can also be
plotted in a scatter plot (log(aRT0) vs. log(aRT1)), which results in a
very attractive presentation, see the slides of Frank Hutter at
http://www.msr-inria.inria.fr/events-news/first-search-biology-day. The
advantage is that the absolute values do not get lost. The disadvantage
(in our case minor) is that there is an upper limit of data that can be
displayed.

"""
import os
import numpy
import numpy as np
import warnings
from pdb import set_trace
from matplotlib import pyplot as plt
try:
    from matplotlib.transforms import blended_transform_factory as blend
except ImportError:
    # compatibility matplotlib 0.8
    from matplotlib.transforms import blend_xy_sep_transform as blend
from .. import genericsettings, htmldesc, ppfigparam, testbedsettings
from ..ppfig import saveFigure
from .. import toolsdivers
from .. import pproc

dimensions = (2, 3, 5, 10, 20, 40)

# formattings
markersize = 14  # modified in config.py
markersize_addon_beyond_maxevals = -6
linewidth_default = 0  # lines look ugly and are not necessary (anymore), because smaller symbols are used beyond maxevals
linewidth_rld_based = 2  # show lines because only 8 symbols are used
max_evals_line_length = 9  # length away from the diagonal as a factor, line indicates maximal evaluations for each data
offset = 0. #0.02 offset provides a way to move away the box boundaries to display the outer markers fully, clip_on=False is more effective 


def prepare_figure_caption():

    caption_start_fixed = r"""Average running time (\aRT\ in $\log_{10}$ of number of function evaluations)
        of \algorithmB\ ($x$-axis) versus \algorithmA\ ($y$-axis) for $NBTARGETS$ target values
        $\Df \in [NBLOW, NBUP]$ in each dimension on functions #1. """

    caption_start_rlbased = r"""Average running time (\aRT\ in $\log_{10}$ of number of function evaluations)
        of \algorithmA\ ($y$-axis) versus \algorithmB\ ($x$-axis) for $NBTARGETS$ runlength-based target
        function values for budgets between $NBLOW$ and $NBUP$ evaluations.
        Each runlength-based target $f$-value is chosen such that the \aRT{}s of the
        REFERENCE_ALGORITHM artificial algorithm for the given and a slightly easier
        target bracket the reference budget. """

    caption_finish = r"""Markers on the upper or right edge indicate that the respective target
        value was never reached. Markers represent dimension:
        2:{\color{cyan}+},
        3:{\color{green!45!black}$\triangledown$},
        5:{\color{blue}$\star$},
        10:$\circ$,
        20:{\color{red}$\Box$},
        40:{\color{magenta}$\Diamond$}. """


    if testbedsettings.current_testbed.name == testbedsettings.testbed_name_bi:
        # NOTE: no runlength-based targets supported yet
        caption = caption_start_fixed + caption_finish
    elif testbedsettings.current_testbed.name == testbedsettings.testbed_name_single or \
         testbedsettings.current_testbed.name == testbedsettings.testbed_name_cons:
        if genericsettings.runlength_based_targets:
            caption = caption_start_rlbased + caption_finish
        else:
            caption = caption_start_fixed + caption_finish
    else:
        warnings.warn("Current settings do not support ppfigdim caption.")

    return caption


def figure_caption(for_html = False):

    targets = testbedsettings.current_testbed.ppscatter_target_values
    if for_html:
        caption = htmldesc.getValue('##bbobppscatterlegend' +
                                    testbedsettings.current_testbed.scenario + '##')
    else:
        caption = prepare_figure_caption()

    caption = caption.replace('NBTARGETS', str(len(targets)))

    if genericsettings.runlength_based_targets:
        caption = caption.replace('NBLOW', toolsdivers.number_to_latex(targets.label(0)) +
                                           r'\times\DIM' if targets.times_dimension else '')
        caption = caption.replace('NBUP', toolsdivers.number_to_latex(targets.label(-1)) +
                                           r'\times\DIM' if targets.times_dimension else '')
        caption = caption.replace('REFERENCE_ALGORITHM', targets.reference_algorithm)
        caption = caption.replace('REFERENCEALGORITHM', targets.reference_algorithm)
    else:
        caption = caption.replace('NBLOW', toolsdivers.number_to_latex(targets.label(0)))
        caption = caption.replace('NBUP', toolsdivers.number_to_latex(targets.label(-1)))

    return caption


def beautify():
    a = plt.gca()
    a.set_xscale('log')
    a.set_yscale('log')
    #a.set_xlabel('aRT0')
    #a.set_ylabel('aRT1')
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    minbnd = min(xmin, ymin)
    maxbnd = max(xmax, ymax)
    maxbnd = maxbnd ** (1 + 11.*offset/(numpy.log10(float(maxbnd)/minbnd)))
    plt.plot([minbnd, maxbnd], [minbnd, maxbnd], ls='-', color='k')
    plt.plot([10*minbnd, 10*maxbnd], [minbnd, maxbnd], ls=':', color='k')
    plt.plot([100*minbnd, 100*maxbnd], [minbnd, maxbnd], ls=':', color='k')
    plt.plot([minbnd, maxbnd], [10*minbnd, 10*maxbnd], ls=':', color='k')
    plt.plot([minbnd, maxbnd], [100*minbnd, 100*maxbnd], ls=':', color='k')

    plt.xlim(minbnd, maxbnd)
    plt.ylim(minbnd, maxbnd)
    #a.set_aspect(1./a.get_data_ratio())
    a.set_aspect('equal')
    plt.grid(True)
    tmp = a.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    a.set_yticklabels(tmp2)
    a.set_xticklabels(tmp2)
    #for line in a.get_xticklines():# + a.get_yticklines():
    #    plt.setp(line, color='b', marker='o', markersize=10)
    #set_trace()

def main(dsList0, dsList1, outputdir, verbose=True):
    """Generate a scatter plot figure.
    
    TODO: """

    markers = genericsettings.dim_related_markers
    colors = genericsettings.dim_related_colors
    #plt.rc("axes", labelsize=24, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    dictFunc0 = dsList0.dictByFunc()
    dictFunc1 = dsList1.dictByFunc()
    funcs = set(dictFunc0.keys()) & set(dictFunc1.keys())

    targets = testbedsettings.current_testbed.ppscatter_target_values
    if isinstance(targets, pproc.RunlengthBasedTargetValues):
        linewidth = linewidth_rld_based
    else:
        linewidth = linewidth_default

    funInfos = ppfigparam.read_fun_infos()    

    for f in funcs:
        dictDim0 = dictFunc0[f].dictByDim()
        dictDim1 = dictFunc1[f].dictByDim()
        dims = set(dictDim0.keys()) & set(dictDim1.keys())
        #set_trace()

        for i, d in enumerate(dimensions):
            try:
                entry0 = dictDim0[d][0] # should be only one element
                entry1 = dictDim1[d][0] # should be only one element
            except (IndexError, KeyError):
                continue
            if linewidth:  # plot all reliable aRT values as a line
                all_targets = np.array(sorted(set(entry0.target).union(entry1.target), reverse=True))
                assert entry0.detSuccessRates([all_targets[0]]) == 1.0
                assert entry1.detSuccessRates([all_targets[0]]) == 1.0
                all_targets = all_targets[np.where(all_targets <= targets((f, d))[0])[0]]  # 
                xdata_all = np.array(entry0.detERT(all_targets))
                ydata_all = np.array(entry1.detERT(all_targets))
                # idx of reliable targets: last index where success rate >= 1/2 and aRT <= maxevals
                idx = []
                for ari in (np.where(entry0.detSuccessRates(all_targets) >= 0.5)[0], 
                         np.where(entry1.detSuccessRates(all_targets) >= 0.5)[0], 
                         np.where(xdata_all <= max(entry0.maxevals))[0], 
                         np.where(ydata_all <= max(entry1.maxevals))[0]
                        ):
                    if len(ari):
                        idx.append(ari[-1])
                if len(idx) == 4:
                    max_idx = min(idx)
                    ## at least up to the most difficult given target
                    ## idx = max((idx, np.where(all_targets >= targets((f, d))[-1])[0][-1])) 
                    xdata_all = xdata_all[:max_idx + 1]
                    ydata_all = ydata_all[:max_idx + 1]
    
                    idx = (numpy.isfinite(xdata_all)) * (numpy.isfinite(ydata_all))
                    assert idx.all() 
                    if idx.any():
                        plt.plot(xdata_all[idx], ydata_all[idx], colors[i], ls='solid', lw=linewidth, 
                                 # TODO: ls has changed, check whether this works out
                                 clip_on=False)
                
            xdata = numpy.array(entry0.detERT(targets((f, d))))
            ydata = numpy.array(entry1.detERT(targets((f, d))))

            # plot "valid" data, those within maxevals
            idx = np.logical_and(xdata < entry0.mMaxEvals(),
                                 ydata < entry1.mMaxEvals())
            # was:
            #       (numpy.isinf(xdata) == False) *
            #       (numpy.isinf(ydata) == False) *
            #       (xdata < entry0.mMaxEvals()) *
            #       (ydata < entry1.mMaxEvals()))
            if idx.any():
                try:
                    plt.plot(xdata[idx], ydata[idx], ls='',
                             markersize=markersize,
                             marker=markers[i], markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=3, 
                             clip_on=False)
                except KeyError:
                    plt.plot(xdata[idx], ydata[idx], ls='', markersize=markersize,
                             marker='x', markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=3,
                             clip_on=False)
                #try:
                #    plt.scatter(xdata[idx], ydata[idx], s=10, marker=markers[i],
                #            facecolor='None', edgecolor=colors[i], linewidth=3)
                #except ValueError:
                #    set_trace()

            # plot beyond maxevals but finite data
            idx = ((numpy.isinf(xdata) == False) *
                   (numpy.isinf(ydata) == False) *
                   np.logical_or(xdata >= entry0.mMaxEvals(),
                                 ydata >= entry1.mMaxEvals()))
            if idx.any():
                try:
                    plt.plot(xdata[idx], ydata[idx], ls='',
                             markersize=markersize + markersize_addon_beyond_maxevals,
                             marker=markers[i], markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=1,
                             clip_on=False)
                except KeyError:
                    plt.plot(xdata[idx], ydata[idx], ls='', markersize=markersize,
                             marker='x', markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=2,
                             clip_on=False)
            #ax = plt.gca()
            ax = plt.axes()

            # plot data on the right edge
            idx = numpy.isinf(xdata) * (numpy.isinf(ydata) == False)
            if idx.any():
                # This (seems to) transform inf to the figure limits!?
                trans = blend(ax.transAxes, ax.transData)
                #plt.scatter([1.]*numpy.sum(idx), ydata[idx], s=10, marker=markers[i],
                #            facecolor='None', edgecolor=colors[i], linewidth=3,
                #            transform=trans)
                try:
                    plt.plot([1.]*numpy.sum(idx), ydata[idx],
                             markersize=markersize + markersize_addon_beyond_maxevals, ls='',
                             marker=markers[i], markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=1,
                             transform=trans, clip_on=False)
                except KeyError:
                    plt.plot([1.]*numpy.sum(idx), ydata[idx],
                             markersize=markersize, ls='',
                             marker='x', markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=2,
                             transform=trans, clip_on=False)
                #set_trace()

            # plot data on the left edge
            idx = (numpy.isinf(xdata)==False) * numpy.isinf(ydata)
            if idx.any():
                # This (seems to) transform inf to the figure limits!?
                trans = blend(ax.transData, ax.transAxes)
                #    plt.scatter(xdata[idx], [1.-offset]*numpy.sum(idx), s=10, marker=markers[i],
                #                facecolor='None', edgecolor=colors[i], linewidth=3,
                #                transform=trans)
                try:
                    plt.plot(xdata[idx], [1.-offset]*numpy.sum(idx),
                             markersize=markersize + markersize_addon_beyond_maxevals, ls='',
                             marker=markers[i], markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=1,
                             transform=trans, clip_on=False)
                except KeyError:
                    plt.plot(xdata[idx], [1.-offset]*numpy.sum(idx),
                             markersize=markersize, ls='',
                             marker='x', markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=2,
                             transform=trans, clip_on=False)

            # plot data in the top corner
            idx = numpy.isinf(xdata) * numpy.isinf(ydata)
            if idx.any():
                #    plt.scatter(xdata[idx], [1.-offset]*numpy.sum(idx), s=10, marker=markers[i],
                #                facecolor='None', edgecolor=colors[i], linewidth=3,
                #                transform=trans)
                try:
                    plt.plot([1.-offset]*numpy.sum(idx), [1.-offset]*numpy.sum(idx),
                             markersize=markersize + markersize_addon_beyond_maxevals, ls='',
                             marker=markers[i], markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=1,
                             transform=ax.transAxes, clip_on=False)
                except KeyError:
                    plt.plot([1.-offset]*numpy.sum(idx), [1.-offset]*numpy.sum(idx),
                             markersize=markersize, ls='',
                             marker='x', markerfacecolor='None',
                             markeredgecolor=colors[i], markeredgewidth=2,
                             transform=ax.transAxes, clip_on=False)

        targetlabels = targets.labels()
        if isinstance(targets, pproc.RunlengthBasedTargetValues):
            text = (str(len(targetlabels)) + ' target RLs/dim: ' +
                    targetlabels[0] + '..' +
                    targetlabels[len(targetlabels)-1] + '\n')
            text += '   from ' + testbedsettings.current_testbed.best_algorithm_filename
        else:
            text = (str(len(targetlabels)) + ' targets in ' +
                    targetlabels[0] + '..' +
                    targetlabels[len(targetlabels)-1])
        # add number of instances 
        text += '\n'
        num_of_instances_alg0 = []
        num_of_instances_alg1 = []
        for d in dims:
            num_of_instances_alg0.append((dictDim0[d][0]).nbRuns())
            num_of_instances_alg1.append((dictDim1[d][0]).nbRuns())
        # issue a warning if the numbers of instances are inconsistent:
        if (len(set(num_of_instances_alg0)) > 1):
            warnings.warn('Inconsistent numbers of instances over dimensions found for ALG0:\n\
                           found instances %s' % str(num_of_instances_alg0))
        if (len(set(num_of_instances_alg1)) > 1):
            warnings.warn('Inconsistent numbers of instances over dimensions found for ALG1:\n\
                           found instances %s' % str(num_of_instances_alg1))
        if (len(set(num_of_instances_alg0)) == 1 and len(set(num_of_instances_alg0)) == 1):
            text += '%s and %s instances' % (num_of_instances_alg0[0], num_of_instances_alg1[0])
        else:
            for n in num_of_instances_alg0:
                text += '%d, ' % n
            text = text.rstrip(', ')
            text += ' and '
            for n in num_of_instances_alg1:
                text += '%d, ' % n
            text = text.rstrip(', ')
            text += ' instances'
        plt.text(0.01, 0.98, text, horizontalalignment="left",
                 verticalalignment="top", transform=plt.gca().transAxes, size='small')

        beautify()

        for i, d in enumerate(dimensions):
            try:
                entry0 = dictDim0[d][0] # should be only one element
                entry1 = dictDim1[d][0] # should be only one element
            except (IndexError, KeyError):
                continue

            minbnd, maxbnd = plt.xlim()
            plt.plot((entry0.mMaxEvals(), entry0.mMaxEvals()),
                     # (minbnd, entry1.mMaxEvals()), ls='-', color=colors[i],
                     (max([minbnd, entry1.mMaxEvals()/max_evals_line_length]), entry1.mMaxEvals()), ls='-', color=colors[i],
                     zorder=-1)
            plt.plot(# (minbnd, entry0.mMaxEvals()),
                     (max([minbnd, entry0.mMaxEvals()/max_evals_line_length]), entry0.mMaxEvals()),
                     (entry1.mMaxEvals(), entry1.mMaxEvals()), ls='-',
                     color=colors[i], zorder=-1)
            plt.xlim(minbnd, maxbnd)
            plt.ylim(minbnd, maxbnd)
            #Set the boundaries again: they changed due to new plots.

            #plt.axvline(entry0.mMaxEvals(), ls='--', color=colors[i])
            #plt.axhline(entry1.mMaxEvals(), ls='--', color=colors[i])

        fontSize = genericsettings.getFontSize(funInfos.values())
        if f in funInfos.keys():        
            plt.ylabel(funInfos[f], fontsize=fontSize)

        filename = os.path.join(outputdir, 'ppscatter_f%03d' % f)
        saveFigure(filename, verbose=verbose)
        plt.close()

    #plt.rcdefaults()
