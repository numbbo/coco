#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates figure of the bootstrap distribution of (function) evaluations.
    
The main method in this module generates figures of Empirical Cumulative
Distribution Functions of the bootstrap distribution of the (function)
evaluations needed to reach a target divided by the dimension for many
algorithms.

The outputs show the ECDFs of the running times of the simulated runs
divided by dimension for 50 different targets logarithmically uniformly
distributed in [1e−8, 1e2]. The crosses (×) give the median number of
function evaluations of unsuccessful runs divided by dimension.

**Example**

.. plot::
    :width: 50%

    import cocopp

    # Empirical cumulative distribution function of bootstrapped evaluations figure
    ds = cocopp.load(cocopp.bbob.get('2009/BIPOP-CMA-ES'))
    figure()
    cocopp.compall.pprldmany.plot(ds) # must rather call main instead of plot?
    cocopp.compall.pprldmany.beautify()

"""

from __future__ import absolute_import, print_function

import os
import warnings
import collections
import numpy as np
import matplotlib.pyplot as plt
from .. import toolsstats, bestalg, genericsettings, testbedsettings
from .. import pproc as pp  # import dictAlgByDim, dictAlgByFun
from .. import toolsdivers  # strip_pathname, str_to_latex
from .. import pprldistr  # plotECDF, beautifyECDF
from .. import ppfig  # consecutiveNumbers, save_figure, plotUnifLogXMarkers, logxticks
from .. import pptex  # numtotex

PlotType = ppfig.enum('ALG', 'DIM', 'FUNC')

displaybest = True
x_limit = genericsettings.xlimit_pprldmany  # also (re-)set via config
divide_by_dimension = True
annotation_line_end_relative = 1.065  # lines between graph and annotation, was 1.11, see also subplots_adjust
annotation_space_end_relative = 1.21  # figure space end relative to x_limit, space is however determined rather by subplots_adjust(...) below!?
save_zoom = False  # save zoom into left and right part of the figures
perfprofsamplesize = genericsettings.simulated_runlength_bootstrap_sample_size  # number of bootstrap samples drawn for each fct+target in the performance profile
nbperdecade = 1
max_evals_marker_format = ['x', 12.5, 1]  # [symbol, size, edgewidth]
max_evals_single_marker_format = ['+', 14, 1]  # [symbol, size, edgewidth]
max_evals_percentile = 90
budget_cross_always = True  # was False before June 2024
label_fontsize = 15  # was 17
xticks_fontsize = 16
yticks_fontsize = 14
title_fontsize = 20
size_correction_from_n_foreground = 1  # is (re-)set in main and used in plotdata

def text_infigure_if_constraints():
    """to be displayed in the figure corner

    TODO: is a method with no arguments
    because if made a variable, an error is raised
    as it is computed before testbedsettings.current_testbed is instantiated
    in some import
    """
    if testbedsettings.current_testbed.has_constraints:
        w = genericsettings.weight_evaluations_constraints
        _text = ("\nevals = "
                 + ("%sx" % w[0] if w[0] != 1 else "") + "f-evals + "
                 + ("%sx" % w[1] if w[1] != 1 else "") + "g-evals")
    else:
        _text = ""
    return _text

save_figure = True
close_figure = True

# TODO: update the list below which are not relevant anymore

best = ('AMaLGaM IDEA', 'iAMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES', 'IPOP-SEP-CMA-ES',
        'BFGS', 'NELDER (Han)', 'NELDER (Doe)', 'NEWUOA', 'full NEWUOA', 'GLOBAL', 'MCS (Neum)',
        'DIRECT', 'DASA', 'POEMS', 'Cauchy EDA', 'Monte Carlo')

best2 = (
'AMaLGaM IDEA', 'iAMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES', 'IPOP-SEP-CMA-ES', 'BFGS', 'NEWUOA',
'GLOBAL')

eseda = (
'AMaLGaM IDEA', 'iAMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES', 'IPOP-SEP-CMA-ES', '(1+1)-CMA-ES',
'(1+1)-ES')

ESs = ('BIPOP-CMA-ES', 'IPOP-SEP-CMA-ES', '(1+1)-CMA-ES', '(1+1)-ES', 'BIPOP-ES')

bestnoisy = ()

bestbest = ('BIPOP-CMA-ES', 'NEWUOA', 'GLOBAL', 'NELDER (Doe)')
nikos = ('AMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES', '(1+1)-CMA-ES', 'G3-PCX', 'NEWUOA',
         'Monte Carlo', 'NELDER (Han)', 'NELDER (Doe)', 'GLOBAL', 'MCS (Neum)')
nikos = ('AMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES',
         '(1+1)-CMA-ES', '(1+1)-ES', 'IPOP-SEP-CMA-ES', 'BIPOP-ES',
         'NEWUOA',
         'NELDER (Doe)', 'BFGS', 'Monte Carlo')

nikos40D = ('AMaLGaM IDEA', 'iAMaLGaM IDEA', 'BIPOP-CMA-ES',
            '(1+1)-CMA-ES', '(1+1)-ES', 'IPOP-SEP-CMA-ES',
            'NEWUOA', 'NELDER (Han)', 'BFGS', 'Monte Carlo')

# three groups which include all algorithms:
GA = ('DE-PSO', '(1+1)-ES', 'PSO_Bounds', 'DASA', 'G3-PCX', 'simple GA', 'POEMS', 'Monte Carlo')  # 7+1

classics = ('BFGS', 'NELDER (Han)', 'NELDER (Doe)', 'NEWUOA', 'full NEWUOA', 'DIRECT', 'LSfminbnd',
            'LSstep', 'Rosenbrock', 'GLOBAL', 'SNOBFIT', 'MCS (Neum)', 'adaptive SPSA', 'Monte Carlo')  # 13+1

EDA = ('BIPOP-CMA-ES', '(1+1)-CMA-ES', 'VNS (Garcia)', 'EDA-PSO', 'IPOP-SEP-CMA-ES', 'AMaLGaM IDEA',
       'iAMaLGaM IDEA', 'Cauchy EDA', 'BayEDAcG', 'MA-LS-Chain', 'Monte Carlo')  # 10+1

# groups according to the talks
petr = ('DIRECT', 'LSfminbnd', 'LSstep', 'Rosenbrock', 'G3-PCX', 'Cauchy EDA', 'Monte Carlo')
TAO = ('BFGS', 'NELDER (Han)', 'NEWUOA', 'full NEWUOA', 'BIPOP-CMA-ES', 'IPOP-SEP-CMA-ES',
       '(1+1)-CMA-ES', '(1+1)-ES', 'simple GA', 'Monte Carlo')
TAOp = TAO + ('NELDER (Doe)',)
MC = ('Monte Carlo',)

third = ('POEMS', 'VNS (Garcia)', 'DE-PSO', 'EDA-PSO', 'PSO_Bounds', 'PSO', 'AMaLGaM IDEA', 'iAMaLGaM IDEA',
         'MA-LS-Chain', 'DASA', 'BayEDAcG')

funi = [1, 2] + list(range(5, 15))  # 2 is paired Ellipsoid
funilipschitz = [1] + [5, 6] + list(range(8, 13)) + [14]  # + [13]  #13=sharp ridge, 7=step-ellipsoid
fmulti = [3, 4] + list(range(15, 25))  # 3 = paired Rastrigin
funisep = [1, 2, 5]

# input parameter settings
show_algorithms = eseda + ('BFGS',)  # ()==all
# show_algorithms = ('IPOP-SEP-CMA-ES', 'IPOP-CMA-ES', 'BIPOP-CMA-ES',)
# show_algorithms = ('IPOP-SEP-CMA-ES', 'IPOP-CMA-ES', 'BIPOP-CMA-ES',
# 'avg NEWUOA', 'NEWUOA', 'full NEWUOA', 'BFGS', 'MCS (Neum)', 'GLOBAL', 'NELDER (Han)',
# 'NELDER (Doe)', 'Monte Carlo') # ()==all
show_algorithms = ()  # could be one of the list above

# '-'     solid line style
# '--'    dashed line style
# '-.'    dash-dot line style
# ':'     dotted line style
# '.'     point marker
# ','     pixel marker
# 'o'     circle marker
# 'v'     triangle_down marker
# '^'     triangle_up marker
# '<'     triangle_left marker
# '>'     triangle_right marker
# '1'     tri_down marker
# '2'     tri_up marker
# '3'     tri_left marker
# '4'     tri_right marker
# 's'     square marker
# 'p'     pentagon marker
# '*'     star marker
# 'h'     hexagon1 marker
# 'H'     hexagon2 marker
# '+'     plus marker
# 'x'     x marker
# 'D'     diamond marker
# 'd'     thin_diamond marker
# '|'     vline marker
# '_'     hline marker


def plt_plot(*args, **kwargs):
    return plt.plot(*args, clip_on=False, **kwargs)


def beautify():
    """Customize figure presentation."""

    # plt.xscale('log') # Does not work with matplotlib 0.91.2
    a = plt.gca()
    a.set_xscale('log')
    # Tick label handling
    plt.xlim(1e-0)

    global divide_by_dimension
    if divide_by_dimension:
        plt.xlabel('log10(%s / dimension)' % testbedsettings.current_testbed.string_evals_legend, fontsize=label_fontsize)
    else:
        plt.xlabel('log10(%s)' % testbedsettings.current_testbed.string_evals_legend, fontsize=label_fontsize)
    plt.ylabel('Fraction of function,target pairs', fontsize=label_fontsize)
    ppfig.logxticks()
    plt.xticks(fontsize=xticks_fontsize)  # the "original" size is for some reason too large
    pprldistr.beautifyECDF()
    plt.yticks(fontsize=yticks_fontsize)
    plt.ylim(-0.0, 1.0)


def plotdata(data, maxval=None, maxevals=None, CrE=0., maxevals2=None, **kwargs):
    """Draw a normalized ECDF. What means normalized?
    
    :param seq data: data set, a 1-D ndarray of runlengths
    :param float maxval: right-most value to be displayed, will use the
                         largest non-inf, non-nan value in data if not
                         provided
    :param seq maxevals: if provided, will plot the median of this
                         sequence as a single cross marker
    :param float CrE: Crafting effort the data will be multiplied by
                      the exponential of this value
    :param maxevals2: a single value or values to be plotted as median(maxevals2)
                      with the same marker as maxevals
    :param kwargs: optional arguments provided to plot function.
    
    """
    # Expect data to be a ndarray.
    x = data[np.isnan(data) == False]  # Take away the nans
    nn = len(x)

    x = x[np.isinf(x) == False]  # Take away the infs
    n = len(x)

    x = np.exp(CrE) * x  # correction by crafting effort CrE

    if n == 0:
        # res = plt.plot((1., ), (0., ), **kwargs)
        res = pprldistr.plotECDF(np.array((1.,)), n=np.inf, **kwargs)
        maxval = np.inf  # trick to plot the cross later if maxevals
    else:
        dictx = {}  # number of appearances of each value in x
        for i in x:
            dictx[i] = dictx.get(i, 0) + 1

        x = np.array(sorted(dictx))  # x is not a multiset anymore
        y = np.cumsum(list(dictx[i] for i in x))  # cumsum of size of y-steps (nb of appearences)
        idx = sum(x <= x_limit ** annotation_space_end_relative) - 1
        y_last, x_last = y[idx] / float(nn), x[idx]
        if maxval is None:
            maxval = max(x)
        end = np.sum(x <= maxval)
        x = x[:end]
        y = y[:end]

        try:  # plot the very last point outside of the "normal" plotting area
            c = kwargs['color']
            plt_plot(x_last, y_last, '.', color=c, markeredgecolor=c, markersize=4)
        except:
            pass
        x2 = np.hstack([np.repeat(x, 2), maxval])  # repeat x-values for each step in the cdf
        y2 = np.hstack([0.0, np.repeat(y / float(nn), 2)])

        res = ppfig.plotUnifLogXMarkers(x2, y2, nbperdecade * 3 / np.log10(maxval),
                                        logscale=False, clip_on=False, **kwargs)
        # res = plotUnifLogXMarkers(x2, y2, nbperdecade, logscale=False, **kwargs)

        for maxeval_, format in ((maxevals, max_evals_marker_format),
                                 (maxevals2, max_evals_single_marker_format)):
            if not maxeval_:  # cover the case where maxevals is None or empty
                continue
            x3 = np.median(maxeval_)  # change it only here
            if ((budget_cross_always or x3 <= maxval) and
                # np.any(x2 <= x3) and   # maxval < median(maxevals)
                not plt.getp(res[-1], 'label').startswith('best')
                ): # TODO: HACK for not considering a "best" algorithm line
                # Setting y3
                if n == 0:
                    y3 = 0
                else:
                    try:
                        y3 = y2[x2 <= x3][-1]  # find right y-value for x3==median(maxevals)
                    except IndexError:  # median(maxevals) is smaller than any data, can only happen because of CrE?
                        y3 = y2[0]
                h = plt.plot((x3,), (y3,),
                            marker=format[0],
                            markersize=format[1] * size_correction_from_n_foreground**0.85,
                            markeredgewidth=format[2],
                            # marker='x', markersize=24, markeredgewidth=3, 
                            markeredgecolor=plt.getp(res[0], 'color'),
                            ls=plt.getp(res[0], 'linestyle'),
                            color=plt.getp(res[0], 'color'),
                            # zorder=1.6   # zorder=0;1;1.5 is behind the grid lines, 2 covers other lines, 1.6 is between
                            )
                # h.extend(res)
                # res = h  # so the last element in res still has the label.

                # Only take sequences for x and y!

    return res


def plotLegend(handles, maxval):
    """Display right-side legend.
    
    :param float maxval: rightmost x boundary
    :returns: list of (ordered) labels and handles.

    The figure is stopped at maxval (upper x-bound), and the graphs in
    the figure are prolonged with straight lines to the right to connect
    with labels of the graphs (uniformly spread out vertically). The
    order of the graphs at the upper x-bound line give the order of the
    labels, in case of ties, the best is the graph for which the x-value
    of the first step (from the right) is smallest.
    
    The annotation string is stripped from preceeding pathnames. 

    """
    reslabels = []
    reshandles = []
    ys = {}
    lh = 0

    def label_length(label_list):
        """Return either `genericsettings.len_of_names_in_pprldmany_legend`
        or the minimal length for the names in `label_list` so that all
        names are different in 2 or more characters. At least 9 characters
        are displayed unless ``0 <
        genericsettings.len_of_names_in_pprldmany_legend < 9``. This
        function is used for the algorithm names legend.
        """
        if genericsettings.len_of_names_in_pprldmany_legend:
            return genericsettings.len_of_names_in_pprldmany_legend

        maxLength = max(len(i) for i in label_list)
        numberOfCharacters = 7  # == len("best 2009") - 2, we add 2 later
        firstPart = [i[:numberOfCharacters] for i in label_list]
        while (len(firstPart) > len(set(firstPart)) and numberOfCharacters <= maxLength):
            numberOfCharacters += 1
            firstPart = [i[:numberOfCharacters] for i in label_list]

        return min(numberOfCharacters + 2, maxLength)

    handles_with_legend = [h for h in handles if not plt.getp(h[-1], 'label').startswith('_line')]
    handles_with_legend = [h for h in handles_with_legend  # fix for matplotlib since v 3.5.0
                           if not plt.getp(h[-1], 'label').startswith('_child')]
    label_list = [toolsdivers.strip_pathname1(plt.getp(h[-1], 'label')) for h in handles_with_legend]
    numberOfCharacters = label_length(label_list)
    for h in handles_with_legend:
        x2 = []
        y2 = []
        for i in h:
            x2.append(plt.getp(i, "xdata"))
            y2.append(plt.getp(i, "ydata"))

        x2 = np.array(np.hstack(x2))
        y2 = np.array(np.hstack(y2))
        tmp = np.argsort(x2)
        x2 = x2[tmp]
        y2 = y2[tmp]

        h = h[-1]  # we expect the label to be in the last element of h
        tmp = (x2 <= maxval)
        try:
            x2bis = x2[y2 < y2[tmp][-1]][-1]
        except IndexError:  # there is no data with a y smaller than max(y)
            x2bis = 0.
        ys.setdefault(y2[tmp][-1], {}).setdefault(x2bis, []).append(h)
        lh += 1

    if len(show_algorithms) > 0:
        lh = min(lh, len(show_algorithms))
    if lh <= 1:
        lh = 2
    fontsize_interp = (30.0 - lh) / 10.0
    if fontsize_interp > 1.0:
        fontsize_interp = 1.0
    if fontsize_interp < 0.0:
        fontsize_interp = 0.0
    fontsize_bounds = genericsettings.minmax_algorithm_fontsize
    fontsize = fontsize_bounds[0] + fontsize_interp * (fontsize_bounds[-1] - fontsize_bounds[0])
    i = 0  # loop over the elements of ys
    for j in sorted(ys.keys()):
        for k in reversed(sorted(ys[j].keys())):
            # enforce "best" algorithm comes first in case of equality
            tmp = []
            for h in ys[j][k]:
                if "best" in plt.getp(h, 'label'):
                    tmp.insert(0, h)
                else:
                    tmp.append(h)
            tmp.reverse()
            ys[j][k] = tmp

            for h in ys[j][k]:
                if (not plt.getp(h, 'label').startswith('_line') and
                        (len(show_algorithms) == 0 or
                                 plt.getp(h, 'label') in show_algorithms)):
                    y = 0.02 + i * 0.96 / (lh - 1)
                    tmp = {}
                    for attr in ('lw', 'linestyle', 'marker',
                                 'markeredgewidth', 'markerfacecolor',
                                 'markeredgecolor', 'markersize', 'zorder'):
                        tmp[attr] = plt.getp(h, attr)
                    tmp['color'] = tmp['markeredgecolor']
                    legx = maxval ** annotation_line_end_relative
                    reshandles.extend(plt_plot((maxval, legx), (j, y), **tmp))
                    reshandles.append(
                        plt.text(maxval ** (0.02 + annotation_line_end_relative), y,
                                 toolsdivers.str_to_latex(
                                     toolsdivers.strip_pathname1(plt.getp(h, 'label'))[:numberOfCharacters]),
                                 horizontalalignment="left",
                                 verticalalignment="center",
                                 fontsize=fontsize))
                    reslabels.append(plt.getp(h, 'label'))
                    # set_trace()
                    i += 1

    # plt.axvline(x=maxval, color='k') # Not as efficient?
    reshandles.append(plt_plot((maxval, maxval), (0., 1.), color='k'))
    reslabels.reverse()
    plt.xlim(None, maxval)
    return reslabels, reshandles


def plot(dsList, targets=None, craftingeffort=0., **kwargs):
    """This function is obsolete?
    Generates a graph of the run length distribution of an algorithm.

    We display the empirical cumulative distribution function ECDF of
    the bootstrapped distribution of the runlength for an algorithm
    (in number of function evaluations) to reach the target functions 
    value :py:data:`targets`.

    :param DataSetList dsList: data set for one algorithm
    :param seq targets: target function values
    :param float crafting effort: the data will be multiplied by the
                                  exponential of this value
    :param dict kwargs: additional parameters provided to plot function.
    
    :returns: handles

    """
    if targets is None:
        targets = testbedsettings.current_testbed.pprldmany_target_values
    try:
        if np.min(targets) >= 1:
            ValueError(
                'smallest target f-value is not smaller than one, use ``pproc.TargetValues(targets)`` to prevent this error')
        targets = pp.TargetValues(targets)
    except TypeError:
        pass
    res = []
    assert len(pp.DataSetList(dsList).dictByDim()) == 1  # We never integrate over dimensions...
    data = []
    maxevals = []
    for entry in dsList:
        for t in targets((entry.funcId, entry.dim)):
            divisor = entry.dim if divide_by_dimension else 1
            x = [np.inf] * perfprofsamplesize
            runlengthunsucc = []
            evals = entry.detEvals([t])[0]
            runlengthsucc = evals[np.isnan(evals) == False] / divisor
            if testbedsettings.current_testbed.has_constraints:
                # maxevals is inconsistent in that case
                maxevals_column = entry.maxfgevals
            else:
                maxevals_column = entry.maxevals
            runlengthunsucc = maxevals_column[np.isnan(evals)] / divisor
            if len(runlengthsucc) > 0:  # else x == [inf, inf,...]
                if testbedsettings.current_testbed.instances_are_uniform:
                    x = toolsstats.drawSP(runlengthsucc, runlengthunsucc,
                                          percentiles=[50],
                                          samplesize=perfprofsamplesize)[1]
                else:
                    nruns = len(runlengthsucc) + len(runlengthunsucc)
                    if perfprofsamplesize % nruns:
                        warnings.warn("without simulated restarts nbsamples=%d"
                                      " should be a multiple of nbruns=%d"
                                      % (perfprofsamplesize, nruns))
                    idx = toolsstats.randint_derandomized(nruns, size=perfprofsamplesize)
                    x = np.hstack((runlengthsucc, len(runlengthunsucc) * [np.inf]))[idx]
            data.extend(x)
            maxevals.extend(runlengthunsucc)

    # Display data
    data = np.array(data)
    data = data[np.isnan(data) == False]  # Take away the nans
    n = len(data)
    data = data[np.isinf(data) == False]  # Take away the infs
    # data = data[data <= maxval] # Take away rightmost data
    data = np.exp(craftingeffort) * data  # correction by crafting effort CrE
    if len(data) == 0:  # data is empty.
        res = pprldistr.plotECDF(np.array((1.,)), n=np.inf, **kwargs)
    else:
        res = pprldistr.plotECDF(np.array(data), n=n, **kwargs)
        # plotdata(np.array(data), x_limit, maxevals,
        #                    CrE=0., **kwargs)
    if maxevals:  # Should cover the case where maxevals is None or empty
        x3 = np.median(maxevals)
        if np.any(data > x3):
            y3 = float(np.sum(data <= x3)) / n
            h = plt_plot((x3,), (y3,), marker='x', markersize=24, markeredgewidth=3,
                         markeredgecolor=plt.getp(res[0], 'color'),
                         ls='', color=plt.getp(res[0], 'color'))
            h.extend(res)
            res = h  # so the last element in res still has the label.
    return res


def all_single_functions(dict_alg, is_single_algorithm, sorted_algs=None,
                         output_dir='.', parent_html_file_name=None, settings=genericsettings):
    single_fct_output_dir = (output_dir.rstrip(os.sep) + os.sep +
                             'pprldmany-single-functions'
                             # + os.sep + ('f%03d' % fg)
                             )
    if not os.path.exists(single_fct_output_dir):
        os.makedirs(single_fct_output_dir)

    if is_single_algorithm:
        main(dict_alg,
             order=sorted_algs,
             outputdir=single_fct_output_dir,
             info='',
             parentHtmlFileName=parent_html_file_name,
             plotType=PlotType.DIM,
             settings=settings)

        dictFG = pp.dictAlgByFuncGroup(dict_alg)
        for fg, entries in sorted(dictFG.items()):
            main(entries,
                 order=sorted_algs,
                 outputdir=single_fct_output_dir,
                 info='%s' % (fg),
                 parentHtmlFileName=parent_html_file_name,
                 plotType=PlotType.DIM,
                 settings=settings)

    dictFG = pp.dictAlgByFun(dict_alg)
    for fg, tempDictAlg in sorted(dictFG.items()):

        if is_single_algorithm:
            main(tempDictAlg,
                 order=sorted_algs,
                 outputdir=single_fct_output_dir,
                 info='f%03d' % (fg),
                 parentHtmlFileName=parent_html_file_name,
                 plotType=PlotType.DIM,
                 settings=settings)
        else:
            dictDim = pp.dictAlgByDim(tempDictAlg)
            dims = sorted(dictDim)
            for i, d in enumerate(dims):
                entries = dictDim[d]
                main(entries,
                     order=sorted_algs,
                     outputdir=single_fct_output_dir,
                     info='f%03d_%02dD' % (fg, d),
                     parentHtmlFileName=parent_html_file_name,
                     settings=settings)

            ppfig.save_single_functions_html(
                os.path.join(single_fct_output_dir, genericsettings.pprldmany_file_name),
                '',  # algorithms names are clearly visible in the figure
                dimensions=dims,
                htmlPage=ppfig.HtmlPage.NON_SPECIFIED,
                parentFileName='../%s' % parent_html_file_name if parent_html_file_name else None,
                header=ppfig.pprldmany_per_func_dim_header
            )

    if is_single_algorithm:
        functionGroups = dict_alg[list(dict_alg.keys())[0]].getFuncGroups()

        dictDim = pp.dictAlgByDim(dict_alg)
        dims = sorted(dictDim)
        for i, d in enumerate(dims):
            tempDictAlg = dictDim[d]
            next_dim = dims[i+1] if i + 1 < len(dims) else dims[0]
            dictFG = pp.dictAlgByFuncGroup(tempDictAlg)
            for fg, entries in sorted(dictFG.items()):
                main(entries,
                     order=sorted_algs,
                     outputdir=single_fct_output_dir,
                     info='gr_%s_%02dD' % (fg, d),
                     parentHtmlFileName=parent_html_file_name,
                     plotType=PlotType.FUNC,
                     settings=settings)

        ppfig.save_single_functions_html(
            os.path.join(single_fct_output_dir, genericsettings.pprldmany_group_file_name),
            '',
            dimensions=dims,
            htmlPage=ppfig.HtmlPage.PPRLDMANY_BY_GROUP,
            function_groups=functionGroups,
            parentFileName='../%s' % parent_html_file_name if parent_html_file_name else None
        )


def main(dictAlg, order=None, outputdir='.', info='default',
         dimension=None, parentHtmlFileName=None, plotType=PlotType.ALG, settings = genericsettings):
    """Generates a figure showing the performance of algorithms.

    From a dictionary of :py:class:`DataSetList` sorted by algorithms,
    generates the cumulative distribution function of the bootstrap
    distribution of evaluations for algorithms on multiple functions for
    multiple targets altogether.

    :param dict dictAlg: dictionary of :py:class:`DataSetList` instances
                         one instance is equivalent to one algorithm,
    :param list targets: target function values
    :param list order: sorted list of keys to dictAlg for plotting order
    :param str outputdir: output directory
    :param str info: output file name suffix
    :param str parentHtmlFileName: defines the parent html page 

    """
    global divide_by_dimension  # not fully implemented/tested yet
    global size_correction_from_n_foreground
    size_correction_from_n_foreground = 1  # reset for reference alg here, later set depending on n

    tmp = pp.dictAlgByDim(dictAlg)
    algorithms_with_data = [a for a in dictAlg.keys() if dictAlg[a] != []]
    # algorithms_with_data.sort()  # dictAlg now is an OrderedDict, hence sorting isn't desired

    if len(algorithms_with_data) > 1 and len(tmp) != 1 and dimension is None:
        raise ValueError('We never integrate over dimension for more than one algorithm.')
    if dimension is not None:
        if dimension not in tmp.keys():
            raise ValueError('dimension %d not in dictAlg dimensions %s'
                             % (dimension, str(tmp.keys())))
        tmp = {dimension: tmp[dimension]}
    dimList = list(tmp.keys())

    # The sort order will be defined inside this function.    
    if plotType == PlotType.DIM:
        order = []

    # Collect data
    # Crafting effort correction: should we consider any?
    CrEperAlg = {}
    for alg in algorithms_with_data:
        CrE = 0.
        if 1 < 3 and str(dictAlg[alg][0].algId).startswith('GLOBAL') and (
                not dictAlg[alg][0].indexFiles or '_pal' in dictAlg[alg][0].indexFiles[0]):
            tmp = dictAlg[alg].dictByNoise()
            assert len(tmp.keys()) == 1
            if list(tmp.keys())[0] == 'noiselessall':
                CrE = 0.5117
            elif list(tmp.keys())[0] == 'nzall':
                CrE = 0.6572
        if plotType == PlotType.DIM:
            for dim in dimList:
                keyValue = '%d-D' % (dim)
                CrEperAlg[keyValue] = CrE
        elif plotType == PlotType.FUNC:
            tmp = pp.dictAlgByFun(dictAlg)
            for f, dictAlgperFunc in tmp.items():
                keyValue = 'f%d' % (f)
                CrEperAlg[keyValue] = CrE
        else:
            CrEperAlg[alg] = CrE
        if CrE != 0.0:
            print('Crafting effort for', alg, 'is', CrE)

    dictData = {}  # list of (ert per function) per algorithm
    dictMaxEvals = collections.defaultdict(list)  # sum(maxevals) / max(1, #success) per instance
    dictMaxEvals2 = collections.defaultdict(list)  # max of successf and unsucc 90%tile runtime over all instances

    # funcsolved = [set()] * len(targets) # number of functions solved per target
    xbest = []
    maxevalsbest = []
    target_values = testbedsettings.current_testbed.pprldmany_target_values

    dictDimList = pp.dictAlgByDim(dictAlg)
    dims = sorted(dictDimList)
    for i, dim in enumerate(dims):
        divisor = dim if divide_by_dimension else 1

        dictDim = dictDimList[dim]
        dictFunc = pp.dictAlgByFun(dictDim)

        # determine a good samplesize
        run_numbers = []
        for dsl in dictDim.values():
            run_numbers.extend([ds.nbRuns() for ds in dsl])
        if genericsettings.in_a_hurry >= 100:
            samplesize = max(run_numbers)
        else:
            try: lcm = np.lcm.reduce(run_numbers)  # lowest common multiplier
            except: lcm = max(run_numbers)  # fallback for old numpy versions
            # slight abuse of bootstrap_sample_size to avoid a huge number
            samplesize = min((int(genericsettings.simulated_runlength_bootstrap_sample_size), lcm))
        if testbedsettings.current_testbed.instances_are_uniform:
            samplesize = max((int(genericsettings.simulated_runlength_bootstrap_sample_size),
                              samplesize))  # maybe more bootstrapping with unsuccessful trials
        if samplesize > 1e4:
            warntxt = ("Sample size equals {} which may take very long. "
                       "This is likely to be unintended, hence a bug.".format(samplesize))
            warnings.warn(warntxt)
        if not isinstance(samplesize, int):
            warntxt = ("samplesize={} was of type {}. This must be considered a bug."
                       "\n run_numbers={} \n lcm={}"
                       "\n genericsettings.simulated_runlength_bootstrap_sample_size={}".format(
                           samplesize,
                           type(samplesize),
                           run_numbers,
                           lcm if 'lcm' in locals() else '"not computed"',
                           genericsettings.simulated_runlength_bootstrap_sample_size))
            warnings.warn(warntxt)
            samplesize = int(samplesize)
        for f, dictAlgperFunc in sorted(dictFunc.items()):
            # print(target_values((f, dim)))
            targets = target_values((f, dim))
            for j, t in enumerate(targets):
                # for j, t in enumerate(testbedsettings.current_testbed.ecdf_target_values(1e2, f)):
                # funcsolved[j].add(f)

                for alg in algorithms_with_data:
                    x = [np.inf] * samplesize
                    runlengthunsucc = []  # this should be a DataSet method
                    try:
                        entry = dictAlgperFunc[alg][0]  # one element per fun and per dim.
                        evals = entry.detEvals([t])[0]
                        assert entry.dim == dim
                        runlengthsucc = evals[np.isnan(evals) == False] / divisor
                        if testbedsettings.current_testbed.has_constraints:
                            # maxevals is inconsistent in that case
                            maxevals_column = entry.maxfgevals
                        else:
                            maxevals_column = entry.maxevals
                        runlengthunsucc = maxevals_column[np.isnan(evals)] / divisor
                        if len(runlengthsucc) > 0:  # else x == [inf, inf,...]
                            if testbedsettings.current_testbed.instances_are_uniform:
                                x = toolsstats.drawSP(runlengthsucc, runlengthunsucc,
                                                      percentiles=[50],
                                                      samplesize=samplesize)[1]
                            else:
                                nruns = len(runlengthsucc) + len(runlengthunsucc)
                                if samplesize % nruns:
                                    warnings.warn("without simulated restarts nbsamples=%d"
                                                  " should be a multiple of nbruns=%d"
                                                  % (samplesize, nruns))
                                idx = toolsstats.randint_derandomized(nruns, size=samplesize)
                                x = np.hstack((runlengthsucc, len(runlengthunsucc) * [np.inf]))[idx]
                    except (KeyError, IndexError):
                        # set_trace()
                        warntxt = ('Data for algorithm %s on function %d in %d-D '
                                   % (alg, f, dim)
                                   + 'are missing.\n')
                        warnings.warn(warntxt)

                    keyValue = alg
                    if plotType == PlotType.DIM:
                        keyValue = '%d-D' % (dim)
                        if keyValue not in order:
                            order.append(keyValue)
                    elif plotType == PlotType.FUNC:
                        keyValue = 'f%d' % (f)
                    dictData.setdefault(keyValue, []).extend(x)
                    # dictMaxEvals.setdefault(keyValue, []).extend(runlengthunsucc)
                    if len(runlengthunsucc) and t == min(targets):  # only once, not for each target as it was before June 2024
                        def percentile(vals, which=max_evals_percentile):
                            return toolsstats.prctile(vals, [which])[0]
                        if 1 < 3:
                            if 'entry' in locals():  # entry was assigned under a try
                                dictMaxEvals[keyValue].append(percentile(entry.budget_effective_estimates.values()))
                        if 1 < 3:
                            maxmed = percentile(runlengthunsucc)
                            if len(runlengthsucc):
                                maxmed = max((maxmed, percentile(runlengthsucc)))
                            dictMaxEvals2[keyValue].append(maxmed)

            displaybest = plotType == PlotType.ALG
            if displaybest:
                # set_trace()
                refalgentries = bestalg.load_reference_algorithm(testbedsettings.current_testbed.reference_algorithm_filename)

                if not refalgentries:
                    displaybest = False
                else:
                    refalgentry = refalgentries[(dim, f)]
                    refalgevals = refalgentry.detEvals(target_values((f, dim)))
                    # print(refalgevals)
                    for j in range(len(refalgevals[0])):
                        if refalgevals[1][j]:
                            evals = refalgevals[0][j]
                            # set_trace()
                            assert dim == refalgentry.dim
                            runlengthsucc = evals[np.isnan(evals) == False] / divisor
                            runlengthunsucc = refalgentry.maxevals[refalgevals[1][j]][np.isnan(evals)] / divisor
                            x = toolsstats.drawSP(runlengthsucc, runlengthunsucc,
                                                  percentiles=[50],
                                                  samplesize=samplesize)[1]
                        else:
                            x = samplesize * [np.inf]
                            runlengthunsucc = []
                        xbest.extend(x)
                        maxevalsbest.extend(runlengthunsucc)

    if order is None:
        order = dictData.keys()

    # Display data
    lines = []
    if displaybest:
        args = {'label': testbedsettings.current_testbed.reference_algorithm_displayname,
                'zorder': -1}
        args.update(genericsettings.reference_algorithm_styles)
        lines.append(plotdata(np.array(xbest), x_limit, maxevalsbest,
                              CrE=0., **args))

    def algname_to_label(algname, dirname=None):
        """to be extended to become generally useful"""
        if isinstance(algname, (tuple, list)):  # not sure this is needed
            return ' '.join([str(name) for name in algname])
        return str(algname)

    plotting_style_list = ppfig.get_plotting_styles(order)
    try:  # set n_foreground to tweak line sizes later
        bg_algs = []
        for v in genericsettings.background.values():
            bg_algs.extend(v)
        n_foreground = len([a for a in dictData if a not in bg_algs])
    except: n_foreground = len(dictData)
    size_correction_from_n_foreground = 1.8 - 1 * min((1, (n_foreground - 1) / 30))

    styles = [s.copy() for s in genericsettings.line_styles]  # list of line/marker style dicts
    if styles[0]['color'] == '#000080':  # fix old styles marker size
        for s in styles:
            try: s['markersize'] /= 3.3  # apparently the appearance of sizes changed
            except: pass
    for plotting_style in plotting_style_list:
        for i, alg in enumerate(plotting_style.algorithm_list):
            try:
                data = dictData[alg]
            except KeyError:
                continue

            args = dict(styles[i % len(styles)])  # kw-args passed to plot
            args.setdefault('markeredgewidth', 1.0)  # was: 1.5
            args.setdefault('markerfacecolor', 'None')  # transparent
            args.setdefault('markeredgecolor', styles[i % len(styles)]['color'])
            args.setdefault('markersize', 9)  # was: 12
            args.setdefault('linewidth', 1)
            args['markersize'] *= genericsettings.marker_size_multiplier
            args['markersize'] *= size_correction_from_n_foreground
            args['linewidth'] *= size_correction_from_n_foreground
            args['linewidth'] *= 1 + max((-0.5, min((1.5,  # larger zorder creates thicker lines below others
                                     2 - args.get('zorder', 2)))))
            args['label'] = algname_to_label(alg)

            if plotType == PlotType.DIM:  # different lines are different dimensions
                args['marker'] = genericsettings.dim_related_markers[i]
                args['markeredgecolor'] = genericsettings.dim_related_colors[i]
                args['color'] = genericsettings.dim_related_colors[i]

                # args['markevery'] = perfprofsamplesize # option available in latest version of matplotlib
                # elif len(show_algorithms) > 0:
                # args['color'] = 'wheat'
                # args['linestyle'] = '-'
                # args['zorder'] = -1
            # plotdata calls pprldistr.plotECDF which calls ppfig.plotUnifLog... which does the work

            args.update(plotting_style.pprldmany_styles)  # no idea what this does, maybe update for background algorithms?

            lines.append(plotdata(np.array(data), x_limit,
                                  dictMaxEvals[alg], maxevals2=dictMaxEvals2[alg],
                                  CrE=CrEperAlg[alg], **args))

    if 11 < 3:
        import time
        plt.text(1e5, 0, ' {}'.format(time.asctime()), fontsize=5)
    labels, handles = plotLegend(lines, x_limit)
    if True:  # isLateXLeg:
        if info:
            file_name = os.path.join(outputdir, '%s_%s.tex' % (genericsettings.pprldmany_file_name, info))
        else:
            file_name = os.path.join(outputdir, '%s.tex' % genericsettings.pprldmany_file_name)
        with open(file_name, 'w') as file_obj:
            file_obj.write(r'\providecommand{\nperfprof}{7}')
            algtocommand = {}  # latex commands
            for i, alg in enumerate(order):
                tmp = r'\alg%sperfprof' % pptex.numtotext(i)
                file_obj.write(r'\providecommand{%s}{\StrLeft{%s}{\nperfprof}}' %
                        (tmp, toolsdivers.str_to_latex(
                            toolsdivers.strip_pathname2(algname_to_label(alg)))))
                algtocommand[algname_to_label(alg)] = tmp
            if displaybest:
                tmp = r'\algzeroperfprof'
                refalgname = testbedsettings.current_testbed.reference_algorithm_displayname
                file_obj.write(r'\providecommand{%s}{%s}' % (tmp, refalgname))
                algtocommand[algname_to_label(refalgname)] = tmp

            commandnames = []
            for label in labels:
                commandnames.append(algtocommand[label])
            # file_obj.write(headleg)
            if len(
                    order) > 28:  # latex sidepanel won't work well for more than 25 algorithms, but original labels are also clipped
                file_obj.write(r'\providecommand{\perfprofsidepanel}{\mbox{%s}\vfill\mbox{%s}}'
                        % (commandnames[0], commandnames[-1]))
            else:
                fontsize_command = r'\tiny{}' if len(order) > 19 else ''
                file_obj.write(r'\providecommand{\perfprofsidepanel}{{%s\mbox{%s}' %
                        (fontsize_command, commandnames[0]))  # TODO: check len(labels) > 0
                for i in range(1, len(labels)):
                    file_obj.write('\n' + r'\vfill \mbox{%s}' % commandnames[i])
                file_obj.write('}}\n')
            # file_obj.write(footleg)
            if genericsettings.verbose:
                print('Wrote right-hand legend in %s' % file_name)

    if info:
        figureName = os.path.join(outputdir, '%s_%s' % (genericsettings.pprldmany_file_name, info))
    else:
        figureName = os.path.join(outputdir, '%s' % genericsettings.pprldmany_file_name)
    # beautify(figureName, funcsolved, x_limit*x_annote_factor, False, fileFormat=figformat)
    beautify()  # see also below where the ticks are set

    if plotType == PlotType.FUNC:
        dictFG = pp.dictAlgByFuncGroup(dictAlg)
        dictKey = list(dictFG.keys())[0]
        functionGroups = dictAlg[list(dictAlg.keys())[0]].getFuncGroups()
        if testbedsettings.current_testbed.has_constraints:
            # HACK: because a function is in at least two groups:
            # {{separ or hcond or multi} and all} with {m} constraints
            # the original method is broken for bbob-constrained
            listGroups = list(functionGroups.values())
            if len(functionGroups) == 2:
                groupName = listGroups[0]  # all is the last one
            if len(functionGroups) > 2:
                groupName = listGroups[-1]  # same reason
        else:
            groupName = functionGroups[dictKey]
        text = '%s\n%s, %d-D' % (testbedsettings.current_testbed.name,
                                 groupName,
                                 dimList[0])
    else:
        text = '%s %s' % (testbedsettings.current_testbed.name,
                            ppfig.consecutiveNumbers(sorted(dictFunc.keys()), 'f'))
        if not (plotType == PlotType.DIM):
            text += ', %d-D' % dimList[0]
    # add information about smallest and largest target and their number
    text += '\n'
    targetstrings = target_values.labels()
    if isinstance(target_values, pp.RunlengthBasedTargetValues):
        text += (str(len(targetstrings)) + ' targets RLs/dim: ' +
                 targetstrings[0] + '..' +
                 targetstrings[len(targetstrings)-1] + '\n')
        text += '  from ' + testbedsettings.current_testbed.reference_algorithm_filename
    else:
        text += (str(len(targetstrings)) + ' targets: ' +
                 targetstrings[0] + '..' +
                 targetstrings[len(targetstrings)-1])
    # add weights for constrained testbeds
    text += text_infigure_if_constraints()
    # add number of instances 
    text += '\n'
    num_of_instances = []
    for alg in algorithms_with_data:
        try:
            num_of_instances.append(len((dictAlgperFunc[alg])[0].instancenumbers))
        except IndexError:
            pass
    # issue a warning if number of instances is inconsistent, but always
    # display only the present number of instances, i.e. remove copies
    if len(set(num_of_instances)) > 1 and genericsettings.warning_level >= 5:
        warnings.warn('Number of instances inconsistent over all algorithms: %s instances found.' % str(num_of_instances))
    num_of_instances = set(num_of_instances)
    for n in num_of_instances:
        text += '%d, ' % n
            
    text = text.rstrip(', ')
    text += ' instances'
    plt.text(0.01, 0.99, text,
             horizontalalignment="left",
             verticalalignment="top",
             transform=plt.gca().transAxes,
             fontsize=0.6*label_fontsize)
    if len(dictFunc) == 1:
        plt.title(' '.join((str(list(dictFunc.keys())[0]),
                            testbedsettings.current_testbed.short_names[list(dictFunc.keys())[0]])),
                  fontsize=title_fontsize)
    a = plt.gca()

    # beautify even more: ticks, grid and frame
    a.set_xlim(1e-0, x_limit)
    log_xlimit = int(np.log10(x_limit))  # last annotatable decade
    a.set_xticks([10**d for d in range(log_xlimit + 1)])
    a.set_xticklabels([str(d) for d in range(log_xlimit + 1)])
    a.set_xticks([i * 10**d for d in range(log_xlimit)  # these should be default
                      for i in [2, 3, 4, 5, 6, 7, 8, 9]],
                 minor=True)
    a.set_yticks(np.linspace(0, 1, 21), minor=True)  # every 0.05 = 5%
    a.tick_params(top=True, bottom=True,  which='both',
                  left=True, right=True, direction='out')
    a.grid(True)
    a.grid(genericsettings.minor_grid_alpha_in_pprldmany > 0,
           alpha=genericsettings.minor_grid_alpha_in_pprldmany, which='minor')
    for pos in ['right', 'left']:  # top makes sense if figure ends at 1.0
        a.spines[pos].set_visible(False)  # remove visible frame

    if save_figure:
        ppfig.save_figure(figureName,
                          dictAlg[algorithms_with_data[0]][0].algId,
                          layout_rect=(0, 0, 0.783, 1),  # see also below
                          # Prevent clipping in matplotlib >=3:
                          # Relative additional space numbers are
                          # bottom, left, 1 - top, and 1 - right.
                          # bottom=0.13 still clips g in the log(#evals) xlabel
                          subplots_adjust=dict(bottom=0.135, right=0.783,  # was: 0.735
                                               top=0.92 if len(dictFunc) == 1 else 0.98  # space for a title or 1.0 y-tick annotation
                                               ),
                          )
        if plotType == PlotType.DIM:
            file_name = genericsettings.pprldmany_file_name
            ppfig.save_single_functions_html(
                os.path.join(outputdir, file_name),
                '',  # algorithms names are clearly visible in the figure
                htmlPage=ppfig.HtmlPage.NON_SPECIFIED,
                parentFileName='../%s' % parentHtmlFileName if parentHtmlFileName else None,
                header=ppfig.pprldmany_per_func_dim_header)

    if close_figure:
        plt.close()

        # TODO: should return status or sthg
