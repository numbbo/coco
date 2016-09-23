#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates figure of the bootstrap distribution of aRT.
    
The main method in this module generates figures of Empirical
Cumulative Distribution Functions of the bootstrap distribution of
the Average Running Time (aRT) divided by the dimension for many
algorithms.

The outputs show the ECDFs of the running times of the simulated runs
divided by dimension for 50 different targets logarithmically uniformly
distributed in [1e−8, 1e2]. The crosses (×) give the median number of
function evaluations of unsuccessful runs divided by dimension.

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
    
    # Empirical cumulative distribution function of bootstrapped aRT figure
    ds = bb.load(glob.glob('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f0*_20.pickle'))
    figure()
    bb.compall.pprldmany.plot(ds) # must rather call main instead of plot?
    bb.compall.pprldmany.beautify()

"""

from __future__ import absolute_import

import os
import warnings
from pdb import set_trace
import numpy as np
import matplotlib.pyplot as plt
from .. import toolsstats, bestalg, genericsettings, testbedsettings
from .. import pproc as pp  # import dictAlgByDim, dictAlgByFun
from .. import toolsdivers  # strip_pathname, str_to_latex
from .. import pprldistr  # plotECDF, beautifyECDF
from .. import ppfig  # consecutiveNumbers, saveFigure, plotUnifLogXMarkers, logxticks
from .. import pptex  # numtotex

PlotType = ppfig.enum('ALG', 'DIM', 'FUNC')

displaybest = True
x_limit = None  # not sure whether this is necessary/useful
x_limit_default = 1e7  # better: 10 * genericsettings.evaluation_setting[1], noisy: 1e8, otherwise: 1e7. maximal run length shown
divide_by_dimension = True
annotation_line_end_relative = 1.11  # lines between graph and annotation
annotation_space_end_relative = 1.24  # figure space end relative to x_limit
save_zoom = False  # save zoom into left and right part of the figures
perfprofsamplesize = genericsettings.simulated_runlength_bootstrap_sample_size  # number of bootstrap samples drawn for each fct+target in the performance profile
nbperdecade = 1
median_max_evals_marker_format = ['x', 24, 3]
label_fontsize = 18
styles = [d.copy() for d in genericsettings.line_styles]  # deep copy

refcolor = 'wheat'
"""color of reference (best) algorithm"""

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

funi = [1, 2] + range(5, 15)  # 2 is paired Ellipsoid
funilipschitz = [1] + [5, 6] + range(8, 13) + [14]  # + [13]  #13=sharp ridge, 7=step-ellipsoid
fmulti = [3, 4] + range(15, 25)  # 3 = paired Rastrigin
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
    plt.xlim(xmin=1e-0)

    global divide_by_dimension
    if divide_by_dimension:
        if testbedsettings.current_testbed.name == testbedsettings.testbed_name_cons:
            plt.xlabel('log10 of # (f+g)-evals / dimension', fontsize=label_fontsize)
        else:
            plt.xlabel('log10 of (# f-evals / dimension)', fontsize=label_fontsize)
    else:
        if testbedsettings.current_testbed.name == testbedsettings.testbed_name_cons:
            plt.xlabel('log10 of # (f+g)-evals', fontsize=label_fontsize)
        else:
            plt.xlabel('log10 of # f-evals', fontsize=label_fontsize)
    plt.ylabel('Proportion of function+target pairs', fontsize=label_fontsize)
    ppfig.logxticks()
    pprldistr.beautifyECDF()


def plotdata(data, maxval=None, maxevals=None, CrE=0., **kwargs):
    """Draw a normalized ECDF. What means normalized?
    
    :param seq data: data set, a 1-D ndarray of runlengths
    :param float maxval: right-most value to be displayed, will use the
                         largest non-inf, non-nan value in data if not
                         provided
    :param seq maxevals: if provided, will plot the median of this
                         sequence as a single cross marker
    :param float CrE: Crafting effort the data will be multiplied by
                      the exponential of this value.
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
            plt_plot([x_last] * 2, [y_last] * 2, '.', color=c, markeredgecolor=c)
        except:
            pass
        x2 = np.hstack([np.repeat(x, 2), maxval])  # repeat x-values for each step in the cdf
        y2 = np.hstack([0.0, np.repeat(y / float(nn), 2)])

        res = ppfig.plotUnifLogXMarkers(x2, y2, nbperdecade * 3 / np.log10(maxval),
                                        logscale=False, clip_on=False, **kwargs)
        # res = plotUnifLogXMarkers(x2, y2, nbperdecade, logscale=False, **kwargs)

        if maxevals:  # Should cover the case where maxevals is None or empty
            x3 = np.median(maxevals)
            if (x3 <= maxval and
                # np.any(x2 <= x3) and   # maxval < median(maxevals)
                not plt.getp(res[-1], 'label').startswith('best')
                ): # TODO: HACK for not considering a "best" algorithm line
                
                try:
                    y3 = y2[x2 <= x3][-1]  # find right y-value for x3==median(maxevals)
                except IndexError:  # median(maxevals) is smaller than any data, can only happen because of CrE?
                    y3 = y2[0]
                h = plt.plot((x3,), (y3,),
                             marker=median_max_evals_marker_format[0],
                             markersize=median_max_evals_marker_format[1],
                             markeredgewidth=median_max_evals_marker_format[2],
                             # marker='x', markersize=24, markeredgewidth=3, 
                             markeredgecolor=plt.getp(res[0], 'color'),
                             ls=plt.getp(res[0], 'ls'),
                             color=plt.getp(res[0], 'color'))
                h.extend(res)
                res = h  # so the last element in res still has the label.
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

    def get_label_length(labelList):
        """ Finds the minimal length of the names used in the label so that 
        all the names are different. Always at least 9 character are displayed.
        """

        numberOfCharacters = 7
        firstPart = [i[:numberOfCharacters] for i in labelList]
        maxLength = max(len(i) for i in labelList)
        while (len(firstPart) > len(set(firstPart)) and numberOfCharacters <= maxLength):
            numberOfCharacters += 1
            firstPart = [i[:numberOfCharacters] for i in labelList]

        return min(numberOfCharacters + 2, maxLength)

    labelList = [toolsdivers.strip_pathname1(plt.getp(h[-1], 'label')) for h in handles]
    numberOfCharacters = get_label_length(labelList)
    for h in handles:
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
    fontsize = genericsettings.minmax_algorithm_fontsize[0] + np.min((1, np.exp(9 - lh))) * (
        genericsettings.minmax_algorithm_fontsize[-1] - genericsettings.minmax_algorithm_fontsize[0])
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
                    for attr in ('lw', 'ls', 'marker',
                                 'markeredgewidth', 'markerfacecolor',
                                 'markeredgecolor', 'markersize', 'zorder'):
                        tmp[attr] = plt.getp(h, attr)
                    legx = maxval ** annotation_line_end_relative
                    if 'marker' in attr:
                        legx = maxval ** annotation_line_end_relative
                    # reshandles.extend(plt_plot((maxval, legx), (j, y),
                    reshandles.extend(plt_plot((maxval, legx), (j, y),
                                               color=plt.getp(h, 'markeredgecolor'), **tmp))
                    reshandles.append(
                        plt.text(maxval ** (0.02 + annotation_line_end_relative), y,
                                 toolsdivers.str_to_latex(
                                     toolsdivers.strip_pathname1(plt.getp(h, 'label'))[:numberOfCharacters]),
                                 horizontalalignment="left",
                                 verticalalignment="center", size=fontsize))
                    reslabels.append(plt.getp(h, 'label'))
                    # set_trace()
                    i += 1

    # plt.axvline(x=maxval, color='k') # Not as efficient?
    reshandles.append(plt_plot((maxval, maxval), (0., 1.), color='k'))
    reslabels.reverse()
    plt.xlim(xmax=maxval ** annotation_space_end_relative)
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
            runlengthunsucc = entry.maxevals[np.isnan(evals)] / divisor
            if len(runlengthsucc) > 0:
                x = toolsstats.drawSP(runlengthsucc, runlengthunsucc,
                                      percentiles=[50],
                                      samplesize=perfprofsamplesize)[1]
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


def all_single_functions(dictAlg, isSingleAlgorithm, sortedAlgs=None,
                         outputdir='.', parentHtmlFileName=None):
    single_fct_output_dir = (outputdir.rstrip(os.sep) + os.sep +
                             'pprldmany-single-functions'
                             # + os.sep + ('f%03d' % fg)
                             )
    if not os.path.exists(single_fct_output_dir):
        os.makedirs(single_fct_output_dir)

    if isSingleAlgorithm:
        main(dictAlg,
             order=sortedAlgs,
             outputdir=single_fct_output_dir,
             info='',
             parentHtmlFileName=parentHtmlFileName,
             plotType=PlotType.DIM)

        dictFG = pp.dictAlgByFuncGroup(dictAlg)
        for fg, entries in dictFG.iteritems():
            main(entries,
                 order=sortedAlgs,
                 outputdir=single_fct_output_dir,
                 info='%s' % (fg),
                 parentHtmlFileName=parentHtmlFileName,
                 plotType=PlotType.DIM)

    dictFG = pp.dictAlgByFun(dictAlg)
    for fg, tempDictAlg in dictFG.iteritems():

        if isSingleAlgorithm:
            main(tempDictAlg,
                 order=sortedAlgs,
                 outputdir=single_fct_output_dir,
                 info='f%03d' % (fg),
                 parentHtmlFileName=parentHtmlFileName,
                 plotType=PlotType.DIM)
        else:
            dictDim = pp.dictAlgByDim(tempDictAlg)
            dims = sorted(dictDim)
            for i, d in enumerate(dims):
                entries = dictDim[d]
                next_dim = dims[i + 1] if i + 1 < len(dims) else dims[0]
                main(entries,
                     order=sortedAlgs,
                     outputdir=single_fct_output_dir,
                     info='f%03d_%02dD' % (fg, d),
                     parentHtmlFileName=parentHtmlFileName,
                     add_to_html_file_name='_%02dD' % d,
                     next_html_page_suffix='_%02dD' % next_dim)

    if isSingleAlgorithm:
        functionGroups = dictAlg[dictAlg.keys()[0]].getFuncGroups()

        dictDim = pp.dictAlgByDim(dictAlg)
        dims = sorted(dictDim)
        for i, d in enumerate(dims):
            tempDictAlg = dictDim[d]
            next_dim = dims[i+1] if i + 1 < len(dims) else dims[0]
            dictFG = pp.dictAlgByFuncGroup(tempDictAlg)
            for fg, entries in dictFG.iteritems():
                main(entries,
                     order=sortedAlgs,
                     outputdir=single_fct_output_dir,
                     info='gr_%s_%02dD' % (fg, d),
                     parentHtmlFileName=parentHtmlFileName,
                     plotType=PlotType.FUNC)

            ppfig.save_single_functions_html(
                os.path.join(single_fct_output_dir, genericsettings.pprldmany_group_file_name),
                '',
                add_to_names='_%02dD' % d,
                next_html_page_suffix='_%02dD' % next_dim,
                htmlPage=ppfig.HtmlPage.PPRLDMANY_BY_GROUP,
                functionGroups=functionGroups,
                parentFileName='../%s' % parentHtmlFileName if parentHtmlFileName else None
            )


def main(dictAlg, order=None, outputdir='.', info='default',
         dimension=None, parentHtmlFileName=None, plotType=PlotType.ALG,
         add_to_html_file_name='', next_html_page_suffix=None):
    """Generates a figure showing the performance of algorithms.

    From a dictionary of :py:class:`DataSetList` sorted by algorithms,
    generates the cumulative distribution function of the bootstrap
    distribution of aRT for algorithms on multiple functions for
    multiple targets altogether.

    :param dict dictAlg: dictionary of :py:class:`DataSetList` instances
                         one instance is equivalent to one algorithm,
    :param list targets: target function values
    :param list order: sorted list of keys to dictAlg for plotting order
    :param str outputdir: output directory
    :param str info: output file name suffix
    :param str parentHtmlFileName: defines the parent html page 

    """
    global x_limit  # late assignment of default, because it can be set to None in config 
    global divide_by_dimension  # not fully implemented/tested yet
    if 'x_limit' not in globals() or x_limit is None:
        x_limit = x_limit_default

    tmp = pp.dictAlgByDim(dictAlg)
    algorithms_with_data = [a for a in dictAlg.keys() if dictAlg[a] != []]

    if len(algorithms_with_data) > 1 and len(tmp) != 1 and dimension is None:
        raise ValueError('We never integrate over dimension for than one algorithm.')
    if dimension is not None:
        if dimension not in tmp.keys():
            raise ValueError('dimension %d not in dictAlg dimensions %s'
                             % (dimension, str(tmp.keys())))
        tmp = {dimension: tmp[dimension]}
    dimList = tmp.keys()

    # The sort order will be defined inside this function.    
    if plotType == PlotType.DIM:
        order = []

    # Collect data
    # Crafting effort correction: should we consider any?
    CrEperAlg = {}
    for alg in algorithms_with_data:
        CrE = 0.
        if 1 < 3 and dictAlg[alg][0].algId == 'GLOBAL':
            tmp = dictAlg[alg].dictByNoise()
            assert len(tmp.keys()) == 1
            if tmp.keys()[0] == 'noiselessall':
                CrE = 0.5117
            elif tmp.keys()[0] == 'nzall':
                CrE = 0.6572
        if plotType == PlotType.DIM:
            for dim in dimList:
                keyValue = '%d-D' % (dim)
                CrEperAlg[keyValue] = CrE
        elif plotType == PlotType.FUNC:
            tmp = pp.dictAlgByFun(dictAlg)
            for f, dictAlgperFunc in tmp.iteritems():
                keyValue = 'f%d' % (f)
                CrEperAlg[keyValue] = CrE
        else:
            CrEperAlg[alg] = CrE
        if CrE != 0.0:
            print 'Crafting effort for', alg, 'is', CrE

    dictData = {}  # list of (ert per function) per algorithm
    dictMaxEvals = {}  # list of (maxevals per function) per algorithm

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
        for f, dictAlgperFunc in dictFunc.iteritems():
            # print target_values((f, dim))
            for j, t in enumerate(target_values((f, dim))):
                # for j, t in enumerate(testbedsettings.current_testbed.ecdf_target_values(1e2, f)):
                # funcsolved[j].add(f)

                for alg in algorithms_with_data:
                    x = [np.inf] * perfprofsamplesize
                    runlengthunsucc = []
                    try:
                        entry = dictAlgperFunc[alg][0]  # one element per fun and per dim.
                        evals = entry.detEvals([t])[0]
                        assert entry.dim == dim
                        runlengthsucc = evals[np.isnan(evals) == False] / divisor
                        runlengthunsucc = entry.maxevals[np.isnan(evals)] / divisor
                        if len(runlengthsucc) > 0:
                            x = toolsstats.drawSP(runlengthsucc, runlengthunsucc,
                                                  percentiles=[50],
                                                  samplesize=perfprofsamplesize)[1]
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
                    dictMaxEvals.setdefault(keyValue, []).extend(runlengthunsucc)

            displaybest = plotType == PlotType.ALG
            if displaybest:
                # set_trace()
                bestalgentries = bestalg.load_best_algorithm(testbedsettings.current_testbed.best_algorithm_filename)

                if not bestalgentries:
                    displaybest = False
                else:
                    bestalgentry = bestalgentries[(dim, f)]
                    bestalgevals = bestalgentry.detEvals(target_values((f, dim)))
                    # print bestalgevals
                    for j in range(len(bestalgevals[0])):
                        if bestalgevals[1][j]:
                            evals = bestalgevals[0][j]
                            # set_trace()
                            assert dim == bestalgentry.dim
                            runlengthsucc = evals[np.isnan(evals) == False] / divisor
                            runlengthunsucc = bestalgentry.maxevals[bestalgevals[1][j]][np.isnan(evals)] / divisor
                            x = toolsstats.drawSP(runlengthsucc, runlengthunsucc,
                                                  percentiles=[50],
                                                  samplesize=perfprofsamplesize)[1]
                        else:
                            x = perfprofsamplesize * [np.inf]
                            runlengthunsucc = []
                        xbest.extend(x)
                        maxevalsbest.extend(runlengthunsucc)

    if order is None:
        order = dictData.keys()

    # Display data
    lines = []
    if displaybest:
        args = {'ls': '-', 'linewidth': 6, 'marker': 'D', 'markersize': 11.,
                'markeredgewidth': 1.5, 'markerfacecolor': refcolor,
                'markeredgecolor': refcolor, 'color': refcolor,
                'label': testbedsettings.current_testbed.best_algorithm_displayname,
                'zorder': -1}
        lines.append(plotdata(np.array(xbest), x_limit, maxevalsbest,
                              CrE=0., **args))

    def algname_to_label(algname, dirname=None):
        """to be extended to become generally useful"""
        if isinstance(algname, (tuple, list)):  # not sure this is needed
            return ' '.join([str(name) for name in algname])
        return str(algname)

    for i, alg in enumerate(order):
        try:
            data = dictData[alg]
            maxevals = dictMaxEvals[alg]
        except KeyError:
            continue

        args = styles[i % len(styles)]
        args = args.copy()
        args['linewidth'] = 1.5
        args['markersize'] = 12.
        args['markeredgewidth'] = 1.5
        args['markerfacecolor'] = 'None'
        args['markeredgecolor'] = args['color']
        args['label'] = algname_to_label(alg)
        if plotType == PlotType.DIM:
            args['marker'] = genericsettings.dim_related_markers[i]
            args['markeredgecolor'] = genericsettings.dim_related_colors[i]
            args['color'] = genericsettings.dim_related_colors[i]

            # args['markevery'] = perfprofsamplesize # option available in latest version of matplotlib
            # elif len(show_algorithms) > 0:
            # args['color'] = 'wheat'
            # args['ls'] = '-'
            # args['zorder'] = -1
        # plotdata calls pprldistr.plotECDF which calls ppfig.plotUnifLog... which does the work
        lines.append(plotdata(np.array(data), x_limit, maxevals,
                              CrE=CrEperAlg[alg], **args))

    labels, handles = plotLegend(lines, x_limit)
    if True:  # isLateXLeg:
        if info:
            fileName = os.path.join(outputdir, '%s_%s.tex' % (genericsettings.pprldmany_file_name, info))
        else:
            fileName = os.path.join(outputdir, '%s.tex' % (genericsettings.pprldmany_file_name))
        with open(fileName, 'w') as f:
            f.write(r'\providecommand{\nperfprof}{7}')
            algtocommand = {}  # latex commands
            for i, alg in enumerate(order):
                tmp = r'\alg%sperfprof' % pptex.numtotext(i)
                f.write(r'\providecommand{%s}{\StrLeft{%s}{\nperfprof}}' %
                        (tmp, toolsdivers.str_to_latex(
                            toolsdivers.strip_pathname2(algname_to_label(alg)))))
                algtocommand[algname_to_label(alg)] = tmp
            if displaybest:
                tmp = r'\algzeroperfprof'
                bestalgname = testbedsettings.current_testbed.best_algorithm_displayname
                f.write(r'\providecommand{%s}{%s}' % (tmp, bestalgname))
                algtocommand[algname_to_label(bestalgname)] = tmp

            commandnames = []
            for label in labels:
                commandnames.append(algtocommand[label])
            # f.write(headleg)
            if len(
                    order) > 28:  # latex sidepanel won't work well for more than 25 algorithms, but original labels are also clipped
                f.write(r'\providecommand{\perfprofsidepanel}{\mbox{%s}\vfill\mbox{%s}}'
                        % (commandnames[0], commandnames[-1]))
            else:
                fontsize_command = r'\tiny{}' if len(order) > 19 else ''
                f.write(r'\providecommand{\perfprofsidepanel}{{%s\mbox{%s}' %
                        (fontsize_command, commandnames[0]))  # TODO: check len(labels) > 0
                for i in range(1, len(labels)):
                    f.write('\n' + r'\vfill \mbox{%s}' % commandnames[i])
                f.write('}}\n')
            # f.write(footleg)
            if genericsettings.verbose:
                print 'Wrote right-hand legend in %s' % fileName

    if info:
        figureName = os.path.join(outputdir, '%s_%s' % (genericsettings.pprldmany_file_name, info))
    else:
        figureName = os.path.join(outputdir, '%s' % (genericsettings.pprldmany_file_name))
    # beautify(figureName, funcsolved, x_limit*x_annote_factor, False, fileFormat=figformat)
    beautify()

    if plotType == PlotType.FUNC:
        dictFG = pp.dictAlgByFuncGroup(dictAlg)
        dictKey = dictFG.keys()[0]
        functionGroups = dictAlg[dictAlg.keys()[0]].getFuncGroups()
        text = '%s\n%s, %d-D' % (testbedsettings.current_testbed.name,
                                 functionGroups[dictKey],
                                 dimList[0])
    else:
        text = '%s - %s' % (testbedsettings.current_testbed.name,
                            ppfig.consecutiveNumbers(sorted(dictFunc.keys()), 'f'))
        if not (plotType == PlotType.DIM):
            text += ', %d-D' % dimList[0]
    # add information about smallest and largest target and their number
    text += '\n'
    targetstrings = target_values.labels()
    if isinstance(target_values, pp.RunlengthBasedTargetValues):
        text += (str(len(targetstrings)) + ' target RLs/dim: ' +
                 targetstrings[0] + '..' +
                 targetstrings[len(targetstrings)-1] + '\n')
        text += '   from ' + testbedsettings.current_testbed.best_algorithm_filename
    else:
        text += (str(len(targetstrings)) + ' targets in ' +
                 targetstrings[0] + '..' +
                 targetstrings[len(targetstrings)-1])        
    # add number of instances 
    text += '\n'
    num_of_instances = []
    for alg in algorithms_with_data:
        if len(dictAlgperFunc[alg]) > 0:
            num_of_instances.append(len((dictAlgperFunc[alg])[0].instancenumbers))
        else:
            warnings.warn('The data for algorithm %s and function %s are missing' % (alg, f))
    # issue a warning if number of instances is inconsistant, otherwise
    # display only the present number of instances, i.e. remove copies
    if len(set(num_of_instances)) > 1:
        warnings.warn('Number of instances inconsistent over all algorithms: %s instances found.' % str(num_of_instances))
    else:
        num_of_instances = set(num_of_instances)
    for n in num_of_instances:
        text += '%d, ' % n
            
    text = text.rstrip(', ')
    text += ' instances'

    plt.text(0.01, 0.98, text, horizontalalignment="left",
             verticalalignment="top", transform=plt.gca().transAxes, size='small')
    if len(dictFunc) == 1:
        plt.title(' '.join((str(dictFunc.keys()[0]),
                            testbedsettings.current_testbed.short_names[dictFunc.keys()[0]])))
    a = plt.gca()

    plt.xlim(xmin=1e-0, xmax=x_limit ** annotation_space_end_relative)
    xticks, labels = plt.xticks()
    tmp = []
    for i in xticks:
        tmp.append('%d' % round(np.log10(i)))
    a.set_xticklabels(tmp)

    if save_figure:
        ppfig.saveFigure(figureName)
        if len(dictFunc) == 1 or plotType == PlotType.DIM:
            fileName = genericsettings.pprldmany_file_name

            header = ppfig.pprldmany_per_func_header if plotType == PlotType.DIM else ppfig.pprldmany_per_func_dim_header
            ppfig.save_single_functions_html(
                os.path.join(outputdir, fileName),
                '',  # algorithms names are clearly visible in the figure
                add_to_names=add_to_html_file_name,
                next_html_page_suffix=next_html_page_suffix,
                htmlPage=ppfig.HtmlPage.NON_SPECIFIED,
                parentFileName='../%s' % parentHtmlFileName if parentHtmlFileName else None,
                header=header)

    if close_figure:
        plt.close()

        # TODO: should return status or sthg


if __name__ == "__main__":
    # should become a test case
    import sys
    import bbob_pproc

    sys.path.append('.')
