#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates figure of the bootstrap distribution of ERT.
    
The main method in this module generates figures of Empirical
Cumulative Distribution Functions of the bootstrap distribution of
the Expected Running Time (ERT) divided by the dimension for many
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
    
    # Empirical cumulative distribution function of bootstrapped ERT figure
    ds = bb.load(glob.glob('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f0*_20.pickle'))
    figure()
    bb.compall.pprldmany.plot(ds)
    bb.compall.pprldmany.beautify()

"""

from __future__ import absolute_import

import os
import warnings
from pdb import set_trace
import numpy as np
import matplotlib.pyplot as plt
from bbob_pproc import bootstrap, bestalg
from bbob_pproc.pproc import dictAlgByDim, dictAlgByFun
from bbob_pproc.pprldistr import plotECDF, beautifyECDF
from bbob_pproc.ppfig import consecutiveNumbers, saveFigure, plotUnifLogXMarkers, logxticks
from bbob_pproc.pptex import writeLabels, numtotext

figformat = ('eps', 'pdf') # Controls the output when using the main method

best = ('AMaLGaM IDEA', 'iAMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES', 'IPOP-SEP-CMA-ES',
   'BFGS', 'NELDER (Han)', 'NELDER (Doe)', 'NEWUOA', 'full NEWUOA', 'GLOBAL', 'MCS (Neum)',
   'DIRECT', 'DASA', 'POEMS', 'Cauchy EDA', 'Monte Carlo')

best2 = ('AMaLGaM IDEA', 'iAMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES', 'IPOP-SEP-CMA-ES', 'BFGS', 'NEWUOA', 'GLOBAL')

eseda = ('AMaLGaM IDEA', 'iAMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES', 'IPOP-SEP-CMA-ES', '(1+1)-CMA-ES', '(1+1)-ES')

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

funi = [1,2] + range(5, 15)  # 2 is paired Ellipsoid
funilipschitz = [1] + [5,6] + range(8,13) + [14] # + [13]  #13=sharp ridge, 7=step-ellipsoid 
fmulti = [3, 4] + range(15,25) # 3 = paired Rastrigin
funisep = [1,2,5]

displaybest2009 = True

# input parameter settings
#show_algorithms = eseda + ('BFGS',) # ()==all
#show_algorithms = ('IPOP-SEP-CMA-ES', 'IPOP-CMA-ES', 'BIPOP-CMA-ES',)
#show_algorithms = ('IPOP-SEP-CMA-ES', 'IPOP-CMA-ES', 'BIPOP-CMA-ES',
#'avg NEWUOA', 'NEWUOA', 'full NEWUOA', 'BFGS', 'MCS (Neum)', 'GLOBAL', 'NELDER (Han)',
#'NELDER (Doe)', 'Monte Carlo') # ()==all
show_algorithms = ()
function_IDs = ()
function_IDs = range(1,200)  # sep ros high mul mulw == 1, 6, 10, 15, 20, 101, 107, 122, 
#function_IDs = range(101,199)  # sep ros high mul mulw == 1, 6, 10, 15, 20, 101, 107, 122, 
#function_IDs = fmulti # funi fmulti  # range(103, 131, 3)   # displayed functions
#function_IDs = [1,2,3,4,5] # separable functions
#function_IDs = [6,7,8,9]   # moderate functions
#function_IDs = [10,11,12,13,14] # ill-conditioned functions
#function_IDs = [15,16,17,18,19] # multi-modal functions
#function_IDs = [20,21,22,23,24] # weak structure functions
#function_IDs = range(101,131) # noisy testbed
#function_IDs = range(101,106+1)  # moderate noise
#function_IDs = range(107,130+1)  # severe noise
#function_IDs = range(101,130+1, 3)  # gauss noise
#function_IDs = range(102,130+1, 3)  # unif noise
#function_IDs = range(103,130+1, 3)  # cauchy noise
# function_IDs = range(15,25) # multimodal nonseparable

x_limit = 1e7   # noisy: 1e8, otherwise: 1e7. maximal run length shown
x_annote_factor = 90 # make space for right-hand legend
fontsize = 10.0 # default setting, is modified in genericsettings.py

save_zoom = False  # save zoom into left and right part of the figures
perfprofsamplesize = 100  # number of bootstrap samples drawn for each fct+target in the performance profile
dpi_global_var = 100  # 100 ==> 800x600 (~160KB), 120 ==> 960x720 (~200KB), 150 ==> 1200x900 (~300KB) looks ugly in latex

nbperdecade = 3

styles = [{'marker': 'o', 'linestyle': '-', 'color': 'b'},
          {'marker': 'd', 'linestyle': '-', 'color': 'g'},
          {'marker': 's', 'linestyle': '-', 'color': 'r'},
          {'marker': 'v', 'linestyle': '-', 'color': 'c'},
          {'marker': '*', 'linestyle': '-', 'color': 'm'},
          {'marker': 'h', 'linestyle': '-', 'color': 'y'},
          {'marker': '^', 'linestyle': '-', 'color': 'k'},
          {'marker': 'p', 'linestyle': '-', 'color': 'b'},
          {'marker': 'H', 'linestyle': '-', 'color': 'g'},
          {'marker': '<', 'linestyle': '-', 'color': 'r'},
          {'marker': 'D', 'linestyle': '-', 'color': 'c'},
          {'marker': '>', 'linestyle': '-', 'color': 'm'},
          {'marker': '1', 'linestyle': '-', 'color': 'y'},
          {'marker': '2', 'linestyle': '-', 'color': 'k'},
          {'marker': '3', 'linestyle': '-', 'color': 'b'},
          {'marker': '4', 'linestyle': '-', 'color': 'g'}]
refcolor = 'wheat'
#'-'     solid line style
#'--'    dashed line style
#'-.'    dash-dot line style
#':'     dotted line style
#'.'     point marker
#','     pixel marker
#'o'     circle marker
#'v'     triangle_down marker
#'^'     triangle_up marker
#'<'     triangle_left marker
#'>'     triangle_right marker
#'1'     tri_down marker
#'2'     tri_up marker
#'3'     tri_left marker
#'4'     tri_right marker
#'s'     square marker
#'p'     pentagon marker
#'*'     star marker
#'h'     hexagon1 marker
#'H'     hexagon2 marker
#'+'     plus marker
#'x'     x marker
#'D'     diamond marker
#'d'     thin_diamond marker
#'|'     vline marker
#'_'     hline marker
headleg = (r'\raisebox{.037\textwidth}{\parbox[b]'
           + r'[.3\textwidth]{.0868\textwidth}{\begin{scriptsize}')
footleg = (r'%do not remove the empty line below' + '\n\n' +
           r'\end{scriptsize}}}')

defaulttargets = tuple(10**np.r_[-8:2:0.2])

def beautify():
    """Customize figure presentation."""

    #plt.xscale('log') # Does not work with matplotlib 0.91.2
    a = plt.gca()
    a.set_xscale('log')
    #Tick label handling
    plt.xlim(xmin=1e-0)

    plt.xlabel('log10 of (ERT / dimension)')
    plt.ylabel('Proportion of functions')
    logxticks()
    beautifyECDF()

def plotdata(data, maxval=None, maxevals=None, CrE=0., **kwargs):
    """Draw a normalized ECDF.
    
    :param seq data: data set
    :param float maxval: right-most value to be displayed, will use the
                         largest non-inf, non-nan value in data if not
                         provided
    :param seq maxevals: if provided, will plot the median of this
                         sequence as a single cross marker
    :param float CrE: Crafting effort the data will be multiplied by
                      the exponential of this value.
    :param kwargs: optional arguments provided to plot function.
    
    """
    #Expect data to be a ndarray.
    x = data[np.isnan(data)==False] # Take away the nans
    nn = len(x)

    x = x[np.isinf(x)==False] # Take away the infs
    n = len(x)

    x = np.exp(CrE) * x  # correction by crafting effort CrE

    if n == 0:
        res = list()
        res.append(plt.axhline(0., **kwargs))
    else:
        dictx = {}
        for i in x:
            dictx[i] = dictx.get(i, 0) + 1

        x = np.array(sorted(dictx))
        
        if maxval is None:
            maxval = max(x)
        x = x[x <= maxval]
        y = np.cumsum(list(dictx[i] for i in x))

        x2 = np.hstack([np.repeat(x, 2), maxval])
        y2 = np.hstack([0.0,
                           np.repeat(y / float(nn), 2)])

        res = plotUnifLogXMarkers(x2, y2, nbperdecade, logscale=False, **kwargs)

        if maxevals: # Should cover the case where maxevals is None or empty
            x3 = np.median(maxevals)
            if (x3 <= maxval and np.any(x2 <= x3)
                and not plt.getp(res[-1], 'label').startswith('best')): # TODO: HACK for not considering the best 2009 line
                y3 = y2[x2<=x3][-1]
                h = plt.plot((x3,), (y3,), marker='x', markersize=30,
                             markeredgecolor=plt.getp(res[0], 'color'),
                             ls=plt.getp(res[0], 'ls'),
                             color=plt.getp(res[0], 'color'))
                h.extend(res)
                res = h # so the last element in res still has the label.
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

    """
    reslabels = []
    reshandles = []
    ys = {}
    lh = 0
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

        h = h[-1] # we expect the label to be in the last element of h
        try:
            tmp = (x2 <= maxval)
            x2bis = x2[y2 < y2[tmp][-1]][-1]
            ys.setdefault(y2[tmp][-1], {}).setdefault(x2bis, []).append(h)
            lh += 1
        except IndexError:
            pass

    if len(show_algorithms) > 0:
        lh = min(lh, len(show_algorithms))
    if lh <= 1:
        lh = 2
    i = 0 # loop over the elements of ys
    for j in sorted(ys.keys()):
        for k in reversed(sorted(ys[j].keys())):
            #enforce best ever comes last in case of equality
            tmp = []
            for h in ys[j][k]:
                if plt.getp(h, 'label') == 'best 2009':
                    tmp.insert(0, h)
                else:
                    tmp.append(h)
            tmp.reverse()
            ys[j][k] = tmp

            for h in ys[j][k]:
                if (not plt.getp(h, 'label').startswith('_line') and
                    (len(show_algorithms) == 0 or
                     plt.getp(h, 'label') in show_algorithms)):
                    y = 0.02 + i * 0.96/(lh-1)
                    tmp = {}
                    for attr in ('lw', 'ls', 'marker',
                                 'markeredgewidth', 'markerfacecolor',
                                 'markeredgecolor', 'markersize', 'zorder'):
                        tmp[attr] = plt.getp(h, attr)
                    legx = maxval * 10
                    if 'marker' in attr:
                        legx = maxval * 9
                    reshandles.extend(plt.plot((maxval, legx), (j, y),
                                      color=plt.getp(h, 'markeredgecolor'), **tmp))
                    reshandles.append(plt.text(maxval*15, y,
                                               plt.getp(h, 'label'),
                                               horizontalalignment="left",
                                               verticalalignment="center", size=fontsize))
                    reslabels.append(plt.getp(h, 'label'))
                    #set_trace()
                    i += 1

    #plt.axvline(x=maxval, color='k') # Not as efficient?
    reshandles.append(plt.plot((maxval, maxval), (0., 1.), color='k'))
    reslabels.reverse()
    plt.xlim(xmax=maxval*x_annote_factor)
    return reslabels, reshandles

def plot(dsList, targets=defaulttargets, craftingeffort=0., **kwargs):
    """Generates a graph of the run length distribution of an algorithm.

    We display the empirical cumulative distribution function ECDF of
    the bootstrapped distribution of the expected running time (ERT)
    for an algorithm to reach the function value :py:data:`targets`.

    :param DataSetList dsList: data set for one algorithm
    :param seq targets: target function values
    :param float crafting effort: the data will be multiplied by the
                                  exponential of this value
    :param dict kwargs: additional parameters provided to plot function.
    
    :returns: handles

    """
    res = []
    assert len(dsList.dictByDim()) == 1 # We never integrate over dimensions...
    data = []
    maxevals = []
    for entry in dsList:
        for t in targets:
            x = [np.inf] * perfprofsamplesize
            runlengthunsucc = []
            evals = entry.detEvals([t])[0]
            runlengthsucc = evals[np.isnan(evals) == False] / entry.dim
            runlengthunsucc = entry.maxevals[np.isnan(evals)] / entry.dim
            if len(runlengthsucc) > 0:
                x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                     percentiles=[50],
                                     samplesize=perfprofsamplesize)[1]
            data.extend(x)
            maxevals.extend(runlengthunsucc)

    # Display data
    data = np.array(data)
    data = data[np.isnan(data)==False] # Take away the nans
    n = len(data)
    data = data[np.isinf(data)==False] # Take away the infs
    # data = data[data <= maxval] # Take away rightmost data
    data = np.exp(craftingeffort) * data  # correction by crafting effort CrE
    if len(data) == 0: # data is empty.
        res.append(plt.axhline(0., **kwargs))
    else:
        res = plotECDF(np.array(data), n=n, **kwargs)
        #plotdata(np.array(data), x_limit, maxevals,
        #                    CrE=0., kwargs=kwargs)
    if maxevals: # Should cover the case where maxevals is None or empty
        x3 = np.median(maxevals)
        if np.any(data > x3):
            y3 = float(np.sum(data <= x3)) / n
            h = plt.plot((x3,), (y3,), marker='x', markersize=30,
                         markeredgecolor=plt.getp(res[0], 'color'),
                         ls='', color=plt.getp(res[0], 'color'))
            h.extend(res)
            res = h # so the last element in res still has the label.
    return res

def main(dictAlg, targets, order=None, outputdir='.', info='default',
         verbose=True):
    """Generates a figure showing the performance of algorithms.

    From a dictionary of :py:class:`DataSetList` sorted by algorithms,
    generates the cumulative distribution function of the bootstrap
    distribution of ERT for algorithms on multiple functions for
    multiple targets altogether.

    :param dict dictAlg: dictionary of :py:class:`DataSetList` instances
                         one instance is equivalent to one algorithm,
    :param list targets: target function values
    :param list order: sorted list of keys to dictAlg for plotting order
    :param str outputdir: output directory
    :param str info: output file name suffix
    :param bool verbose: controls verbosity

    """

    tmp = dictAlgByDim(dictAlg)
    if len(tmp) != 1:
        raise Exception('We never integrate over dimension.')
    d = tmp.keys()[0]

    dictFunc = dictAlgByFun(dictAlg)

    # Collect data
    # Crafting effort correction: should we consider any?
    #CrEperAlg = {}
    #for alg in dictAlg:
        #CrE = 0.
        #if dictAlg[alg][0].algId == 'GLOBAL':
            #tmp = dictAlg[alg].dictByNoise()
            #assert len(tmp.keys()) == 1
            #if tmp.keys()[0] == 'noiselessall':
                #CrE = 0.5117
            #elif tmp.keys()[0] == 'nzall':
                #CrE = 0.6572
        #CrEperAlg[alg] = CrE

    dictData = {} # list of (ert per function) per algorithm
    dictMaxEvals = {} # list of (maxevals per function) per algorithm
    bestERT = [] # best ert per function
    funcsolved = [set()] * len(targets) # number of functions solved per target
    xbest2009 = []
    maxevalsbest2009 = []

    for f, dictAlgperFunc in dictFunc.iteritems():
        if function_IDs and f not in function_IDs:
            continue

        for j, t in enumerate(targets):
            funcsolved[j].add(f)

            # Loop over all algs, not only those with data for f
            for alg in dictAlg:
                x = [np.inf] * perfprofsamplesize
                runlengthunsucc = []
                try:
                    entry = dictAlgperFunc[alg][0] # one element per fun and per dim.
                    evals = entry.detEvals([t])[0]
                    runlengthsucc = evals[np.isnan(evals) == False] / entry.dim
                    runlengthunsucc = entry.maxevals[np.isnan(evals)] / entry.dim
                    if len(runlengthsucc) > 0:
                        x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                             percentiles=[50],
                                             samplesize=perfprofsamplesize)[1]
                except (KeyError, IndexError):
                    #set_trace()
                    txt = ('Data for algorithm %s on function %d in %d-D '
                           % (alg, f, d)
                           + 'are missing.')
                    warnings.warn(txt)

                dictData.setdefault(alg, []).extend(x)
                dictMaxEvals.setdefault(alg, []).extend(runlengthunsucc)

        if displaybest2009:
            #set_trace()
            if not bestalg.bestalgentries2009:
                bestalg.loadBBOB2009()
            bestalgentry = bestalg.bestalgentries2009[(d, f)]
            bestalgevals = bestalgentry.detEvals(targets)
            for j in range(len(targets)):
                if bestalgevals[1][j]:
                    evals = bestalgevals[0][j]
                    #set_trace()
                    runlengthsucc = evals[np.isnan(evals) == False] / bestalgentry.dim
                    runlengthunsucc = bestalgentry.maxevals[bestalgevals[1][j]][np.isnan(evals)] / bestalgentry.dim
                    x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                         percentiles=[50],
                                         samplesize=perfprofsamplesize)[1]
                else:
                    x = perfprofsamplesize * [np.inf]
                    runlengthunsucc = []
                xbest2009.extend(x)
                maxevalsbest2009.extend(runlengthunsucc)

    if order is None:
        order = dictData.keys()

    # Display data
    lines = []
    for i, alg in enumerate(order):
        try:
            data = dictData[alg]
            maxevals = dictMaxEvals[alg]
        except KeyError:
            continue

        args = styles[(i) % len(styles)]
        args['linewidth'] = 1.5
        args['markersize'] = 15.
        args['markeredgewidth'] = 1.5
        args['markerfacecolor'] = 'None'
        args['markeredgecolor'] = args['color']
        args['label'] = alg
        #args['markevery'] = perfprofsamplesize # option available in latest version of matplotlib
        #elif len(show_algorithms) > 0:
            #args['color'] = 'wheat'
            #args['ls'] = '-'
            #args['zorder'] = -1
        lines.append(plotdata(np.array(data), x_limit, maxevals,
                                  CrE=0., kwargs=args))

    if displaybest2009:
        args = {'ls': '-', 'linewidth': 1.5, 'marker': 'D', 'markersize': 7.,
                'markeredgewidth': 1.5, 'markerfacecolor': refcolor,
                'markeredgecolor': refcolor, 'color': refcolor,
                'label': 'best 2009', 'zorder': -1}
        lines.append(plotdata(np.array(xbest2009), x_limit, maxevalsbest2009,
                                  CrE = 0., kwargs=args))

    labels, handles = plotLegend(lines, x_limit)
    if True: #isLateXLeg:
        fileName = os.path.join(outputdir,'pprldmany_%s.tex' % (info))
        try:
            f = open(fileName, 'w')
            f.write(r'\providecommand{\nperfprof}{7}')
            algtocommand = {}
            for i, alg in enumerate(order):
                tmp = r'\alg%sperfprof' % numtotext(i)
                f.write(r'\providecommand{%s}{\StrLeft{%s}{\nperfprof}}' % (tmp, writeLabels(alg)))
                algtocommand[alg] = tmp
            commandnames = []
            if displaybest2009:
                tmp = r'\algzeroperfprof'
                f.write(r'\providecommand{%s}{best 2009}' % (tmp))
                algtocommand['best 2009'] = tmp

            for l in labels:
                commandnames.append(algtocommand[l])
            f.write(headleg)
            f.write(r'\mbox{%s}' % commandnames[0]) # TODO: check len(labels) > 0
            for i in range(1, len(labels)):
                f.write('\n' + r'\vfill \mbox{%s}' % commandnames[i])
            f.write(footleg)
            if verbose:
                print 'Wrote right-hand legend in %s' % fileName
        except:
            raise # TODO: Does this make sense?
        else:
            f.close()

    figureName = os.path.join(outputdir,'pprldmany_%s' % (info))
    #beautify(figureName, funcsolved, x_limit*x_annote_factor, False, fileFormat=figformat)
    beautify()

    text = 'f%s' % (consecutiveNumbers(sorted(dictFunc.keys())))
    plt.text(0.01, 0.98, text, horizontalalignment="left",
             verticalalignment="top", transform=plt.gca().transAxes)

    a = plt.gca()

    plt.xlim(xmin=1e-0, xmax=x_limit*x_annote_factor)
    xticks, labels = plt.xticks()
    tmp = []
    for i in xticks:
        tmp.append('%d' % round(np.log10(i)))
    a.set_xticklabels(tmp)
    saveFigure(figureName, figFormat=figformat, verbose=verbose)

    plt.close()

    # TODO: should return status or sthg

