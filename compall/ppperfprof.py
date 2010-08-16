#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates Empirical Cumulative Distribution of the bootstrap distribution of
the Expected Running Time (ERT) divided by the dimension."""

from __future__ import absolute_import

import os
import warnings
from pdb import set_trace
import numpy
import matplotlib.pyplot as plt
from bbob_pproc import bootstrap, bestalg
from bbob_pproc.pproc import dictAlgByDim, dictAlgByFun
from bbob_pproc.ppfig import consecutiveNumbers, saveFigure, plotUnifLogXMarkers

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

displaybest2009 = False
displaybest2010 = False
displaybestever = True

# MORE TO COME

funi = [1,2] + range(5, 15)  # 2 is paired Ellipsoid
funilipschitz = [1] + [5,6] + range(8,13) + [14] # + [13]  #13=sharp ridge, 7=step-ellipsoid 
fmulti = [3, 4] + range(15,25) # 3 = paired Rastrigin
funisep = [1,2,5]

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

def beautify():
    """Format the figure."""

    #plt.xscale('log') # Does not work with matplotlib 0.91.2
    a = plt.gca()
    a.set_xscale('log')
    #Tick label handling

    plt.xlabel('log10 of (ERT / dimension)')
    plt.ylabel('Proportion of functions')
    plt.grid(True)

    plt.ylim(0, 1)

def get_plot_args(args):
    """args is one dict element according to algorithmshortinfos
    """

    if not args.has_key('label') or args['label'] in show_algorithms:
        args['linewidth'] = 2
    elif len(show_algorithms) > 0:
        args['color'] = 'wheat'
        args['ls'] = '-'
        args['zorder'] = -1
    elif not (args.has_key('linewidth') or args.has_key('lw')):
        args['linewidth'] = 1.3
    return args

def plotPerfProf(data, maxval=None, maxevals=None, CrE=0., kwargs={}):
    """Draw a performance profile."""

    #Expect data to be a ndarray.
    x = data[numpy.isnan(data)==False] # Take away the nans
    nn = len(x)

    x = x[numpy.isinf(x)==False] # Take away the infs
    n = len(x)

    x = numpy.exp(CrE) * x  # correction by crafting effort CrE

    if n == 0:
        if maxval is None:
            res = plt.plot([], [], **kwargs) # TODO: plot a horizontal line instead
        else:
            res = plt.plot([1., maxval], [0., 0.], **kwargs)
    else:
        dictx = {}
        for i in x:
            dictx[i] = dictx.get(i, 0) + 1

        x = numpy.array(sorted(dictx))
        if maxval is None:
            maxval = max(x)
        x = x[x <= maxval]
        y = numpy.cumsum(list(dictx[i] for i in x))

        x2 = numpy.hstack([numpy.repeat(x, 2), maxval])
        y2 = numpy.hstack([0.0,
                           numpy.repeat(y / float(nn), 2)])

        if 11 < 3:
            # Downsampling
            # first try to downsample for reduced figure size, is not effective while reducing dvi is
            idx = range(0, len(x2), 2*perfprofsamplesize)
            if numpy.mod(len(x2), 2*perfprofsamplesize) != 1:
                idx.append(len(x2) - 1)
            x2 = x2[idx]
            y2 = y2[idx]

        if not 'markeredgecolor' in kwargs and 'color' in kwargs:
            kwargs['markeredgecolor'] = kwargs['color']
        res = plt.plot(x2, y2, **kwargs)
        if maxevals: # Should cover the case where maxevals is None or empty
            x3 = numpy.median(maxevals)
            if x3 <= maxval and numpy.any(x2 <= x3) and plt.getp(res[0], 'color') is not 'wheat':
                y3 = y2[x2<=x3][-1]
                plt.plot((x3,), (y3,), marker='x', markersize=15, markeredgecolor=plt.getp(res[0], 'color'),
                         ls=plt.getp(res[0], 'ls'),
                         color=plt.getp(res[0], 'color'))
                # Only take sequences for x and y!

    return res

def downsample(xdata, ydata):
    """Downsample arrays of data, zero-th column elements are evenly spaced."""

    # powers of ten 10**(i/nbperdecade)
    minidx = numpy.ceil(numpy.log10(xdata[0]) * nbperdecade)
    maxidx = numpy.floor(numpy.log10(xdata[-1]) * nbperdecade)
    alignmentdata = 10.**(numpy.arange(minidx, maxidx)/nbperdecade)
    # Look in the original data
    res = []
    for i in alignmentdata:
        res.append(ydata[xdata <= i][-1])

    return alignmentdata, res

def plotPerfProf2(data, maxval=None, maxevals=None, CrE=0., kwargs={}):
    """Draw a performance profile.
    Difference with the above: trying something smart for the markers.
    """

    #Expect data to be a ndarray.
    x = data[numpy.isnan(data)==False] # Take away the nans
    nn = len(x)

    x = x[numpy.isinf(x)==False] # Take away the infs
    n = len(x)

    x = numpy.exp(CrE) * x  # correction by crafting effort CrE

    if n == 0:
        res = list()
        res.append(plt.axhline(0., **kwargs))
    else:
        dictx = {}
        for i in x:
            dictx[i] = dictx.get(i, 0) + 1

        x = numpy.array(sorted(dictx))
        if maxval is None:
            maxval = max(x)
        x = x[x <= maxval]
        y = numpy.cumsum(list(dictx[i] for i in x))

        x2 = numpy.hstack([numpy.repeat(x, 2), maxval])
        y2 = numpy.hstack([0.0,
                           numpy.repeat(y / float(nn), 2)])

        res = plotUnifLogXMarkers(x2, y2, nbperdecade, kwargs)

        if maxevals: # Should cover the case where maxevals is None or empty
            x3 = numpy.median(maxevals)
            if x3 <= maxval and numpy.any(x2 <= x3) and plt.getp(res[0], 'color') is not 'wheat':
                y3 = y2[x2<=x3][-1]
                plt.plot((x3,), (y3,), marker='x', markersize=30,
                         markeredgecolor=plt.getp(res[0], 'color'),
                         ls=plt.getp(res[0], 'ls'),
                         color=plt.getp(res[0], 'color'))
                # Only take sequences for x and y!

    return res

def plotLegend(handles, maxval):
    """Display right-side legend."""

    ys = {}
    lh = 0
    for h in handles:
        x2 = []
        y2 = []
        for i in h:
            x2.append(plt.getp(i, "xdata"))
            y2.append(plt.getp(i, "ydata"))
        tmp = numpy.argsort(numpy.hstack(x2))
        x2 = numpy.hstack(x2)[tmp]
        y2 = numpy.hstack(y2)[tmp]
        h = h[-1]
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
                if plt.getp(h, 'label') == 'best ever':
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
                    plt.plot((maxval, legx), (j, y),
                             color=plt.getp(h, 'markeredgecolor'), **tmp)

                    plt.text(maxval*11, y,
                             plt.getp(h, 'label'), horizontalalignment="left",
                             verticalalignment="center", size=fontsize)
                    #set_trace()
                    i += 1

    #plt.axvline(x=maxval, color='k') # Not as efficient?
    plt.plot((maxval, maxval), (0., 1.), color='k')

def main(dictAlg, target, order=None, plotArgs={}, outputdir='',
          info='default', verbose=True):
    """Generates a figure showing the performance of algorithms.
    From a dictionary of DataSetList sorted by algorithms, generates the
    cumulative distribution function of the bootstrap distribution of ERT
    for algorithms on multiple functions for multiple targets altogether.
    Keyword arguments:
    dictAlg --
    target -- list of dictionaries with (function, dimension) as keys, target
    function values as values
    order -- determines the plotting order of the algorithm (used by the legend
    and in the case the algorithm has no plotting arguments specified).
    """

    xlim = x_limit
    dictData = {} # list of (ert per function) per algorithm
    dictMaxEvals = {} # list of (maxevals per function) per algorithm
    # the functions will not necessarily sorted.
    xbest = []
    maxevalsbest = []
    xbest2 = []
    maxevalsbest2 = []
    xbestever = []
    maxevalsbestever = []

    bestERT = [] # best ert per function, not necessarily sorted as well.
    funcsolved = [set()] * len(target)

    dictFunc = {}
    d = set()
    for alg, tmpdsList in dictAlg.iteritems():
        for i in tmpdsList:
            d.add(i.dim)
            tmp = dictFunc.setdefault(i.funcId, {})
            if tmp.has_key(alg):
                txt = ('Duplicate data: algorithm %s, function %f'
                       % (alg, i.funcId))
                warnings.warn(txt)
            tmp.setdefault(alg, i)
            # if the number of entries is larger than 1, the rest of
            # the data is disregarded.

    if len(d) != 1:
        raise Usage('We never integrate over dimension.')
    d = d.pop()

    for f, tmpdictAlg in dictFunc.iteritems():
        if function_IDs and f not in function_IDs:
            continue

        for j, t in enumerate(target):
            try:
                if numpy.isnan(t[(f, d)]):
                    continue
            except KeyError:
                continue

            funcsolved[j].add(f)

            # Loop over all algs, not only those with data for f
            for alg in dictAlg:
                x = [numpy.inf] * perfprofsamplesize
                y = numpy.inf
                runlengthunsucc = []

                try:
                    entry = tmpdictAlg[alg]
                    runlengthunsucc = entry.maxevals / entry.dim
                    for line in entry.evals:
                        if line[0] <= t[(f, d)]:
                            tmp = line[1:]
                            runlengthsucc = tmp[numpy.isfinite(tmp)] / entry.dim
                            runlengthunsucc = entry.maxevals[numpy.isnan(tmp)] / entry.dim
                            x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                                 percentiles=[50],
                                                 samplesize=perfprofsamplesize)[1]
                            break

                except KeyError:
                    txt = ('Data for algorithm %s on function %d in %d-D '
                           % (alg, f, d)
                           + 'are missing.')
                    warnings.warn(txt)
                    #raise Usage(txt)
                    #pass

                dictData.setdefault(alg, []).extend(x)
                dictMaxEvals.setdefault(alg, []).extend(runlengthunsucc)

        if displaybest2009:
            if not bestalg.bestalgentries2009:
                bestalg.loadBBOB2009()
            bestalgentry = bestalg.bestalgentries2009[(d, f)]

            tmptargets = list(t[(f, d)] for t in target)
            bestalgevals = bestalgentry.detEvals(tmptargets)
            for j in range(len(tmptargets)):
                if bestalgevals[1][j]:
                    evals = bestalgevals[0][j]
                    #set_trace()
                    runlengthsucc = evals[numpy.isnan(evals) == False] / bestalgentry.dim
                    runlengthunsucc = bestalgentry.maxevals[bestalgevals[1][j]][numpy.isnan(evals)] / bestalgentry.dim
                    x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                         percentiles=[50],
                                         samplesize=perfprofsamplesize)[1]
                else:
                    x = perfprofsamplesize * [numpy.inf]
                    runlengthunsucc = []
                xbest.extend(x)
                maxevalsbest.extend(runlengthunsucc)

        if displaybest2010:
            if not bestalg.bestalgentries2010:
                bestalg.loadBBOB2010()
            bestalgentry = bestalg.bestalgentries2010[(d, f)]
            #set_trace()
            tmptargets = list(t[(f, d)] for t in target)
            bestalgevals = bestalgentry.detEvals(tmptargets)
            for j in range(len(tmptargets)):
                if bestalgevals[1][j]:
                    evals = bestalgevals[0][j]
                    #set_trace()
                    runlengthsucc = evals[numpy.isnan(evals) == False] / bestalgentry.dim
                    runlengthunsucc = bestalgentry.maxevals[bestalgevals[1][j]][numpy.isnan(evals)] / bestalgentry.dim
                    x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                         percentiles=[50],
                                         samplesize=perfprofsamplesize)[1]
                else:
                    x = perfprofsamplesize * [numpy.inf]
                    runlengthunsucc = []
                xbest2.extend(x)
                maxevalsbest2.extend(runlengthunsucc)

        if displaybestever:
            if not bestalg.bestalgentriesever:
                bestalg.loadBBOBever()
            bestalgentry = bestalg.bestalgentriesever[(d, f)]
            #set_trace()
            tmptargets = list(t[(f, d)] for t in target)
            bestalgevals = bestalgentry.detEvals(tmptargets)
            for j in range(len(tmptargets)):
                if bestalgevals[1][j]:
                    evals = bestalgevals[0][j]
                    #set_trace()
                    runlengthsucc = evals[numpy.isnan(evals) == False] / bestalgentry.dim
                    runlengthunsucc = bestalgentry.maxevals[bestalgevals[1][j]][numpy.isnan(evals)] / bestalgentry.dim
                    x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                         percentiles=[50],
                                         samplesize=perfprofsamplesize)[1]
                else:
                    x = perfprofsamplesize * [numpy.inf]
                    runlengthunsucc = []
                xbestever.extend(x)
                maxevalsbestever.extend(runlengthunsucc)

    #picklefilename = os.path.join(outputdir,'perfprofdata_%s.pickle' %(info))
    #f = file(picklefilename, 'w')
    #pickle.dump(dictData, f)
    #pickle.dump(dictMaxEvals, f)
    #f.close()

    if order is None:
        order = dictData.keys()

    lines = []
    for i, alg in enumerate(order):
        maxevals = []

        try:
            data = dictData[alg]
            maxevals = dictMaxEvals[alg]
        except KeyError:
            continue

        CrE = 0
        # need to know noisy or non-noisy functions here!
        if function_IDs and max(function_IDs) < 100:  # non-noisy functions
            if alg[-6:] == 'GLOBAL' and len(function_IDs) > 5:
                CrE = 0.5117
                # print 'GLOBAL corrected'
        elif function_IDs and min(function_IDs) > 100 :  # noisy functions
            if alg[-6:] == 'GLOBAL'  and len(function_IDs) > 5:
                CrE = 0.6572
        else:
            pass
            # print 'mixing noisy and non-noisy functions will yield questionable results'

        #Get one element in the set of the algorithm description
        try:
            elem = set((i.algId, i.comment) for i in dictAlg[alg]).pop()
            try:
                tmp = plotArgs[elem]
            except KeyError:
                tmp = {}
            lines.append(plotPerfProf2(numpy.array(data), xlim, maxevals,
                                       CrE=CrE, kwargs=get_plot_args(tmp)))
        except KeyError:
            #No data
            pass

    if displaybest2009:
        args = {'ls': '-', 'linewidth': 1.5, 'marker': 'D', 'markersize': 7.,
                'markeredgewidth': 1.5, 'markerfacecolor': 'wheat',
                'markeredgecolor': 'wheat', 'color': 'wheat',
                'label': 'best 2009', 'zorder': -1}
        lines.append(plotPerfProf2(numpy.array(xbest), xlim, maxevalsbest,
                                   CrE = 0., kwargs=args))
    if displaybest2010:
        args = {'ls': '-', 'linewidth': 3, 'marker': 'D', 'markersize': 7.,
                'markeredgewidth': 3, 'markerfacecolor': 'wheat',
                'markeredgecolor': 'wheat', 'color': 'wheat',
                'label': 'best 2010', 'zorder': -1}
        lines.append(plotPerfProf2(numpy.array(xbest2), xlim, maxevalsbest2,
                                   CrE = 0., kwargs=args))

    if displaybestever:
        args = {'ls': '-', 'linewidth': 3, 'marker': 'D', 'markersize': 7.,
                'markeredgewidth': 3, 'markerfacecolor': 'wheat',
                'markeredgecolor': 'wheat', 'color': 'wheat',
                'label': 'best 09/10', 'zorder': -1}
        lines.append(plotPerfProf2(numpy.array(xbestever), xlim, maxevalsbestever,
                                   CrE = 0., kwargs=args))

    plotLegend(lines, xlim)

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    beautify()

    a = plt.gca()

    if not funcsolved is None:
        txt = ''

        if len(list(i for i in funcsolved if len(i) > 0)) > 1:
            txt = '(%d' % len(funcsolved[0])
            for i in range(1, len(funcsolved)):
                if len(funcsolved[i]) > 0:
                    txt += ', %d' % len(funcsolved[i])
            txt += ') = '

        txt += ('%d funcs' % numpy.sum(len(i) for i in funcsolved))

        plt.text(0.01, 1.01, txt, horizontalalignment='left',
                 verticalalignment="bottom", transform=plt.gca().transAxes)

    if save_zoom:  # first half only, takes about 2.5 seconds
        plt.xlim(xmin=1e-2, xmax=1e3)
        xticks, labels = plt.xticks()
        tmp = []
        for i in xticks:
            tmp.append('%d' % round(numpy.log10(i)))
        a.set_xticklabels(tmp)

        saveFigure(figureName + 'a', figFormat=figformat)

    if save_zoom:  # second half only
        plt.xlim(xmin=1e2, xmax=xlim*x_annote_factor)
        xticks, labels = plt.xticks()
        tmp = []
        for i in xticks:
            tmp.append('%d' % round(numpy.log10(i)))
        a.set_xticklabels(tmp)
        saveFigure(figureName + 'b', figFormat=figformat, verbose=verbose)

    plt.xlim(xmin=1e-0, xmax=xlim*x_annote_factor)
    xticks, labels = plt.xticks()
    tmp = []
    for i in xticks:
        tmp.append('%d' % round(numpy.log10(i)))
    a.set_xticklabels(tmp)
    saveFigure(figureName, figFormat=figformat, verbose=verbose)

    plt.close()

def main2(dictAlg, targets, order=None, plotArgs={}, outputdir='',
          info='default', verbose=True):
    """Generates a figure showing the performance of algorithms.
    From a dictionary of DataSetList sorted by algorithms, generates the
    cumulative distribution function of the bootstrap distribution of ERT
    for algorithms on multiple functions for multiple targets altogether.

    differences with method main: Symbols instead of lines

    Keyword arguments:
    dictAlg --
    targets -- list of target function values
    order -- determines the plotting order of the algorithm (used by the legend
    and in the case the algorithm has no plotting arguments specified).
    """

    xlim = x_limit # variable defined in header

    tmp = dictAlgByDim(dictAlg)
    if len(tmp) != 1:
        raise Usage('We never integrate over dimension.')
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
    xbestever = []
    maxevalsbestever = []

    for f, dictAlgperFunc in dictFunc.iteritems():
        if function_IDs and f not in function_IDs:
            continue

        for j, t in enumerate(targets):
            funcsolved[j].add(f) # TODO: weird

            # Loop over all algs, not only those with data for f
            for alg in dictAlg:
                x = [numpy.inf] * perfprofsamplesize
                runlengthunsucc = []
                try:
                    entry = dictAlgperFunc[alg][0]
                    evals = entry.detEvals([t])[0]
                    runlengthsucc = evals[numpy.isnan(evals) == False] / entry.dim
                    runlengthunsucc = entry.maxevals[numpy.isnan(evals)] / entry.dim
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

        if displaybestever:
            #set_trace()
            if not bestalg.bestalgentriesever:
                bestalg.loadBBOBever()
            bestalgentry = bestalg.bestalgentriesever[(d, f)]
            bestalgevals = bestalgentry.detEvals(targets)
            for j in range(len(targets)):
                if bestalgevals[1][j]:
                    evals = bestalgevals[0][j]
                    #set_trace()
                    runlengthsucc = evals[numpy.isnan(evals) == False] / bestalgentry.dim
                    runlengthunsucc = bestalgentry.maxevals[bestalgevals[1][j]][numpy.isnan(evals)] / bestalgentry.dim
                    x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                         percentiles=[50],
                                         samplesize=perfprofsamplesize)[1]
                else:
                    x = perfprofsamplesize * [numpy.inf]
                    runlengthunsucc = []
                xbestever.extend(x)
                maxevalsbestever.extend(runlengthunsucc)

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
        lines.append(plotPerfProf2(numpy.array(data), xlim, maxevals,
                                   CrE=0., kwargs=args))

    if displaybestever:
        args = {'ls': '-', 'linewidth': 1.5, 'marker': 'D', 'markersize': 7.,
                'markeredgewidth': 1.5, 'markerfacecolor': 'wheat',
                'markeredgecolor': 'wheat', 'color': 'wheat',
                'label': 'best ever', 'zorder': -1}
        lines.append(plotPerfProf2(numpy.array(xbestever), xlim, maxevalsbestever,
                                   CrE = 0., kwargs=args))

    plotLegend(lines, xlim)

    figureName = os.path.join(outputdir,'ppperfprof_%s' % (info))
    #beautify(figureName, funcsolved, xlim*x_annote_factor, False, fileFormat=figformat)
    beautify()

    text = 'f%s' % (consecutiveNumbers(sorted(dictFunc.keys())))
    plt.text(0.01, 0.98, text, horizontalalignment="left",
             verticalalignment="top", transform=plt.gca().transAxes)

    a = plt.gca()

    plt.xlim(xmin=1e-0, xmax=xlim*x_annote_factor)
    xticks, labels = plt.xticks()
    tmp = []
    for i in xticks:
        tmp.append('%d' % round(numpy.log10(i)))
    a.set_xticklabels(tmp)
    saveFigure(figureName, figFormat=figformat, verbose=verbose)

    plt.close()

def mainbis(dictAlg, targets, order=None, plotArgs={}, outputdir='',
            info='default', verbose=True):
    """Generates a figure showing the performance of algorithms.
    From a dictionary of DataSetList sorted by algorithms, generates the
    cumulative distribution function of the bootstrap distribution of ERT
    for algorithms on multiple functions for multiple targets altogether.

    differences with method main: Symbols instead of lines
    Used to display best 2009, best 2010 and best ever.

    Keyword arguments:
    dictAlg --
    targets -- list of target function values
    order -- determines the plotting order of the algorithm (used by the legend
    and in the case the algorithm has no plotting arguments specified).
    """

    xlim = x_limit # variable defined in header

    tmp = dictAlgByDim(dictAlg)
    if len(tmp) != 1:
        raise Usage('We never integrate over dimension.')
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
    xbest = []
    maxevalsbest = []
    xbest2 = []
    maxevalsbest2 = []
    xbestever = []
    maxevalsbestever = []

    for f, dictAlgperFunc in dictFunc.iteritems():
        if function_IDs and f not in function_IDs:
            continue

        for j, t in enumerate(targets):
            funcsolved[j].add(f) # TODO: weird

            # Loop over all algs, not only those with data for f
            for alg in dictAlg:
                x = [numpy.inf] * perfprofsamplesize
                runlengthunsucc = []
                try:
                    entry = dictAlgperFunc[alg][0]
                    evals = entry.detEvals([t])[0]
                    runlengthsucc = evals[numpy.isnan(evals) == False] / entry.dim
                    runlengthunsucc = entry.maxevals[numpy.isnan(evals)] / entry.dim
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

        if not bestalg.bestalgentries2009:
            bestalg.loadBBOB2009()
        bestalgentry = bestalg.bestalgentries2009[(d, f)]
        bestalgevals = bestalgentry.detEvals(targets)
        for j in range(len(targets)):
            if bestalgevals[1][j]:
                evals = bestalgevals[0][j]
                #set_trace()
                runlengthsucc = evals[numpy.isnan(evals) == False] / bestalgentry.dim
                runlengthunsucc = bestalgentry.maxevals[bestalgevals[1][j]][numpy.isnan(evals)] / bestalgentry.dim
                x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                     percentiles=[50],
                                     samplesize=perfprofsamplesize)[1]
            else:
                x = perfprofsamplesize * [numpy.inf]
                runlengthunsucc = []
            xbest.extend(x)
            maxevalsbest.extend(runlengthunsucc)

        if not bestalg.bestalgentries2010:
            bestalg.loadBBOB2010()
        bestalgentry = bestalg.bestalgentries2010[(d, f)]
        bestalgevals = bestalgentry.detEvals(targets)
        for j in range(len(targets)):
            if bestalgevals[1][j]:
                evals = bestalgevals[0][j]
                #set_trace()
                runlengthsucc = evals[numpy.isnan(evals) == False] / bestalgentry.dim
                runlengthunsucc = bestalgentry.maxevals[bestalgevals[1][j]][numpy.isnan(evals)] / bestalgentry.dim
                x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                     percentiles=[50],
                                     samplesize=perfprofsamplesize)[1]
            else:
                x = perfprofsamplesize * [numpy.inf]
                runlengthunsucc = []
            xbest2.extend(x)
            maxevalsbest2.extend(runlengthunsucc)

        if not bestalg.bestalgentriesever:
            bestalg.loadBBOBever()
        bestalgentry = bestalg.bestalgentriesever[(d, f)]
        bestalgevals = bestalgentry.detEvals(targets)
        for j in range(len(targets)):
            if bestalgevals[1][j]:
                evals = bestalgevals[0][j]
                #set_trace()
                runlengthsucc = evals[numpy.isnan(evals) == False] / bestalgentry.dim
                runlengthunsucc = bestalgentry.maxevals[bestalgevals[1][j]][numpy.isnan(evals)] / bestalgentry.dim
                x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                     percentiles=[50],
                                     samplesize=perfprofsamplesize)[1]
            else:
                x = perfprofsamplesize * [numpy.inf]
                runlengthunsucc = []
            xbestever.extend(x)
            maxevalsbestever.extend(runlengthunsucc)

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
        lines.append(plotPerfProf2(numpy.array(data), xlim, maxevals,
                                   CrE=0., kwargs=args))

    args = {'ls': '-', 'linewidth': 3, 'marker': 'D', 'markersize': 15.,
            'markeredgewidth': 3, 'markerfacecolor': 'green',
            'markeredgecolor': 'green', 'color': 'green',
            'label': 'best 2009', 'zorder': -1}
    lines.append(plotPerfProf2(numpy.array(xbest), xlim, maxevalsbest,
                               CrE = 0., kwargs=args))

    args = {'ls': '-', 'linewidth': 3, 'marker': 'D', 'markersize': 15.,
            'markeredgewidth': 3, 'markerfacecolor': 'blue',
            'markeredgecolor': 'blue', 'color': 'blue',
            'label': 'best 2010', 'zorder': -1}
    lines.append(plotPerfProf2(numpy.array(xbest2), xlim, maxevalsbest2,
                               CrE = 0., kwargs=args))

    args = {'ls': '-', 'linewidth': 3, 'marker': 'D', 'markersize': 15.,
            'markeredgewidth': 3, 'markerfacecolor': 'red',
            'markeredgecolor': 'red', 'color': 'red',
            'label': 'best ever', 'zorder': -1}
    lines.append(plotPerfProf2(numpy.array(xbestever), xlim, maxevalsbestever,
                               CrE = 0., kwargs=args))

    plotLegend(lines, xlim)

    figureName = os.path.join(outputdir,'ppperfprof_%s' % (info))
    #beautify(figureName, funcsolved, xlim*x_annote_factor, False, fileFormat=figformat)
    beautify()

    a = plt.gca()

    plt.xlim(xmin=1e-0, xmax=xlim*x_annote_factor)
    xticks, labels = plt.xticks()
    tmp = []
    for i in xticks:
        tmp.append('%d' % round(numpy.log10(i)))
    a.set_xticklabels(tmp)
    saveFigure(figureName, figFormat=figformat, verbose=verbose)

    plt.close()
