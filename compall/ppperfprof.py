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
from bbob_pproc import bootstrap

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

# MORE TO COME

funi = [1] + range(5, 15)  # without paired Ellipsoid
funilipschitz = [1] + [5,6] + range(8,13) + [14] # + [13]  #13=sharp ridge, 7=step-ellipsoid 
fmulti = [4] + range(15,25) # without paired Rastrigin
funisep = [1,2,5]

# input parameter settings
#show_algorithms = eseda + ('BFGS',) # ()==all
#show_algorithms = ('IPOP-SEP-CMA-ES', 'IPOP-CMA-ES', 'BIPOP-CMA-ES',)
#show_algorithms = ('IPOP-SEP-CMA-ES', 'IPOP-CMA-ES', 'BIPOP-CMA-ES',
#'avg NEWUOA', 'NEWUOA', 'full NEWUOA', 'BFGS', 'MCS (Neum)', 'GLOBAL', 'NELDER (Han)',
#'NELDER (Doe)', 'Monte Carlo') # ()==all
show_algorithms = ()
function_IDs = fmulti  # range(103, 131, 3)   # displayed functions
function_IDs = range(1,999)  # sep ros high mul mulw == 1, 6, 10, 15, 20, 101, 107, 122, 
#function_IDs = [1,2,3,4,5] # separable functions
#function_IDs = [6,7,8,9]   # moderate functions
#function_IDs = [10,11,12,13,14] # ill-conditioned functions
#function_IDs = [15,16,17,18,19] # multi-modal functions
#function_IDs = [20,21,22,23,24] # weak structure functions
#function_IDs = range(122,131)  # noise-free testbed
#function_IDs = range(101,131) # noisy testbed

x_limit = 1e7   # noisy: 1e8, otherwise: 1e7. maximal run length shown
x_annote_factor = 90

save_zoom = False  # save zoom into left and right part of the figures
perfprofsamplesize = 100  # number of bootstrap samples drawn for each fct+target in the performance profile
dpi_global_var = 100  # 100 ==> 800x600 (~160KB), 120 ==> 960x720 (~200KB), 150 ==> 1200x900 (~300KB) looks ugly in latex

def beautify(figureName='perfprofile', funcsolved=None, maxval=None,
             isLegend=True, fileFormat=('pdf', 'eps')):
    """Format the figure."""

    #plt.xscale('log') # Does not work with matplotlib 0.91.2
    a = plt.gca()
    a.set_xscale('log')
    plt.xlabel('Running length / dimension')
    plt.ylabel('Proportion of functions')
    plt.grid(True)

    if not funcsolved is None and funcsolved:
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

    if isLegend:
        plt.legend(loc='best')

    plt.ylim(0, 1)

    plt.xlim(xmax=1e3)
    plt.xlim(xmin=1e-2)
    if save_zoom:  # first half only, takes about 2.5 seconds
        for entry in fileFormat:
            plt.savefig(figureName + 'a.' + entry, dpi = 300, format = entry)

    plt.xlim(xmin=1e2)
    if not maxval is None: # TODO: reset to default value...
        plt.xlim(xmax=maxval)
    if save_zoom:  # second half only
        for entry in fileFormat:
            plt.savefig(figureName + 'b.' + entry, dpi = 300, format = entry)

    plt.xlim(xmin=1e-0)
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = dpi_global_var, format = entry)

def get_plot_args(args):
    """args is one dict element according to algorithmshortinfos
    """

    if not args.has_key('label') or args['label'] in show_algorithms:
        args['linewidth'] = 2
    elif len(show_algorithms) > 0:
        args['color'] = 'wheat'
        args['ls'] = '-'
        args['zorder'] = -1
    return args

def plotPerfProf(data, maxval=None, maxevals=None, isbeautify=True, order=None, CrE=0,
                 kwargs={}):
    """Draw a performance profile.
    """

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

        if 11 < 3:  # to be removed
            # first try to downsample for reduced figure size, is not effective while reducing dvi is
            idx = range(0, len(x2), 5)
            if numpy.mod(len(x2), 5) != 1:
                idx.append(len(x2) - 1)
            x2 = x2[idx]
            y2 = y2[idx]

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

def plotLegend(handles, maxval):
    """Display right-side legend."""

    ys = {}
    lh = 0
    for h in handles:
        h = h[0]
        x2 = plt.getp(h, "xdata")
        y2 = plt.getp(h, "ydata")
        try:
            tmp = sum(x2 <= maxval) - 1
            x2bis = x2[sum(y2 < y2[tmp]) - 1]
            ys.setdefault(y2[tmp], {}).setdefault(x2bis, []).append(h)
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
            for h in ys[j][k]:
                if (not plt.getp(h, 'label').startswith('_line') and
                    (len(show_algorithms) == 0 or
                     plt.getp(h, 'label') in show_algorithms)):
                    y = 0.02 + i * 0.96/(lh-1)
                    plt.plot((maxval, maxval*10), (j, y),
                             ls=plt.getp(h, 'ls'), color=plt.getp(h, 'color'),
                             lw=plt.getp(h, 'lw'))
                    plt.text(maxval*11, y,
                             plt.getp(h, 'label'), horizontalalignment="left",
                             verticalalignment="center")
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
        if f not in function_IDs:
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
        if max(function_IDs) < 100:  # non-noisy functions
            if alg[0][0] == 'GLOBAL':
                CrE = 0.5117
                # print 'GLOBAL corrected'
        elif min(function_IDs) > 100 :  # noisy functions
            if alg[0][0] == 'GLOBAL':
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
            lines.append(plotPerfProf(numpy.array(data),
                         xlim, maxevals, order=(i, len(order)), CrE=CrE,
                         kwargs=get_plot_args(tmp)))
        except KeyError:
            #No data
            pass

    plotLegend(lines, xlim)

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    beautify(figureName, funcsolved, xlim*x_annote_factor, False, fileFormat=figformat)

    plt.close()
