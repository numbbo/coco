#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os
import warnings
import numpy # According to PEP 8 imports should be on different lines
import matplotlib.pyplot as plt
from bbob_pproc import bootstrap
from pdb import set_trace

"""Generates Empirical Cumulative Distribution of the bootstrap distribution of
the Expected Running Time (ERT) divided by the dimension."""

best = ('AMaLGaM IDEA', 'iAMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES', 'IPOP-SEP-CMA-ES',  
   'BFGS', 'NELDER (Han)', 'NELDER (Doe)', 'NEWUOA', 'full NEWUOA', 'GLOBAL', 'MCS (Neum)',
   'DIRECT', 'DASA', 'POEMS', 'Cauchy EDA', 'Monte Carlo')

bestnoisy = ()

nikos = ('AMaLGaM IDEA', 'VNS (Garcia)', 'MA-LS-Chain', 'BIPOP-CMA-ES', '(1+1)-CMA-ES', 'G3-PCX', 'NEWUOA', 
         'Monte Carlo', 'NELDER (Han)', 'NELDER (Doe)', 'GLOBAL', 'MCS (Neum)')

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

third = ('POEMS', 'VNS (Garcia)', 'DE-PSO', 'EDA-PSO', 'PSO_Bounds', 'PSO', 'AMaLGaM IDEA', 'iAMaLGaM IDEA',
         'MA-LS-Chain', 'DASA', 'BayEDAcG')

# MORE TO COME

funi = [1] + range(5, 15)  # without paired Ellipsoid 
fmulti = [4] + range(15,25) # without paired Rastrigin
funisep = [1,2,5]

# input parameter settings
show_algorithms = () # ()==all
function_IDs = [8]  # range(103, 131, 3)   # displayed functions
function_IDs = range(1,999)  # sep ros high mul mulw == 1, 6, 10, 15, 20, 101, 107, 122, 
x_limit = 1e7   # noisy: 1e8, otherwise: 1e7. maximal run length shown


save_zoom = False  # save zoom into left and right part of the figures
perfprofsamplesize = 100 # resolution of the performance profile.
dpi_global_var = 60  # 100 ==> 800x600 (~160KB), 120 ==> 960x720 (~200KB), 150 ==> 1200x900 (~300KB) looks ugly in latex

def get_plot_args(args):
    """args is one dict element according to algorithmshortinfos
    """
    if args['label'] in show_algorithms:
        args['linewidth'] = 2
    elif len(show_algorithms) > 0:
        args['color'] = 'wheat'
        args['ls'] = '-'
        args['zorder'] = -1
    return args

def beautify(figureName='perfprofile', funcsolved=None, maxval=None,
             isLegend=True, fileFormat=('eps', 'png')):
    """Format the figure."""

    plt.xscale('log')
    plt.xlabel('Running length / dimension')
    plt.ylabel('Proportion of functions')
    plt.grid(True)

    if not funcsolved is None and funcsolved:
        txt = ''
        try:
            if len(list(i for i in funcsolved if i > 0)) > 1:
                txt = '(%d' % funcsolved[0]
                for i in range(1, len(funcsolved)):
                    if funcsolved[i] > 0:
                        txt += ', %d' % funcsolved[i]
                txt += ') = '
            txt += '%d funcs' % numpy.sum(funcsolved)
        except TypeError:
            txt = '%d funcs' % funcsolved

        #plt.figtext(0.01, 1.01, txt, horizontalalignment='left',
                 #verticalalignment="bottom")
        plt.text(0.01, 1.01, txt, horizontalalignment='left',
                 verticalalignment="bottom", transform=plt.gca().transAxes)

    if isLegend:
        plt.legend(loc='best')

    plt.ylim(0, 1)

    plt.xlim(xmax=1e3) #TODO: save default value?
    plt.xlim(xmin=1e-2)
    if save_zoom:  # first half only, takes about 2.5 seconds
        #set_trace()
        for entry in fileFormat:
            plt.savefig(figureName + 'a.' + entry, dpi = 300, format = entry)

    plt.xlim(xmin=1e2)
    if not maxval is None: # TODO: reset to default value...
        plt.xlim(xmax=maxval)
    if save_zoom:  # second half only
        #plt.ylim(0, 1)
        for entry in fileFormat:
            plt.savefig(figureName + 'b.' + entry, dpi = 300, format = entry)

    plt.xlim(xmin=1e-0)
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = dpi_global_var, format = entry)


def plotPerfProf(data, maxval=None, maxevals=None, isbeautify=True, order=None, CrE=0,
                 kwargs={}):
    """Draw a performance profile.
    """

    #Expect data to be a ndarray.
    x = data[numpy.isnan(data)==False] # Take away the nans
    #set_trace()
    nn = len(x)

    x = x[numpy.isinf(x)==False] # Take away the infs
    n = len(x)

    x = numpy.exp(CrE) * x  # correction by crafting effort CrE

    if n == 0:
        #TODO: problem if no maxval
        if maxval is None:
            res = plt.plot([], [], **kwargs) # TODO: plot a horizontal line instead?
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
        if not maxevals is None:
            x3 = numpy.median(maxevals)
            if x3 <= maxval and numpy.any(x2 <= x3):
                y3 = y2[x2<=x3][-1]
                plt.plot((x3,), (y3,), marker='x',
                         ls=plt.getp(res[0], 'ls'),
                         color=plt.getp(res[0], 'color'))
                # Only take sequences for x and y!

    return res

def plotLegend(handles, maxval):
    """Display right-side legend.
    """
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
                if len(show_algorithms) == 0 or plt.getp(h, 'label') in show_algorithms: 
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

def main(dsList, target, order=None, plotArgs={}, outputdir='',
          info='default', fileFormat=('eps', 'png'), verbose=True):
    """From a dataSetList, generates the performance profiles for multiple
    functions for multiple targets altogether.
    keyword arguments:
    target: list of dictionaries with (function, dimension) as keys, target
    function values as values
    order: determines the plotting order of the algorithm (used by the legend
    and in the case the algorithm has no plotting arguments specified).

    """

    xlim = x_limit
    dictData = {} # list of (ert per function) per algorithm
    dictMaxEvals = {} # list of (maxevals per function) per algorithm
    # the functions will not necessarily sorted.

    bestERT = [] # best ert per function, not necessarily sorted as well.
    funcsolved = [0] * len(target)

    dictFunc = dsList.dictByFunc()

    for f, samefuncEntries in dictFunc.iteritems():
        if f not in function_IDs:
           continue
        dictDim = samefuncEntries.dictByDim()
        for d, samedimEntries in dictDim.iteritems():
            dictAlg = samedimEntries.dictByAlg()

            for j, t in enumerate(target):
                try:
                    #set_trace()
                    if numpy.isnan(t[(f, d)]):
                        continue
                except KeyError:
                    continue
                funcsolved[j] += 1

                for alg, entry in dictAlg.iteritems():
                    # entry is supposed to be a single item DataSetList
                    entry = entry[0]
                    x = [numpy.inf] * perfprofsamplesize
                    y = numpy.inf
                    runlengthunsucc = entry.maxevals / entry.dim
                    for line in entry.evals:
                        if line[0] <= t[(f, d)]:
                            tmp = line[1:]
                            runlengthsucc = tmp[numpy.isfinite(tmp)] / entry.dim
                            runlengthunsucc = entry.maxevals[numpy.isnan(tmp)] / entry.dim
                            #if len(runlengthunsucc) > 0:
                            #if info == '20_fE1.0':
                            #    set_trace()
                            x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                                 percentiles=[50],
                                                 samplesize=perfprofsamplesize)[1]
                            #else: # Problem in this case due to samplesize.
                            #    x = runlengthsucc

                            break

                    dictData.setdefault(alg, []).extend(x)
                    dictMaxEvals.setdefault(alg, []).extend(runlengthunsucc)
                    #TODO: there may be addition for every target... is it the desired behaviour?

    if order is None:
        order = dictData.keys()

    lines = []
    for i, alg in enumerate(order):
        data = []
        maxevals = []
        isData = False
        for elem in alg:
            if dictData.has_key(elem):
                isData = True
                data.extend(dictData[elem])
                maxevals.extend(dictMaxEvals[elem])

        if isData:
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

            lines.append(plotPerfProf(numpy.array(data),
                         xlim, maxevals, order=(i, len(order)), CrE=CrE, 
                         kwargs=get_plot_args(plotArgs[elem]), )) #elem is an element in alg...

    # re-plot show_algorithms in front, does not work
    #for i, alg in enumerate(order):
    #    if plotArgs[alg[0]] in show_algorithms:
    #        lines.append(plotPerfProf(numpy.array(data),
    #                     xlim, maxevals, order=(i, len(order)),
    #                     kwargs=get_plot_args(plotArgs[alg[0]]))) #elem is an element in alg...

    plotLegend(lines, xlim)

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    #set_trace()
    beautify(figureName, funcsolved, xlim*100, False, fileFormat=fileFormat)

    plt.close()
