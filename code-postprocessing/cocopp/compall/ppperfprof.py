#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""For generating performance profiles. This module is not in use (anymore or not yet?).

**Example**

.. plot::
   :width: 75%

   import urllib
   import tarfile
   import glob
   from pylab import *
   import pickle
   import cocopp
   import cocopp.compall.ppperfprof
   import cocopp.bestalg

   # Collect and unarchive data
   dsets = {}
   for alg in cocopp.compall.ppperfprof.best:
       for date in ('2010', '2009'):
           try:
               dataurl = 'http://coco.lri.fr/BBOB'+date+'/pythondata/' + alg + '.tar.gz'
               filename, headers = urllib.urlretrieve(dataurl)
               archivefile = tarfile.open(filename)
               archivefile.extractall()  # write to disc
               dsets[alg] = cocopp.load(glob.glob('BBOB'+date+'pythondata/' + alg + '/ppdata_f0*_20.pickle'))
           except:
               pass

   # plot the profiles
   figure()
   # cocopp.compall.ppperfprof.plotmultiple(dsets, dsref=cocopp.bestalg.bestAlgorithmEntries)

"""
from __future__ import absolute_import

import os
import numpy as np
import matplotlib.pyplot as plt
from .. import genericsettings, toolsstats, bestalg
from ..pprldistr import plotECDF, beautifyECDF
from ..ppfig import save_figure, logxticks
from .pprldmany import plotLegend

__all__ = ['beautify', 'main', 'plot']

best = ('AMALGAM', 'iAMALGAM', 'VNS', 'MA-LS-CHAIN', 'BIPOP-CMA-ES', 'IPOP-ACTCMA-ES', 'MOS', 'IPOP-SEP-CMA-ES',
   'BFGS', 'NELDER', 'NELDERDOERR', 'NEWUOA', 'FULLNEWUOA', 'GLOBAL', 'MCS',
   'DIRECT', 'DASA', 'POEMS', 'Cauchy-EDA', 'RANDOMSEARCH')

# input parameter settings
perfprofsamplesize = genericsettings.simulated_runlength_bootstrap_sample_size  # number of bootstrap samples drawn for each fct+target in the performance profile
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

defaulttargets = tuple(10**np.r_[-8:2:0.2])

def beautify():
    """Customize figure presentation."""

    #plt.xscale('log') # Does not work with matplotlib 0.91.2
    a = plt.gca()
    a.set_xscale('log')
    #Tick label handling

    plt.xlabel('log10 of (aRT / aRTref)')
    plt.ylabel('Proportion of functions')
    logxticks()
    beautifyECDF()

def plotmultiple(dictAlg, dsref=None, order=None, targets=defaulttargets,
                 istoolsstats=False, rhleg=True):
    """Generate performance profile figure.

    :param dict dictAlg: dictionary of :py:class:`DataSetList` instances
                         one instance = one algorithm
    :param DataSetList dsref: reference data set
    :param seq targets: target function values
    :param bool istoolsstats: if True, uses bootstrapped distribution
    :param bool rhleg: if True, displays the right-hand legend

    """

    if not dsref:
        dsref = bestalg.generate(dictAlg)
    if not order:
        order = dictAlg.keys()
    lines = []
    for i, k in enumerate(order):
        args = styles[(i) % len(styles)]
        args['linewidth'] = 1.5
        args['markersize'] = 15.
        args['markeredgewidth'] = 1.5
        args['markerfacecolor'] = 'None'
        args['markeredgecolor'] = args['color']
        lines.append(plot(dictAlg[k], dsref, targets, istoolsstats, label=k,
                          **args))
    #plt.xlim(xmin=1e-0, xmax=xlim*x_annote_factor)
    beautify()
    if rhleg:
        plotLegend(lines, plt.xlim()[1])

def plot(dsList, dsref, targets=defaulttargets, istoolsstats=False, **kwargs):
    """Generates a graph showing the performance profile of an algorithm.

    We display the empirical cumulative distribution function ECDF of
    the bootstrapped distribution of the average running time (aRT)
    for an algorithm to reach the function value :py:data:`targets`
    normalized by the aRT of the reference algorithm for these
    targets.

    :param DataSetList dsList: data set for one algorithm
    :param DataSetList dsref: reference data set for normalization
    :param seq targets: target function values
    :param dict kwargs: additional parameters provided to plot function.

    :returns: handles

    """
    res = []
    assert len(dsList.dictByDim()) == 1 # We never integrate over dimensions...
    data = []
    for entry in dsList:
        for t in targets:
            # TODO: alternative: min(dsref[(entry.dim, entry.funcId)].detEvals((t,))[0])
            #       is the min from the alg with the best aRT
            flg_ert = 1
            if flg_ert:
                normalizer = dsref[(entry.dim, entry.funcId)].detERT((t,))[0]
            else:
                pass
            if np.isinf(normalizer):
                continue
            if istoolsstats:
                x = [np.inf] * perfprofsamplesize
                runlengthunsucc = []
                evals = entry.detEvals([t])[0]
                runlengthsucc = evals[np.isnan(evals) == False]
                runlengthunsucc = entry.maxevals[np.isnan(evals)]
                if len(runlengthsucc) > 0:
                    x = toolsstats.drawSP(runlengthsucc, runlengthunsucc,
                                         percentiles=[50],
                                         samplesize=perfprofsamplesize)[1]
                data.extend(i/normalizer for i in x)
            else:
                x = entry.detERT([t])[0]
                data.append(x/normalizer)
            #if (np.array(tmp) < 1e-1).any():
            #    set_trace()

    # Display data
    data = np.array(data)
    data = data[np.isnan(data)==False] # Take away the nans
    n = len(data)
    data = data[np.isinf(data)==False] # Take away the infs
    # data = data[data <= maxval] # Take away rightmost data
    #data = np.exp(craftingeffort) * data  # correction by crafting effort CrE
    #set_trace()
    if len(data) == 0: # data is empty.
        res = plotECDF(np.array((1., )), n=np.inf, **kwargs)
    else:
        plt.plot((min(data), ), (0, ), **kwargs)
        res = plotECDF(np.array(data), n=n, **kwargs)
        h = plt.plot((max(data), ), (float(len(data))/n, ), **kwargs)
        plt.setp(h, 'marker', 'x')
        #plotdata(np.array(data), x_limit, maxevals,
        #                    CrE=0., kwargs=kwargs)
#     if maxevals: # Should cover the case where maxevals is None or empty
#         x3 = np.median(maxevals)
#         if np.any(data > x3):
#             y3 = float(np.sum(data <= x3)) / n
#             h = plt.plot((x3,), (y3,), marker='x', markersize=30,
#                          markeredgecolor=plt.getp(res[0], 'color'),
#                          ls='', color=plt.getp(res[0], 'color'))
#             h.extend(res)
#             res = h # so the last element in res still has the label.
    return res

def main(dictAlg, dsref=None, order=None, targets=defaulttargets, outputdir='',
         info='default', verbose=True):
    """Generates image files of the performance profiles of algorithms

    From a dictionary of :py:class:`DataSetList` sorted by algorithms,
    generates the performance profile (Moré:2008) on multiple functions
    for multiple targets altogether.

    :param dict dictAlg: dictionary of :py:class:`DataSetList` instances, one
                         dataSetList

    :param list targets: target function values
    :param list order: sorted list of keys to dictAlg for plotting order

    """
    for d, dictalgdim in dictAlg.dictAlgByDim().items():
        plotmultiple(dictalgdim, dsref, targets)
        figureName = os.path.join(outputdir, 'ppperfprof_%02dD_%s' % (d, info))
        save_figure(figureName)
        plt.close()
