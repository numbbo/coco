#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""For generating performance profiles.

**Example**

.. plot::
   :width: 75%
   
   import urllib
   import tarfile
   import glob
   from pylab import *
   import pickle
   import bbob_pproc as bb
   import bbob_pproc.ppperfprof
   import bbob_pproc.bestalg

   # Collect and unarchive data
   dsets = {}
   for alg in bb.ppperfprof.best:  # ('BIPOP-CMA-ES', 'NEWUOA'):
        for date in ('2010', '2009'):
            try: 
                dataurl = 'http://coco.lri.fr/BBOB'+date+'/pythondata/' + alg + '.tar.gz'
                filename, headers = urllib.urlretrieve(dataurl)
                archivefile = tarfile.open(filename)
                archivefile.extractall()  # write to disc
                dsets[alg] = bb.load(glob.glob('BBOB'+date+'pythondata/' + alg + '/ppdata_f0*_20.pickle'))
            except:
                pass
   
   # plot the profiles
   figure()
   bb.bestalg.loadBBOB2009() # loads virtual best algorithm from BBOB 2009
   # bb.ppperfprof.plotmultiple(dsets.dictAlgByFuncGroup()['lcond'], dsref=bb.bestalg.bestalgentries2009)
   bb.ppperfprof.plotmultiple(dsets)

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
from bbob_pproc.compall.pprldmany import plotLegend

__all__ = ['beautify', 'main', 'plot']

figformat = ('eps', 'pdf') # Controls the output when using the main method

best = ('AMALGAM', 'iAMALGAM', 'VNS', 'MA-LS-CHAIN', 'BIPOP-CMA-ES', 'IPOP-ACTCMA-ES', 'MOS', 'IPOP-SEP-CMA-ES',
   'BFGS', 'NELDER', 'NELDERDOERR', 'NEWUOA', 'FULLNEWUOA', 'GLOBAL', 'MCS',
   'DIRECT', 'DASA', 'POEMS', 'Cauchy-EDA', 'RANDOMSEARCH')

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

    plt.xlabel('log10 of (ERT / ERTref)')
    plt.ylabel('Proportion of functions')
    logxticks()
    beautifyECDF()

def plotmultiple(dictAlg, dsref=None, targets=defaulttargets,
                 isbootstrap=False, rhleg=True):
    """Generate performance profile figure.
    
    :param dict dictAlg: dictionary of :py:class:`DataSetList` instances
                         one instance = one algorithm
    :param DataSetList dsref: reference data set
    :param seq targets: target function values
    :param bool isbootstrap: if True, uses bootstrapped distribution
    :param bool rhleg: if True, displays the right-hand legend
    
    """

    if not dsref:
        dsref = bestalg.generate(dictAlg)
    lines = []
    for i, k in enumerate(dictAlg):
        args = styles[(i) % len(styles)]
        args['linewidth'] = 1.5
        args['markersize'] = 15.
        args['markeredgewidth'] = 1.5
        args['markerfacecolor'] = 'None'
        args['markeredgecolor'] = args['color']
        lines.append(plot(dictAlg[k], dsref, targets, isbootstrap, label=k,
                          **args))
    #plt.xlim(xmin=1e-0, xmax=xlim*x_annote_factor)
    beautify()
    if rhleg:
        plotLegend(lines, plt.xlim()[1])

def plot(dsList, dsref, targets=defaulttargets, isbootstrap=False, **kwargs):
    """Generates a graph showing the performance profile of an algorithm.

    We display the empirical cumulative distribution function ECDF of
    the bootstrapped distribution of the expected running time (ERT)
    for an algorithm to reach the function value :py:data:`targets`
    normalized by the ERT of the reference algorithm for these
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
            #       is the min from the alg with the best ERT 
            flg_ert = 1
            if flg_ert:
                normalizer = dsref[(entry.dim, entry.funcId)].detERT((t,))[0]
            else:
                pass
            if np.isinf(normalizer):
                continue
            if isbootstrap:
                x = [np.inf] * perfprofsamplesize
                runlengthunsucc = []
                evals = entry.detEvals([t])[0]
                runlengthsucc = evals[np.isnan(evals) == False]
                runlengthunsucc = entry.maxevals[np.isnan(evals)]
                if len(runlengthsucc) > 0:
                    x = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
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
        res.append(plt.axhline(0., **kwargs))
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

    for d, dictalgdim in dictAlg.dictAlgByDim().iteritems():
        plotmultiple(dictalgdim, dsref, targets)
        figureName = os.path.join(outputdir, 'ppperfprof_%02dD_%s' % (d, info))
        saveFigure(figureName, figFormat=figformat, verbose=verbose)
        plt.close()
