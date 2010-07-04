#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates ERTs and convergence figures for multiple algorithms."""

import os
import sys
import matplotlib.pyplot as plt
import numpy
from pdb import set_trace
from bbob_pproc import bootstrap, bestalg, pproc
from bbob_pproc.ppfig import saveFigure
from bbob_pproc.dataoutput import algPlotInfos

styles = [{'color': 'k', 'marker': 'o', 'markeredgecolor': 'k'},
          {'color': 'b'},
          {'color': 'c', 'marker': 'v', 'markeredgecolor': 'c'},
          {'color': 'g'},
          {'color': 'y', 'marker': '^', 'markeredgecolor': 'y'},
          {'color': 'm'},
          {'color': 'r', 'marker': 's', 'markeredgecolor': 'r'}] # sort of rainbow style

#Get benchmark short infos.
funInfos = {}
figformat = ('eps', 'pdf') # Controls the output when using the main method
isBenchmarkinfosFound = True
infofile = os.path.join(os.path.split(__file__)[0], '..', 'benchmarkshortinfos.txt')
try:
    f = open(infofile,'r')
    for line in f:
        if len(line) == 0 or line.startswith('%') or line.isspace() :
            continue
        funcId, funcInfo = line[0:-1].split(None,1)
        funInfos[int(funcId)] = funcId + ' ' + funcInfo
    f.close()
except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    isBenchmarkinfosFound = False
    print 'Could not find file', infofile, \
          'Titles in figures will not be displayed.'

def beautify(title='', legend=True):
    """ Customize a figure by adding a legend, axis label, etc and save to a file.
        Is identical to beautify except for the linear and quadratic scaling
        lines which are quadratic and cubic

        Keyword arguments:
        title -- 
        legend --

    """

    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_xscale("log")
    axisHandle.set_yscale("log")

    # Grid options
    axisHandle.grid('True')

    ymin, ymax = plt.ylim()

    # quadratic and cubic "grid"
    plt.plot((2,200), (1, 1e2), 'k:', zorder=-1)
    plt.plot((2,200), (1, 1e4), 'k:', zorder=-1)
    plt.plot((2,200), (1e3, 1e5), 'k:', zorder=-1)  
    plt.plot((2,200), (1e3, 1e7), 'k:', zorder=-1)
    plt.plot((2,200), (1e6, 1e8), 'k:', zorder=-1)  
    plt.plot((2,200), (1e6, 1e10), 'k:', zorder=-1)

    # axes limites
    plt.xlim(1.8, 45)                # Should depend on xmin and xmax
    plt.ylim(ymin=10**-0.2, ymax=ymax) # Set back the default maximum.

    # ticks on axes
    #axisHandle.invert_xaxis()
    dimticklist = (2, 3, 4, 5, 10, 20, 40)  # TODO: should become input arg at some point? 
    dimannlist = (2, 3, '', 5, 10, 20, 40)  # TODO: should become input arg at some point? 
    # TODO: All these should depend on (xlim, ylim)

    axisHandle.set_xticks(dimticklist)
    axisHandle.set_xticklabels([str(n) for n in dimannlist])

    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    axisHandle.set_yticklabels(tmp2)

def generateData(dataSet, target):
    """Returns an array of results to be plotted.

    Oth column is ert, 1st is the success rate, 2nd the number of successes,
    3rd the mean of the number of function evaluations, and 4th the median
    of number of function evaluations of successful runs or numpy.nan.
    """

    res = []

    data = dataSet.detEvals([target])[0]
    succ = (numpy.isnan(data) == False)
    data[numpy.isnan(data)] = dataSet.maxevals[numpy.isnan(data)]
    res.extend(bootstrap.sp(data, issuccessful=succ, allowinf=False))
    res.append(numpy.mean(data))
    if res[2] > 0:
        res.append(bootstrap.prctile(data[succ], 50)[0])
    else:
        res.append(numpy.nan)

    return res

def main(dictAlg, sortedAlgs, target, outputdir, verbose=True):
    """From a DataSetList, returns figures showing the scaling: ERT/dim vs dim.
    
    One function and one target per figure.
    
    """

    dictFunc = pproc.dictAlgByFun(dictAlg)

    for f in dictFunc:
        filename = os.path.join(outputdir,'ppfigs_f%d' % (f))
        for alg in sortedAlgs:
            dictDim = dictFunc[f][alg].dictByDim()

            #Collect data
            dimert = []
            ert = []
            dimnbsucc = []
            ynbsucc = []
            nbsucc = []
            dimmaxevals = []
            maxevals = []
            dimmedian = []
            medianfes = []
            for dim in sorted(dictDim):
                entry = dictDim[dim][0]
                data = generateData(entry, target)
                if data[2] == 0: # No success
                    dimmaxevals.append(dim)
                    maxevals.append(float(data[3])/dim)
                else:
                    dimmedian.append(dim)
                    medianfes.append(data[4]/dim)
                    dimert.append(dim)
                    ert.append(float(data[0])/dim)
                    if data[1] < 1.:
                        dimnbsucc.append(dim)
                        ynbsucc.append(float(data[0])/dim)
                        nbsucc.append('%d' % data[2])

            # Draw lines
            tmp = plt.plot(dimert, ert, marker='.', markersize=30,
                          **algPlotInfos[(entry.algId, entry.comment)])[0]
            plt.setp(tmp, markeredgecolor=plt.getp(tmp, 'color'))
            tmp = plt.plot(dimmaxevals, maxevals, marker='x', markersize=20,
                           **algPlotInfos[(entry.algId, entry.comment)])[0]
            plt.setp(tmp, markeredgecolor=plt.getp(tmp, 'color'))
            #tmp2 = plt.plot(dimmedian, medianfes, ls='', marker='+',
            #               markersize=30, markeredgewidth=5,
            #               markeredgecolor=plt.getp(tmp, 'color'))[0]
            #for i, n in enumerate(nbsucc):
            #    plt.text(dimnbsucc[i], numpy.array(ynbsucc[i])*1.85, n,
            #             verticalalignment='bottom',
            #             horizontalalignment='center')

        if not bestalg.bestalgentries:
            bestalg.loadBBOB2009()

        bestalgdata = []
        dimsbestalg = list(df[0] for df in bestalg.bestalgentries if df[1] == f)
        dimsbestalg.sort()
        for d in dimsbestalg:
            entry = bestalg.bestalgentries[(d, f)]
            bestalgdata.append(float(entry.detERT([target])[0])/d)

        plt.plot(dimsbestalg, bestalgdata, color='wheat', linewidth=10,
                 marker='d', markersize=25, markeredgecolor='wheat',
                 zorder=-1)

        if isBenchmarkinfosFound:
            title = funInfos[f]
            plt.gca().set_title(title)

        beautify()

        saveFigure(filename, figFormat=figformat, verbose=verbose)

        plt.close()

