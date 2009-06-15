#! /usr/bin/env python

from __future__ import absolute_import

import os
import warnings
import numpy
#import matplotlib
#matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
from bbob_pproc import bootstrap
from pdb import set_trace

"""Generates Performance Profiles (More Wild 2002)."""
percentiles = 50
samplesize = 200

def beautify(figureName='perfprofile', funcsolved=None, maxval=None,
             isLegend=True, fileFormat=('png',)):

    plt.xscale('log')
    plt.xlabel('Running Lengths/Dim')
    plt.ylabel('Proportion of functions')
    plt.grid(True)

    if not funcsolved is None and funcsolved:
        try:
            txt = '(%d' % funcsolved[0]
            for i in range(1, len(funcsolved)):
                txt += ', %d' % funcsolved[i]
            txt += ') = %d funcs' % numpy.sum(funcsolved)
        except TypeError:
            txt = '%d funcs' % funcsolved

        plt.text(0.01, 1.01, txt, horizontalalignment='left',
                 verticalalignment="bottom",
                 transform=plt.gca().transAxes)

    if isLegend:
        plt.legend(loc='best')

    plt.xlim(xmin=1e-2)
    plt.xlim(xmax=1e3)
    #plt.ylim(0, 1)
    #set_trace()
    for entry in fileFormat:
        plt.savefig(figureName + 'a.' + entry, dpi = 300, format = entry)

    plt.xlim(xmin=1e2)
    if not maxval is None:
        plt.xlim(xmax=maxval)
    #plt.ylim(0, 1)
    for entry in fileFormat:
        plt.savefig(figureName + 'b.' + entry, dpi = 300, format = entry)
    
def plotPerfProf(data, maxval=None, maxevals=None, isbeautify=True, order=None,
                 kwargs={}):
    #Expect data to be a ndarray.
    x = data[numpy.isnan(data)==False] # Take away the nans
    #set_trace()
    nn = len(x)

    x = x[numpy.isinf(x)==False] # Take away the infs
    n = len(x)

    if n == 0:
        res = plt.plot([], [], **kwargs) #Why?
    else:
        x.sort()
        if maxval is None:
            maxval = max(x)
        x = x[x <= maxval]
        n = len(x) # redefine n to correspond to the x that will be shown...

        x2 = numpy.hstack([numpy.repeat(x, 2), maxval])
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, n+1) / float(nn), 2)])
        res = plt.plot(x2, y2, **kwargs)
        if not maxevals is None:
            #set_trace()
            x3 = numpy.median(maxevals) # or mean?
            try:
                y3 = y2[x2<=x3][-1]
            except IndexError: # x3 < x2 !!! TODO: possible?
                #set_trace()
                y3 = 0.

            #set_trace()
            plt.plot((x3,), (y3,), marker='x', ls=plt.getp(res[0], 'ls'),
                     color=plt.getp(res[0], 'color'))
            # Only take sequences for x and y!

        #set_trace()
        #if not order is None:
            ##set_trace()
            #plt.plot((maxval, maxval*2), (y2[-1], 0.05 + order[0]*0.9/(order[1]-1)),
                     #ls=plt.getp(res[0], 'ls'), color=plt.getp(res[0], 'color'))
            #plt.text(maxval*2.1, 0.05 + order[0]*0.9/(order[1]-1),
                     #kwargs['label'], horizontalalignment="left",
                     #verticalalignment="center")

    return res

def plotLegend(handles, maxval):
    ys = {}
    lh = 0
    for h in handles:
        h = h[0]
        x2 = plt.getp(h, "xdata")
        y2 = plt.getp(h, "ydata")
        try:
            ys.setdefault(y2[sum(x2 <= maxval) - 1], []).append(h)
            lh += 1
        except IndexError:
            pass

    i = 0
    for j in sorted(ys.keys()):
        for h in ys[j]:
            y = 0.02 + i * 0.96/(lh-1)
            plt.plot((maxval, maxval*10), (j, y),
                     ls=plt.getp(h, 'ls'), color=plt.getp(h, 'color'))
            plt.text(maxval*11, y,
                     plt.getp(h, 'label'), horizontalalignment="left",
                     verticalalignment="center")
            i += 1

def main(dsList, target, minERT=None, order=None,
         plotArgs={}, outputdir='', info='default', verbose=True):
    """From a list of IndexEntry, generates the performance profiles for
    multiple functions and a single target."""

    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    if len(dsList.dictByDim()) > 1:
        warnings.warn('Provided with data from multiple dimension.')

    dictData = {} # list of (ert per function) per algorithm
    dictMaxEvals = {} # list of (maxevals per function) per algorithm
    funcsolved = len(target)
    # the functions will not necessarily sorted.

    bestERT = [] # best ert per function, not necessarily sorted as well.

    # per instance instead of per function?
    dictFunc = dsList.dictByFunc()

    for f, samefuncEntries in dictFunc.iteritems():
        dictAlg = samefuncEntries.dictByAlg()
        erts = []
        try:
            target[f]
        except KeyError:
            continue
        #funcsolved.setdefault(alg, []).append(f)

        for alg, entry in dictAlg.iteritems():
            # entry is supposed to be a single item DataSetList
            entry = entry[0]

            x = [numpy.inf]*samplesize
            y = numpy.inf
            runlengthunsucc = []
            for j, line in enumerate(entry.evals):
                if line[0] <= target[f]:
                    runlengthsucc = line[1:][numpy.isfinite(line[1:])]
                    runlengthunsucc = entry.maxevals[numpy.isnan(line[1:])]
                    tmp = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                           percentiles=percentiles,
                                           samplesize=samplesize)
                    #set_trace()
                    x = list(float(i)/entry.dim for i in tmp[1])
                    y = entry.ert[j]
                    # j is the index of the line in entry.evals.
                    break

            dictData.setdefault(alg, []).extend(x)
            dictMaxEvals.setdefault(alg, []).extend(float(i)/entry.dim for i in runlengthunsucc)
            erts.append(y)

        if minERT is None:
            bestERT.append(min(erts))
        else:
            bestERT.append(minERT)
        #set_trace()

    #set_trace()
    # what about infs?
    maxval = 0
    #set_trace()
    for data in dictData.values():
        tmp = numpy.array(data) #/numpy.array(bestERT)
        if any(numpy.isfinite(tmp)):
            maxval = max(maxval, max(tmp[numpy.isfinite(tmp)]))

    #set_trace()
    if order is None:
        order = dictData.keys()

    lines = []
    for i, alg in enumerate(order):
        for elem in alg:
            if dictData.has_key(elem):
                lines.append(plotPerfProf(numpy.array(dictData[elem]), #/numpy.array(bestERT),
                             1e7, dictMaxEvals[elem],
                             kwargs=plotArgs[elem]))
                break
        #else: problem!
        #set_trace()
    #plotPerfProf(numpy.array(bestERT), #/numpy.array(bestERT),
                 #maxval, dictMaxEvals[alg],
                 #{'label': 'bestERT', 'color': 'k', 'marker': '*'})

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    beautify(figureName, funcsolved, 1e7, False)

    plt.close()

    #plt.rcdefaults()

def main2(dsList, target, order=None,
          plotArgs={}, outputdir='', info='default', verbose=True):
    """From a dataSetList, generates the performance profiles for multiple
    functions for multiple targets altogether.
    """

    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    dictData = {} # list of (ert per function) per algorithm
    dictMaxEvals = {} # list of (maxevals per function) per algorithm
    # the functions will not necessarily sorted.

    bestERT = [] # best ert per function, not necessarily sorted as well.
    funcsolved = {}
    if order is None:
        order = dictData.keys()

    # per instance instead of per function?
    dictFunc = dsList.dictByFunc()

    for f, samefuncEntries in dictFunc.iteritems():
        dictDim = samefuncEntries.dictByDim()
        for d, samedimEntries in dictDim.iteritems():
            dictAlg = samedimEntries.dictByAlg()

            #for alg, entry in dictAlg.iteritems():
                #entry = entry[0]
                #dictMaxEvals.setdefault(alg, []).extend((float(i)/entry.dim for i in entry.maxevals))

            for t in sorted(target.keys()):
                try:
                    if numpy.isnan(target[t][(f, d)]):
                        continue
                except KeyError:
                    continue
                funcsolved.setdefault(t, []).append(f)

                erts = []

                for alg, entry in dictAlg.iteritems():
                    # entry is supposed to be a single item DataSetList
                    entry = entry[0]

                    x = [numpy.inf]*samplesize
                    y = numpy.inf
                    runlengthunsucc = []
                    for j, line in enumerate(entry.evals):
                        if line[0] <= target[t][(f, d)]:
                            runlengthsucc = line[1:][numpy.isfinite(line[1:])]
                            runlengthunsucc = entry.maxevals[numpy.isnan(line[1:])]
                            tmp = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                                   percentiles=percentiles,
                                                   samplesize=samplesize)
                            #set_trace()
                            x = list(float(i)/entry.dim for i in tmp[1])
                            y = entry.ert[j]
                            #set_trace()
                            # j is the index of the line in entry.evals.
                            break

                    dictData.setdefault(alg, []).extend(x)
                    #set_trace()
                    dictMaxEvals.setdefault(alg, []).extend(float(i)/entry.dim for i in runlengthunsucc)
                    #TODO: there may be addition for every target... is it the desired behaviour?
                    erts.append(y)

                bestERT.append(min(erts))
        #set_trace()

    #set_trace()
    # what about infs?
    maxval = 0
    #set_trace()
    for data in dictData.values():
        tmp = numpy.array(data) #/numpy.array(bestERT)
        if any(numpy.isfinite(tmp)):
            maxval = max(maxval, max(tmp[numpy.isfinite(tmp)]))

    #set_trace()

    lines = []
    for i, alg in enumerate(order):
        for elem in alg:
            if dictData.has_key(elem):
                lines.append(plotPerfProf(numpy.array(dictData[elem]), #/numpy.array(bestERT),
                             1e7, dictMaxEvals[elem],
                             order=(i, len(order)), kwargs=plotArgs[elem]))
                break
    #set_trace()

    plotLegend(lines, 1e7)
    #plotPerfProf(numpy.array(bestERT), #/numpy.array(bestERT),
                 #maxval, dictMaxEvals[alg], {'label': 'bestERT', 'color' :'k', 'marker': '*'})

        #else: problem!
        #set_trace()

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    #set_trace()
    funcsolved = list(len(funcsolved[i]) for i in sorted(funcsolved.keys()))
    beautify(figureName, funcsolved, 1e7*1000, False)

    plt.close()

    #plt.rcdefaults()
