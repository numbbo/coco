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
             fileFormat=('png',)):

    plt.xscale('log')
    plt.xlim(xmin=1e-2)
    if not maxval is None:
        plt.xlim(xmax=maxval)
    plt.ylim(0, 1)
    plt.xlabel('Running Lengths/Dim')
    plt.ylabel('Proportion of functions')
    plt.grid(True)

    if not funcsolved is None:
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

    plt.legend(loc='best')

    #set_trace()
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 300, format = entry)

def plotPerfProf(data, maxval=None, maxevals=None, isbeautify=True, kwargs={}):
    x = data[numpy.isnan(data)==False] # Take away the nans
    x.sort()
    #set_trace()
    nn = len(x)
    x = x[numpy.isinf(x)==False] # Take away the infs
    n = len(x)

    if n == 0:
        res = plt.plot([], [], **kwargs) #Why?
    else:
        x.sort()
        if maxval is None:
            maxval = x[-1]
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
            kwargs3 = kwargs.copy()
            kwargs3['marker'] = 'x'
            del kwargs3['label']
            plt.plot((x3,), (y3,), **kwargs3) # Only take sequences for x and y!

    return res

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
            dictMaxEvals.setdefault(alg, []).extend((float(i)/entry.dim for i in entry.maxevals))
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

    for alg in order:
        if dictData.has_key(alg):
            kwargs = plotArgs[alg].copy()
            plotPerfProf(numpy.array(dictData[alg]), #/numpy.array(bestERT),
                         max(maxval, 1e7), dictMaxEvals[alg], kwargs=kwargs)
        #else: problem!
        #set_trace()
    #plotPerfProf(numpy.array(bestERT), #/numpy.array(bestERT),
                 #maxval, dictMaxEvals[alg],
                 #{'label': 'bestERT', 'color': 'k', 'marker': '*'})

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    beautify(figureName, funcsolved, 1e7)

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

    # per instance instead of per function?
    dictFunc = dsList.dictByFunc()

    for f, samefuncEntries in dictFunc.iteritems():
        dictDim = samefuncEntries.dictByDim()
        for d, samedimEntries in dictDim.iteritems():
            dictAlg = samedimEntries.dictByAlg()

            for alg, entry in dictAlg.iteritems():
                entry = entry[0]
                dictMaxEvals.setdefault(alg, []).extend((float(i)/entry.dim for i in entry.maxevals))

            for t in target.keys():
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
    if order is None:
        order = dictData.keys()

    for alg in order:
        if dictData.has_key(alg):
            plotPerfProf(numpy.array(dictData[alg]), #/numpy.array(bestERT),
                         max(maxval, 1e7), dictMaxEvals[alg], plotArgs[alg])
    #plotPerfProf(numpy.array(bestERT), #/numpy.array(bestERT),
                 #maxval, dictMaxEvals[alg], {'label': 'bestERT', 'color' :'k', 'marker': '*'})

        #else: problem!
        #set_trace()

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    #set_trace()
    funcsolved = list(len(funcsolved[i]) for i in sorted(funcsolved.keys()))
    beautify(figureName, funcsolved, 1e7)

    plt.close()

    #plt.rcdefaults()
