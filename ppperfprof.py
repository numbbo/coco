#! /usr/bin/env python

from __future__ import absolute_import

import os
import warnings
import numpy
#import matplotlib
#matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
from bbob_pproc import bootstrap
from bbob_pproc.dataoutput import algLongInfos, algShortInfos
from pdb import set_trace

"""Generates Performance Profiles (More Wild 2002)."""
percentiles = 50
samplesize = 200

def beautify(figureName='perfprofile', maxval=None, fileFormat=('png',)):
    plt.xscale('log')
    plt.legend(loc='best')
    if not maxval is None:
        plt.xlim(plt.xlim()[0], maxval**1.05)
    plt.xlabel('Median Bootstrap ERT/Best')
    plt.ylabel('Proportion of functions')
    plt.grid(True)
    #set_trace()
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 300, format = entry)

def plotPerfProf(data, maxval=None, kwargs={}):
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
        x2 = numpy.hstack([numpy.repeat(x, 2), maxval**1.05])
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, n+1) / float(nn), 2)])
        res = plt.plot(x2, y2, **kwargs)
    return res

def main(dsList, target, order=None,
         plotArgs={}, outputdir='', info='default', verbose=True):
    """From a list of IndexEntry, generates the performance profiles."""

    #plt.rc("axes", labelsize=20, titlesize=24)
    #plt.rc("xtick", labelsize=20)
    #plt.rc("ytick", labelsize=20)
    #plt.rc("font", size=20)
    #plt.rc("legend", fontsize=20)

    if len(dsList.dictByDim()) > 1:
        warnings.warn('Provided with data from multiple dimension.')

    dictData = {} # list of (ert per function) per algorithm
    # the functions will not necessarily sorted.

    bestERT = [] # best ert per function, not necessarily sorted as well.

    # per instance instead of per function?
    dictFunc = dsList.dictByFunc()

    for f, samefuncEntries in dictFunc.iteritems():
        dictAlg = samefuncEntries.dictByAlg()
        erts = []

        for alg, entry in dictAlg.iteritems():
            # entry is supposed to be a single item DataSetList
            entry = entry[0]
            # Duck typing would not work here: the function ids are integers.
            #if isinstance(target, dict):
                #x = entry.ert[entry.target <= target[f]]
            #else:
                #x = entry.ert[entry.target <= target]
            x = numpy.inf # default value used if no success
            for j in entry.evals:
                if j[0] <= target[f]:
                    runlengthsucc = j[1:][numpy.isfinite(j[1:])]
                    runlengthunsucc = entry.maxevals[numpy.isnan(j[1:])]
                    tmp = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                           percentiles=percentiles,
                                           samplesize=samplesize)
                    #set_trace()
                    x = tmp[0][0]
                    break

            dictData.setdefault(alg, []).append(x)
            erts.append(x)

        bestERT.append(min(erts))

    # what about infs?
    maxval = 0
    for data in dictData.values():
        tmp = numpy.array(data)/numpy.array(bestERT)
        maxval = max(maxval, max(tmp[numpy.isfinite(tmp)]))

    #set_trace()
    if order is None:
        order = dictData.keys()

    for alg in order:
        plotPerfProf(numpy.array(dictData[alg])/numpy.array(bestERT),
                     maxval,plotArgs[alg])
        #set_trace()

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    beautify(figureName, maxval)

    plt.close()

    #plt.rcdefaults()
