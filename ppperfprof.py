#! /usr/bin/env python

from __future__ import absolute_import

import os
import warnings
import numpy
#import matplotlib
#matplotlib.use('GtkAgg')
import matplotlib.pyplot as plt
from bbob_pproc.dataoutput import algLongInfos, algShortInfos
from pdb import set_trace

"""Generates Performance Profiles (More Wild 2002)."""

def beautify(figureName='perfprofile', fileFormat=('png',)):
    plt.legend(loc='best')
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 300, format = entry)

def plotPerfProf(data, label=None):
    x = data[numpy.isnan(data)==False] # Take away the nans
    x.sort()
    #set_trace()
    nn = len(x)
    x = x[numpy.isinf(x)==False] # Take away the infs
    n = len(x)
    if n == 0:
        res = plt.plot([], [], label=label) #Why?
    else:
        x.sort()
        x2 = numpy.hstack([numpy.repeat(x, 2)])
        y2 = numpy.hstack([0.0,
                          numpy.repeat(numpy.arange(1, n) / float(nn), 2), 1.0])
        res = plt.plot(x2, y2, label=label)

def main(dsList, target, 
         order=None, outputdir='', info='default', verbose=True):
    """From a list of IndexEntry, generates the performance profiles."""

    plt.rc("axes", labelsize=20, titlesize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("font", size=20)
    plt.rc("legend", fontsize=20)

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
            if isinstance(target, dict):
                x = entry.ert[entry.target <= target[f]]
            else:
                x = entry.ert[entry.target <= target]

            if len(x) > 0:
                x = x[0]
            else: # no success
                x = numpy.inf

            dictData.setdefault(alg, []).append(x)
            erts.append(x)

        bestERT.append(min(erts))
    # TODO: bootstrap something... but what?

    # what about infs?
    if order is not None:
        for alg in order:
            #set_trace()
            plotPerfProf(numpy.array(dictData[algLongInfos[alg]])/numpy.array(bestERT),
                         label=alg)
            #set_trace()    
    else:
        for alg, data in dictData.iteritems():
            #set_trace()
            plotPerfProf(numpy.array(data)/numpy.array(bestERT),
                         label=algShortInfos[alg])
            #set_trace()

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    beautify(figureName)

    plt.close()

    plt.rcdefaults()
