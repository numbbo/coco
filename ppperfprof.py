#! /usr/bin/env python

from __future__ import absolute_import

import os
import warnings
import numpy
import matplotlib.pyplot as plt
from bbob_pproc import bootstrap
from pdb import set_trace

"""Generates Runtime distributions (Performance Profiles, More Wild 2002)."""
percentiles = 50
samplesize = 200

function_IDs = range(1,999)  # 1, 6, 10, 15, 20, 101, 107, 122, TODO: functions should become input argument
    
def beautify(figureName='perfprofile', funcsolved=None, maxval=None,
             isLegend=True, fileFormat=('eps', 'png')):

    plt.xscale('log')
    plt.xlabel('Running lengths / dimension')
    plt.ylabel('Proportion of functions')
    plt.grid(True)

    if not funcsolved is None and funcsolved:
        txt = ''
        try:
            if len(funcsolved) > 1:
                txt = '(%d' % funcsolved[0]
                for i in range(1, len(funcsolved)):
                    if funcsolved[i] > 0:
                        txt += ', %d' % funcsolved[i]
                txt += ') = '
            txt += '%d funcs' % numpy.sum(funcsolved)
        except TypeError:
            txt = '%d funcs' % funcsolved

        plt.figtext(0.01, 1.01, txt, horizontalalignment='left',
                 verticalalignment="bottom")

    if isLegend:
        plt.legend(loc='best')

    plt.ylim(0, 1)

    plt.xlim(xmax=1e3) #TODO: save default value?
    plt.xlim(xmin=1e-2)
    if 1 < 3:  # first half only, takes about 2.5 seconds
        #set_trace()
        for entry in fileFormat:
            plt.savefig(figureName + 'a.' + entry, dpi = 300, format = entry)

    plt.xlim(xmin=1e2)
    if not maxval is None: # TODO: reset to default value...
        plt.xlim(xmax=maxval)
    if 1 < 3:  # second half only
        #plt.ylim(0, 1)
        for entry in fileFormat:
            plt.savefig(figureName + 'b.' + entry, dpi = 300, format = entry)

    plt.xlim(xmin=1e-0)
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 300, format = entry)


def plotPerfProf(data, maxval=None, maxevals=None, isbeautify=True, order=None,
                 kwargs={}):
    #Expect data to be a ndarray.
    x = data[numpy.isnan(data)==False] # Take away the nans
    #set_trace()
    nn = len(x)

    x = x[numpy.isinf(x)==False] # Take away the infs
    n = len(x)

    if n == 0:
        #TODO: problem if no maxval
        if maxval is None:
            res = plt.plot([], [], **kwargs)
        else:
            res = plt.plot([1., maxval], [0., 0.], **kwargs)
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
            x3 = numpy.median(maxevals)
            if numpy.any(x2 <= x3):
                y3 = y2[x2<=x3][-1]
                if y3 <= maxval:
                    plt.plot((x3,), (y3,), marker='x',
                             ls=plt.getp(res[0], 'ls'),
                             color=plt.getp(res[0], 'color'))
                # Only take sequences for x and y!

    return res

def plotLegend(handles, maxval):
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

    if lh <= 1:
        lh = 2
    i = 0 # loop over the elements of ys
    for j in sorted(ys.keys()):
        for k in reversed(sorted(ys[j].keys())):
            for h in ys[j][k]:
                y = 0.02 + i * 0.96/(lh-1)
                plt.plot((maxval, maxval*10), (j, y),
                         ls=plt.getp(h, 'ls'), color=plt.getp(h, 'color'))
                plt.text(maxval*11, y,
                         plt.getp(h, 'label'), horizontalalignment="left",
                         verticalalignment="center")
                i += 1

    #plt.axvline(x=maxval, color='k') # Not as efficient?
    plt.plot((maxval, maxval), (0., 1.), color='k')

def main2(dsList, target, order=None, plotArgs={}, outputdir='',
          info='default', fileFormat=('eps', 'png'), verbose=True):
    """From a dataSetList, generates the performance profiles for multiple
    functions for multiple targets altogether.
    keyword arguments:
    target: list of dictionaries with (function, dimension) as keys, target
    function values as values
    order: determines the plotting order of the algorithm (used by the legend
    and in the case the algorithm has no plotting arguments specified).

    """

    xlim = 1e7
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

                    x = [numpy.inf]*samplesize
                    y = numpy.inf
                    runlengthunsucc = []
                    for line in entry.evals:
                        if line[0] <= t[(f, d)]:
                            tmp = line[1:]/entry.dim
                            runlengthsucc = tmp[numpy.isfinite(tmp)]
                            runlengthunsucc = entry.maxevals[numpy.isnan(tmp)]
                            tmp = bootstrap.drawSP(runlengthsucc, runlengthunsucc,
                                                   percentiles=percentiles,
                                                   samplesize=samplesize)
                            x = tmp[1]
                            break

                    dictData.setdefault(alg, []).extend(x)
                    dictMaxEvals.setdefault(alg, []).extend(runlengthunsucc)
                    #TODO: there may be addition for every target... is it the desired behaviour?

    if order is None:
        order = dictData.keys()

    lines = []
    for i, alg in enumerate(order):
        for elem in alg:
            if dictData.has_key(elem):
                lines.append(plotPerfProf(numpy.array(dictData[elem]),
                             xlim, dictMaxEvals[elem], order=(i, len(order)),
                             kwargs=plotArgs[elem]))
                break

    plotLegend(lines, xlim)

    figureName = os.path.join(outputdir,'ppperfprof_%s' %(info))
    #set_trace()
    beautify(figureName, funcsolved, xlim*1000, False, fileFormat=fileFormat)

    plt.close()
