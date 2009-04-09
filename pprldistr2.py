#! /usr/bin/env python

"""Creates run length distribution figures."""


from __future__ import absolute_import

import os
import numpy
import matplotlib.pyplot as plt
from bbob_pproc import bootstrap
from pdb import set_trace

#__all__ = []

rldColors = ('k', 'c', 'm', 'r', 'k', 'c', 'm', 'r', 'k', 'c', 'm', 'r')
rldUnsuccColors = ('k', 'c', 'm', 'k', 'c', 'm', 'k', 'c', 'm', 'k', 'c', 'm')  # should not be too short
# Used as a global to store the largest xmax and align the FV ECD figures.
fmax = None
evalfmax = None

def beautify(figHandle, figureName, fileFormat=('png', 'eps'),
             text=None, verbose=True):
    """Format the figure of the run length distribution and save into files."""
    axisHandle = figHandle.gca()
    #try:
    axisHandle.set_xscale('log')
    #plt.xlim(1.0, maxEvalsF)
    plt.ylim(0.0, 1.0)
    axisHandle.set_xlabel('log10 of ERT1/ERT0')
    axisHandle.set_ylabel('proportion of instances...')
    # Grid options
    axisHandle.grid('True')
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in xtic:
        newxtic.append('%d' % round(numpy.log10(j)))
    axisHandle.set_xticklabels(newxtic)

    #plt.text(0.5, 0.93, text, horizontalalignment="center",
             #transform=axisHandle.transAxes)
             #bbox=dict(ec='k', fill=False), 

    #set_trace()
    #plt.legend(loc='best')
    #if legend:
        #axisHandle.legend(legend, locLegend)

    # Save figure
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 300,
                    format = entry)
        if verbose:
            print 'Wrote figure in %s.' %(figureName + '.' + entry)



def plotLogAbs(indexEntries0, indexEntries1, fvalueToReach, isByInstance=False,
               verbose=True):
    """Creates one run length distribution from a sequence of indexEntries.

    Keyword arguments:
    indexEntries0 -- reference
    indexEntries1
    fvalueToReach
    maxEvalsF
    isByInstance -- loop over the function instances instead of the functions
    verbose

    Outputs:
    res -- resulting plot.
    fsolved -- number of different functions solved.
    funcs -- number of different function considered.
    """

    x = []
    nn = 0
    fsolved = set()
    funcs = set()
    funIndexEntries0 = indexEntries0.dictByFunc()
    funIndexEntries1 = indexEntries1.dictByFunc()
    for func in set(funIndexEntries0.keys()).union(funIndexEntries1.keys()):
        try:
            i0 = funIndexEntries0[func][0]
            i1 = funIndexEntries1[func][0]
        except KeyError:
            continue

        if isByInstance:
            dictinstance = []

        ERT = []
        if not isByInstance:
            for i, entry in enumerate((i0, i1)):
                for j in entry.hData:
                    if j[0] <= fvalueToReach:
                        break
                success = (j[entry.nbRuns()+1:] <= fvalueToReach)
                fevals = j[1:entry.nbRuns()+1]
                for j, issuccess in enumerate(success):
                    if not issuccess:
                        fevals[j] = entry.vData[-1, 1+j]
                tmp = bootstrap.sp(fevals, issuccessful=success)
                #set_trace()
                if tmp[2]: #success probability is larger than 0
                    ERT.append(tmp[0])
                else:
                    set_trace()
                    ERT.append(numpy.inf)
            if not all(numpy.isinf(ERT)):
                x.append(ERT[1]/ERT[0])
                nn += 1
            #TODO check it is the same as ERT[1]
        else:
            for i, entry in enumerate((i0, i1)):
                dictinstance = {}
                for j in range(len(entry.itrials)):
                    dictinstance.setdefault(entry.itrials[j], []).append(j)
                ERT.append(dictinstance.copy())
                for k in dictinstance:
                    for j in entry.hData:
                        if j[0] <= fvalueToReach:
                            break
                    success = (j[list(entry.nbRuns()+i for i in dictinstance[k])] <= fvalueToReach)
                    fevals = j[list(1+i for i in dictinstance[k])]
                    for l, issuccess in enumerate(success):
                        if issuccess:
                            fevals[l] = entry.vData[-1, 1+dictinstance[k][l]]
                    tmp = bootstrap.sp(fevals, issuccessful=success)
                    if tmp[2]: #success probability is larger than 0
                        ERT[i][k] = tmp[0]
                    else:
                        ERT[i][k] = numpy.inf
                    ERT[i][k] = tmp[0]
            s0 = set(ERT[0])
            s1 = set(ERT[1])
            #Could be done simpler
            for j in s0 - s1:
                x.append(0)
            for j in s0 & s1:
                x.append(ERT[1][j]/ERT[0][j])
            for j in s1 - s0:
                x.append(inf)
            nn += len(s0 | s1)

    #set_trace()
    #label = ('%+d:%d/%d' %
             #(numpy.log10(fvalueToReach), len(fsolved), len(funcs)))
    n = len(x)
    if n == 0:
        res = plt.plot([], [], label=label)
    else:
        x.sort()
        x2 = numpy.hstack([numpy.repeat(x, 2)])
        #maxEvalsF: used for the limit of the plot.
        y2 = numpy.hstack([0.0,
                           numpy.repeat(numpy.arange(1, n) / float(nn), 2),
                           n/float(nn)])
        res = plt.plot(x2, y2) #, label=label)

    return res#, fsolved, funcs

def main(indexEntriesAlg0, indexEntriesAlg1, valuesOfInterest,
         isStoringXMax=False, outputdir='', info='default', verbose=True):
    """Generate figures of empirical cumulative distribution functions.

    Keyword arguments:
    indexEntries -- list of IndexEntry instances to process.
    valuesOfInterest -- target function values to be displayed.
    isStoringXMax -- if set to True, the first call BeautifyVD sets the globals
                     fmax and maxEvals and all subsequent calls will use these
                     values as rightmost xlim in the generated figures.
     -- if set to True, the first call BeautifyVD sets the global
                     fmax and all subsequent call will have the same maximum
                     xlim.
    outputdir -- output directory (must exist)
    info --- string suffix for output file names.

    Outputs:
    Image files of the empirical cumulative distribution functions.
    """

    #sortedIndexEntries = sortIndexEntries(indexEntries)

    plt.rc("axes", labelsize=20, titlesize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("font", size=20)
    plt.rc("legend", fontsize=20)

    #maxEvalsFactor = max(i.mMaxEvals()/i.dim for i in indexEntriesAlg0+indexEntriesAlg1) ** 1.05

    #set_trace()

    #dictFunc0 = indexEntriesAlg0.dictByFunc()
    #dictFunc1 = indexEntriesAlg1.dictByFunc()
    #funcs = set.union(set(dictFunc0), set(dictFunc0))

    #maxEvalsFactorCeil = numpy.power(10,
                                     #numpy.ceil(numpy.log10(maxEvalsFactor)))

    #if isStoringXMax:
        #global evalfmax
    #else:
        #evalfmax = None

    #if not evalfmax:
        #evalfmax = maxEvalsFactor

    figureName = os.path.join(outputdir,'pplog%s' %('_' + info))
    fig = plt.figure()
    legend = []
    for j in range(len(valuesOfInterest)):
        tmp = plotLogAbs(indexEntriesAlg0, indexEntriesAlg1,
                         valuesOfInterest[j], verbose=verbose)
        #set_trace()
        if not tmp is None:
            plt.setp(tmp, 'color', rldColors[j])
            #set_trace()
            #legend.append('%+d:%d/%d' %  
                          #(numpy.log10(valuesOfInterest[j]), len(fsolved), 
                           #len(f)))
            if rldColors[j] == 'r':  # 1e-8 in bold
                plt.setp(tmp, 'linewidth', 3)

    #funcs = list(i.funcId for i in indexEntries)
    #if len(funcs) > 1:
        #text = 'f%d-%d' %(min(funcs), max(funcs))
    #else:
        #text = 'f%d' %(funcs[0])
    beautify(fig, figureName, verbose=verbose)
    plt.close(fig)

    #figureName = os.path.join(outputdir,'ppfvdistr_%s' %(info))
    #fig = plt.figure()
    #for j in range(len(valuesOfInterest)):
        ##set_trace()
        #tmp = plotFVDistr(indexEntries, valuesOfInterest[j],
                          #maxEvalsFactor, verbose=verbose)
        ##if not tmp is None:
        #plt.setp(tmp, 'color', rldColors[j])
        #if rldColors [j] == 'r':  # 1e-8 in bold
            #plt.setp(tmp, 'linewidth', 3)

    #tmp = numpy.floor(numpy.log10(maxEvalsFactor))
    ## coloring left to right:
    ##maxEvalsF = numpy.power(10, numpy.arange(tmp, 0, -1) - 1)
    ## coloring right to left:
    #maxEvalsF = numpy.power(10, numpy.arange(0, tmp))

    ##The last index of valuesOfInterest is still used in this loop.
    ##set_trace()
    #for k in range(len(maxEvalsF)):
        #tmp = plotFVDistr(indexEntries, valuesOfInterest[j],
                          #maxEvalsF=maxEvalsF[k], verbose=verbose)
        #plt.setp(tmp, 'color', rldUnsuccColors[k])

    #beautifyFVD(fig, figureName, text=text, isStoringXMax=isStoringXMax,
                #verbose=verbose)

    #plt.close(fig)

    plt.rcdefaults()
