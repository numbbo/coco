#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Creates ENFEs and convergence figures for BBOB post-processing for the
comparison of 2 algorithms.
"""

import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy
from pdb import set_trace
from bbob_pproc import bootstrap, readalign


colors = ('k', 'b', 'c', 'g', 'y', 'm', 'r', 'k', 'k', 'c', 'r', 'm')
markers = ('o', 'v', 's', '*', '+', 'x', 'D')
linewidth = 3

#Get benchmark short infos.
funInfos = {}
isBenchmarkinfosFound = True
#infofile = os.path.join(os.path.split(__file__)[0], '..', '..',
                        #'benchmarkshortinfos.txt')
infofile = os.path.join(os.path.split(__file__)[0], '..',
                        'benchmarkshortinfos.txt')

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
          'Titles in scaling figures will not be displayed.'


def createFigure(data, label=None, figHandle=None):
    """ Create a plot in a figure (eventually new) from a data array.

        Mandatory input:
        data - 2d-array where the 1st column contains x-values
               and the remaining columns contain y-values

        Optional inputs:
        figHandle - handle to existing figure, if omitted a new figure
                    will be created

        Output:
        lines - handles of the Line2D instances.
    """

    # initialize figure and handles
    if figHandle is None:
        figHandle = plt.figure()
    else:
        plt.figure(figHandle.number)

    lines = []
    # Plot data sets
    for i in range(1,len(data[0,:])):
        yValues = data[:,i]

        # Plot figure
        tmp = numpy.where(numpy.isfinite(data[:,i]))[0]
        if tmp.size > 0:
            lines.extend(plt.plot(data[tmp,0], yValues[tmp], label=label,
                                  linewidth=linewidth))
            lines.extend(plt.plot([data[tmp[-1],0]], [yValues[tmp[-1]]],
                                  marker='+', markersize=5*linewidth,
                                  markeredgewidth=linewidth))
            #TODO: larger size.
        #else: ???

    return lines


def customLegend():
    """
    o In the end of each graph + the dimension is annotated, if no trial was
    successful, or both algorithms have more than 5 successful trials.
    Otherwise the #succ is annotated for both algorithms using >9 in case, e.g.
    2/>9.
    """
    
    #Set Text
    
    
    #Plot Text
    plt.text()
    
#TODO improve interface.
def customizeFigure(figHandle, figureName = None, xmin=None, title='',
                    fileFormat=('png','eps'), labels=None,
                    legend=True, locLegend='best', verbose=True):
    """ Customize a figure by adding a legend, axis label, etc. At the
        end the figure is saved.

        Inputs:
        figHandle - handle to existing figure

        Optional Inputs:
        figureName - name of the output figure
        xmin - boundary value of the x-axis
        fileFormat - list of formats of output files
        labels - list with xlabel and ylabel
        legend - if this list is not empty a legend will be generated from
                 the entries of this list
        locLegend - location of legend

    """

    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')
    axisHandle.set_yscale('log')

    # We are setting xmin
    if xmin:
        xminauto, xmaxauto = plt.xlim()
        #plt.xlim(max(xminauto, 1e-8), xmaxauto)
        plt.xlim(xmin, xmaxauto)

    axisHandle.invert_xaxis()

    # Annotate figure
    if not labels is None:
        axisHandle.set_xlabel(labels[0])
        axisHandle.set_ylabel(labels[1])

    # Grid options
    axisHandle.grid('True')
    tmp = axisHandle.get_xticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    axisHandle.set_xticklabels(tmp2)
    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    axisHandle.set_yticklabels(tmp2)

    # Legend
    if legend:
        plt.legend(loc=locLegend)
    axisHandle.set_title(title)

    # Save figure
    if not (figureName is None or fileFormat is None):
        if isinstance(fileFormat, basestring):
            plt.savefig(figureName + '.' + fileFormat, dpi = 120,
                        format=fileFormat)
            if verbose:
                print 'Wrote figure in %s.' %(figureName + '.' + fileFormat)
        else:
            #TODO: that is if fileFormat is iterable.
            for entry in fileFormat:
                plt.savefig(figureName + '.' + entry, dpi = 120,
                            format = entry)
                if verbose:
                    print 'Wrote figure in %s.' %(figureName + '.' + entry)

    # Close figure
    plt.close(figHandle)

    # TODO:    *much more options available (styles, colors, markers ...)
    #       *output directory - contained in the file name or extra parameter?


def sortIndexEntries(indexEntries):
    """From a list of IndexEntry, returns a sorted dictionary."""
    sortByFunc = {}
    dims = set()
    funcs = set()
    for elem in indexEntries:
        sortByFunc.setdefault(elem.funcId,{})
        sortByFunc[elem.funcId][elem.dim] = elem
        funcs.add(funcId)
        dims.add(elem.dim)

    return sortByFunc


#def computeERT(hdata):
    #res = []
    #nbRuns = (numpy.shape(hdata)[1]-1)/2
    #for i in hdata:
        #success = i[nbRuns+1:] <= i[0]
        #tmp = [i[0]]
        #tmp.extend(bootstrap.sp(i[1:nbRuns+1], issuccessful=success))
        #res.append(tmp)
    ##set_trace()
    #return numpy.vstack(res)

def computeERT(hdata, maxevals):
    res = []
    for i in hdata:
        data = i.copy()
        data = data[1:]
        succ = (numpy.isnan(data)==False)
        if any(numpy.isnan(data)):
            data[numpy.isnan(data)] = maxevals[numpy.isnan(data)]
        tmp = [i[0]]
        tmp.extend(bootstrap.sp(data, issuccessful=succ))
        res.append(tmp)
    #set_trace()
    return numpy.vstack(res)


def completion1(data, data0, data1, color, fig):
    """Completes the figure of the log(ERT1/ERT0) with -log(ERT1) or log(ERT0)
       from the last point depending on whether algorithm 0 or 1 stopped early.
    """
    if (data0[:, 2] == 0).any():
        #set_trace()
        tmp = data1[(data0[:, 2] == 0), 0:2]
        tmp = numpy.vstack([data1[(data0[:, 2] > 0) * (data1[:, 2] > 0)][-1, 0:2], tmp])
        #set_trace()
        tmp[:,1] = tmp[0,1] / tmp[:,1] * data[(data0[:, 2] > 0) * (data1[:, 2] > 0)][-1, 1]
        h = createFigure(tmp, figHandle=fig)
        plt.setp(h, 'color', color)
    if (data1[:, 2] == 0).any():
        #set_trace()
        tmp = data0[(data1[:, 2] == 0), 0:2]
        tmp = numpy.vstack([data0[(data0[:, 2] > 0) * (data1[:, 2] > 0)][-1, 0:2], tmp])
        #set_trace()
        tmp[:,1] = tmp[:,1] / tmp[0,1] * data[(data0[:, 2] > 0) * (data1[:, 2] > 0)][-1, 1]
        h = createFigure(tmp, figHandle=fig)
        plt.setp(h, 'color', color)

def completion2(data0, data1, fig):
    '''* If Alg0 does not reach the respective Df, plot the number of
    successful trials of Alg1 on a linear scale between -2==100% and 0=0%.
    Change the line style (to dashed) when ERT_Alg1 becomes larger than
    median(maxevals_Alg0). The line ends where both algorithms fail to reach
    the target. The same applies equivalently if Alg1 does not reach the
    respective Df first with a line above zero.
    o Two well-visible marks are shown (in case): (a) for the largest Df, where
    one algorithm fails to reach Df, (b) for the largest Df, where ERT>maxevals
    (at switch to dashed line style). Probably also each dimension should have
    different markers over the graph.
    o  + markers give the (two-sided) significance level between the
    algorithm, one *-marker for p<0.05 or k>=2 *-markers for p<1/10**k. The
    significance correction for multiple testing is applied within each figure
    according to Bonferroni.
    '''

    

def ranksumtest(N1, N2):
    """Custom rank-sum (Mann-Whitney-Wilcoxon) test
    http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U
    Small sample sizes (direct method).
    Keyword arguments:
    N1    sample 1
    N2    sample 2
    """

    # Possible optimization by setting sample 1 to be the one with the smallest
    # rank.

    #TODO: deal with more general type of sorting.
    s1 = sorted(N1)
    s2 = sorted(N2)
    U = 0.
    for i in s1:
        Ui = 0. # increment of U
        for j in s2:
            if j < i:
                Ui += 1.
            elif j == i:
                Ui += .5
            else:
                break
        #if Ui == 0.:
            #break
        U += Ui
    return U

def alignData(i0, i1):
    """Align the data in i0.evals and i1.evals, returns two arrays of aligned
    data.
    """

    res = readalign.alignArrayData(readalign.HArrayMultiReader([i0.evals,
                                                                i1.evals]))
    idx = 1 + i0.nbRuns()
    data0 = res[:, numpy.r_[0, 1:idx]]
    #data0 = computeERT(data0, i0.maxevals)
    data1 = res[:, numpy.r_[0, idx:idx+i1.nbRuns()]]
    #data1 = computeERT(data1, i1.maxevals)
    return data0, data1

def generateERT(i0, i1):
    """Align the data in i0.ert and i1.ert
    """

    data0 = numpy.vstack((i0.evals[:, 0], i0.ert)).transpose()
    data1 = numpy.vstack((i1.evals[:, 0], i1.ert)).transpose()
    res = readalign.alignArrayData(readalign.HArrayMultiReader([data0, data1]))

    data0 = res[:, numpy.r_[0, 1]]
    data1 = res[:, numpy.r_[0, 2]]
    return data0, data1

def generatePlot(dataset0, dataset1, i, dim):
    [data0, data1] = alignData(dataset0, dataset1)
    ert0 = computeERT(data0, dataset0.maxevals)
    ert1 = computeERT(data1, dataset1.maxevals)
    data = numpy.vstack((ert0[:, 0], (ert1[:, 1]/ert0[:, 1]))).transpose()

    isfinite = data[numpy.isfinite(ert0[:, 1]) * numpy.isfinite(ert1[:, 1]), :]
    idx = len(isfinite) # index for which either alg0 or alg1 stopped

    plt.plot(isfinite[:, 0], isfinite[:, 1], color=colors[i],
             linewidth=linewidth, ls='--')
    plt.plot((isfinite[-1, 0], ), (isfinite[-1, 1], ), marker=markers[i],
             markersize=6*linewidth, markeredgewidth=linewidth,
             markeredgecolor=colors[i], markerfacecolor=None, color='w')
    #completion1(data, data0, data1, colors[i], fig)

    #The following part is new.
    #set_trace()
    remainingert = None
    if idx == len(data):
        #Both went as far
        soliddata = isfinite

    else:
        #set_trace()
        if numpy.isinf(ert0[idx, 1]) and numpy.isfinite(ert1[idx, 1]): #alg0 stopped first
            medmaxeval = numpy.median(dataset0.maxevals)
            soliddata = data[data1[:, 1] <= medmaxeval, :]
            remainingert = ert1[idx-1:, :]
            isalg1 = True
        elif numpy.isinf(ert1[idx, 1]) and numpy.isfinite(ert0[idx, 1]): #alg1 stopped first
            medmaxeval = numpy.median(dataset1.maxevals)
            soliddata = data[data0[:, 1] <= medmaxeval, :]
            remainingert = ert0[idx-1:, :]
            isalg1 = False
        else:
            soliddata = isfinite

    plt.plot(soliddata[:, 0], soliddata[:, 1], label='%d-D' % dim,
             color=colors[i], linewidth=linewidth)
    plt.plot((soliddata[-1, 0], ), (soliddata[-1, 1], ), marker=markers[i],
             markersize=6*linewidth, markeredgewidth=linewidth,
             markeredgecolor=colors[i], markerfacecolor=None, color='w')

    #Number of successful trials on a linear scale!
    if not remainingert is None:
        #set_trace()
        additionallines = numpy.array([
        [remainingert[-1, 0] * 10.**(-0.2), remainingert[-1, 1]] + [0.] * (numpy.shape(remainingert)[1] - 2),
        [remainingert[-1, 0] * 10.**(-16.), remainingert[-1, 1]] + [0.] * (numpy.shape(remainingert)[1] - 2)])
        remainingert = numpy.concatenate((remainingert, additionallines))
        ydata = numpy.power(10., 2. * remainingert[:, 2])
        if isalg1:
            ydata = numpy.power(ydata, -1.)

        #set_trace()
        plt.plot(remainingert[:, 0], ydata,  color=colors[i],
             linewidth=linewidth, marker='d', markeredgecolor=colors[i],
             ls='--')
        plt.plot(remainingert[remainingert[:, 1] <= medmaxeval, 0],
                 ydata[remainingert[:, 1] <= medmaxeval],  color=colors[i],
                 linewidth=linewidth, marker='d', markeredgecolor=colors[i])

        #Go to the right? meaning with 0 as probability of success...?

    #completion2()


def main(dsList0, dsList1, outputdir,
         mintargetfunvalue = 1e-8, verbose=True):
    """From a list of IndexEntry, returns ERT figure."""

    plt.rc("axes", labelsize=20, titlesize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("font", size=20)
    plt.rc("legend", fontsize=20)

    dictFun0 = dsList0.dictByFunc()
    dictFun1 = dsList1.dictByFunc()

    for fun in set.union(set(dictFun0), set(dictFun1)):
        dictDim0 = dictFun0[fun].dictByDim()
        dictDim1 = dictFun1[fun].dictByDim()
        if isBenchmarkinfosFound:
            title = funInfos[fun]
        else:
            title = ''

        filename = os.path.join(outputdir,'cmpdata_f%d' % (fun))
        fig = plt.figure()
        dims = sorted(set.union(set(dictDim0), set(dictDim1)))
        for i, dim in enumerate(dims):
            try:
                if len(dictDim0[dim]) != 1 or len(dictDim1[dim]) != 1:
                    warnings.warn('Could not find some data for f%d in %d-D.'
                                  % (fun, dim))
                    continue
            except KeyError:
                warnings.warn('Could not find some data for f%d in %d-D.'
                              % (fun, dim))
                continue

            dataset0 = dictDim0[dim][0]
            dataset1 = dictDim1[dim][0]
            # TODO: warn if there are not one element in each of those dictionaries

            generatePlot(dataset0, dataset1, i, dim)

        #legend = True # if func in (1, 24, 101, 130):
        customizeFigure(fig, filename, mintargetfunvalue, title=title,
                        fileFormat=('png', 'pdf'), labels=['', ''],
                        legend=False, locLegend='best', verbose=verbose)

        #for i in h:
            #plt.setp(i,'color',colors[dim])
        #h = ppfig.createFigure(entry.arrayFullTab[:,[0,medianindex]], fig)
        #for i in h:
            #plt.setp(h,'color',colors[dim],'linestyle','--')
        #Do all this in createFigure?

    plt.rcdefaults()
