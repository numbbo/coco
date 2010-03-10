#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates ERTs and convergence figures for the comparison of 2 algorithms."""

from __future__ import absolute_import

import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy
from pdb import set_trace
from bbob_pproc import bootstrap, readalign
from bbob_pproc.bootstrap import ranksums
#try:
    #supersede this module own ranksums method
    #from scipy.stats import ranksums as ranksums
#except ImportError:
    #from bbob_pproc.bootstrap import ranksums
    #pass

colors = ('k', 'b', 'c', 'g', 'y', 'm', 'r', 'k', 'k', 'c', 'r', 'm')
markers = ('o', 'v', 's', '+', 'x', 'D')
linewidth = 3

figformat = ('eps', 'pdf') # Controls the output when using the main method

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

#TODO improve interface.
def beautify(figHandle, figureName=None, xmin=None, title='',
                    fileFormat=('pdf','eps'), legend=True, locLegend='best',
                    verbose=True):
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
        plt.xlim(xmin=xmin)
    plt.xlim(xmax=1000)

    axisHandle.invert_xaxis()

    # Annotate figure
    #axisHandle.set_xlabel(r'$ \log 10(\Delta \mathrm{ftarget}) $')
    #axisHandle.set_ylabel(r'$ \log 10(ERT1 / ERT0) $')
    axisHandle.set_xlabel('log10(Delta ftarget)')
    axisHandle.set_ylabel('log10(ERT1/ERT0)')

    # Grid options
    axisHandle.grid('True')
    tmp = axisHandle.get_xticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    axisHandle.set_xticklabels(tmp2)
    ymin, ymax = plt.ylim()
    ybnd = max(1./ymin, ymax)
    plt.ylim(1./ybnd, ybnd)

    tmp = axisHandle.get_yticks()
    #set_trace()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    axisHandle.set_yticklabels(tmp2)

    tmp = axisHandle.get_yticks(minor=True)
    #Hack:
    tmp[tmp<1] = sorted(1/(tmp[tmp<1]*numpy.power(10, -2*numpy.floor(numpy.log10(tmp[tmp<1]))-1)))
    tmp = tmp[tmp<ybnd] # TODO: watch this...
    tmp = tmp[tmp>1./ybnd] # TODO: watch this...
    axisHandle.set_yticks(tmp, minor=True)

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
    return numpy.vstack(res)

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

def generatelogERTData(ert0, dataset0, ert1, dataset1):
    data = numpy.vstack((ert0[:, 0], (ert1[:, 1]/ert0[:, 1]))).transpose()
    soliddata = data
    isfinite = data[numpy.isfinite(ert0[:, 1]) * numpy.isfinite(ert1[:, 1]), :]
    isfiniteidx = len(isfinite)
    if isfiniteidx < len(data): # which is the same as len(ert1) and len(ert0)
        if (numpy.isinf(ert0[isfiniteidx, 1])
            and numpy.isfinite(ert1[isfiniteidx, 1])): #alg0 stopped first
            medmaxeval = numpy.median(dataset0.maxevals)
            soliddata = data[ert1[:isfiniteidx, 1] <= medmaxeval, :]
        elif (numpy.isinf(ert1[isfiniteidx, 1])
              and numpy.isfinite(ert0[isfiniteidx, 1])): #alg1 stopped first
            medmaxeval = numpy.median(dataset1.maxevals)
            soliddata = data[ert0[:isfiniteidx, 1] <= medmaxeval, :]

    return soliddata[:, 0:2], isfinite[:, 0:2]

def generatePSData(ert0, ert1):
    remainingert = None
    isfinite = numpy.isfinite(ert0[:, 1]) * numpy.isfinite(ert1[:, 1])
    isfiniteidx = len(numpy.nonzero(isfinite)[0])

    if isfiniteidx != len(ert0): # which is the same as len(ert1)
        if (numpy.isinf(ert0[isfiniteidx, 1])
            and numpy.isfinite(ert1[isfiniteidx, 1])): #alg0 stopped first
            remainingert = ert1[isfiniteidx:, :]
            isalg1 = True
            # isfiniteidx-1 to keep the last value for which both ert was still finite
        elif (numpy.isinf(ert1[isfiniteidx, 1])
              and numpy.isfinite(ert0[isfiniteidx, 1])): #alg1 stopped first
            remainingert = ert0[isfiniteidx:, :]
            # isfiniteidx-1 to keep the last value for which both ert was still finite
            isalg1 = False

    else:
        return None

    # Line added to the data to show the probability of success going to
    # zero, TODO: the value 10**(-0.2) is benchmark dependent
    additionallines = numpy.array([[remainingert[-1, 0] * 10.**(-0.2),
         remainingert[-1, 1]] + [0.] * (numpy.shape(remainingert)[1] - 2)])
    remainingert = numpy.concatenate((remainingert, additionallines))
    ydata = numpy.power(10., 2. * remainingert[:, 2])
    # have the probability of success below 10**0 if alg0 stopped first,
    # above 10**0 otherwise
    if isalg1:
        ydata = numpy.power(ydata, -1.)
    remainingert[:, 2] = ydata

    return remainingert[:, numpy.r_[0, 2]]

def generatePSData2(ert0, ert1, plotdata):
    xdata = None
    isfinite = numpy.isfinite(ert0[:, 1]) * numpy.isfinite(ert1[:, 1])
    isfiniteidx = len(numpy.nonzero(isfinite)[0])

    if isfiniteidx != len(ert0): # which is the same as len(ert1)
        if (numpy.isinf(ert0[isfiniteidx, 1])
            and numpy.isfinite(ert1[isfiniteidx, 1])): #alg0 stopped first
            xdata = ert1[isfiniteidx-1:, 0]
            isalg1 = True
            # isfiniteidx-1 to keep the last value for which both ert was still finite
        elif (numpy.isinf(ert1[isfiniteidx, 1])
              and numpy.isfinite(ert0[isfiniteidx, 1])): #alg1 stopped first
            xdata = ert0[isfiniteidx-1:, 0]
            # isfiniteidx-1 to keep the last value for which both ert was still finite
            isalg1 = False
    else:
        return None

    ydata = [ert1[isfiniteidx-1, 1]/ert0[isfiniteidx-1, 1]] * len(xdata)
    return numpy.transpose(numpy.vstack((xdata, numpy.array(ydata))))

def generatePlotData(data, minfvalue):
    """Process data according to a xmax bound.
       cuts off some data if minfvalue is defined.
    """

    if minfvalue is None:
        return data, False

    isPlotFinished = False
    plotdata = data
    idx = numpy.shape(numpy.nonzero(plotdata[:, 0] > minfvalue))[1]
    if idx < len(plotdata):
        plotdata = plotdata[0:idx+1, :] # the last value is smaller or equal to minfvalue
        # Hack: replace function values of zero by min/max function values.
        idxzero = (plotdata[:, 0] == 0)
        if len(plotdata[idxzero == False, 0]) > 0:
            plotdata[idxzero, 0] = (min(minfvalue,
                                        numpy.min(plotdata[idxzero == False, 0]))
                                    / numpy.max(plotdata[:, 0]))
        else:
            # in this case all x data is zero.
            #TODO: How should we deal with this case?
            plotdata[idxzero, 0] = minfvalue / numpy.max(plotdata[:, 0])

        isPlotFinished = True
    return plotdata, isPlotFinished

def generatePlot(dataset0, dataset1, i, dim, minfvalue=None, nbtests=1):

    [data0, data1] = alignData(dataset0, dataset1)
    ert0 = computeERT(data0, dataset0.maxevals)
    ert1 = computeERT(data1, dataset1.maxevals)

    handles = []
    #resdata = []
    resplotdata = []
    isPlotFinished = False

    soliddata, data = generatelogERTData(ert0, dataset0, ert1, dataset1)
    plotdata, isPlotFinished = generatePlotData(soliddata, minfvalue)
    resplotdata.append(plotdata.copy())

    # Solid line for when the ert of the still going algorithm is smaller than
    # the median of the number of function evaluations of the other algorithm.
    h = plt.plot(plotdata[:, 0], plotdata[:, 1], label='%d-D' % dim,
                 color=colors[i], linewidth=linewidth)
    handles.extend(h)

    # A marker is put when the ERT is larger than the median of number of
    # function evaluations of the other algorithm.
    if not isPlotFinished:
        h = plt.plot((plotdata[-1, 0], ), (plotdata[-1, 1], ),
                     marker=markers[i], markersize=6*linewidth,
                     markeredgewidth=linewidth, markeredgecolor=colors[i],
                     markerfacecolor='None', color='w')
        handles.extend(h)

    # Dashed line for while both algorithm have a finite ert
    if not isPlotFinished:
        plotdata, isPlotFinished = generatePlotData(data, minfvalue) # replace isPlotFinished
        resplotdata.append(plotdata.copy())
        h = plt.plot(plotdata[:, 0], plotdata[:, 1], color=colors[i],
                     linewidth=linewidth, ls='--')
        handles.extend(h)

        if not isPlotFinished:
            # Marker for when an algorithm stopped$
            h = plt.plot((plotdata[-1, 0], ), (plotdata[-1, 1], ),
                         marker=markers[i], markersize=6*linewidth,
                         markeredgewidth=linewidth, markeredgecolor=colors[i],
                         markerfacecolor='None', color='w')
            handles.extend(h)

    #Number of successful trials on a linear scale
    if not isPlotFinished:
        data = generatePSData(ert0, ert1)
        if not data is None:
            plotdata, isPlotFinished = generatePlotData(data, minfvalue)
            resplotdata.append(plotdata.copy())

            if any(numpy.isfinite(plotdata[:, 0])):
                h = plt.plot(plotdata[:, 0], plotdata[:, 1],  color=colors[i],
                             linewidth=linewidth, marker='d', markeredgecolor=colors[i],
                             ls='--')
                handles.extend(h)

        #handles.append(plt.plot(tmpplotdata[tmpplotdata[:, 1] <= medmaxeval, 0],
                                #tmpplotdata[tmpplotdata[:, 1] <= medmaxeval],  color=colors[i],
                                #linewidth=linewidth, marker='d', markeredgecolor=colors[i]))

    annot = {}
    annot["isPlotFinished"] = isPlotFinished
    annot["coord"] = plotdata[-1].copy()
    # Should correspond to the last value for which delta ftarget is larger than minfvalue

    tmp = data0[:, 0] <= annot["coord"][0]
    if any(tmp):
        line0 = data0[tmp, :][0]
    else:
        tmp = [annot["coord"][0]]
        tmp.extend([numpy.nan] * (numpy.shape(data0)[1] - 1))
        line0 = numpy.array(tmp)
    nbsucc0 = numpy.sum(numpy.isnan(line0[1:]) == False)

    tmp = data1[:, 0] <= annot["coord"][0]
    line1 = data1[-1]
    if any(tmp):
        line1 = data1[tmp, :][0]
    else:
        tmp = [annot["coord"][0]]
        tmp.extend([numpy.nan] * (numpy.shape(data1)[1] - 1))
        line1 = numpy.array(tmp)
    nbsucc1 = numpy.sum(numpy.isnan(line1[1:]) == False)

    if (nbsucc0 == 0 and nbsucc1 == 0) or (nbsucc0 > 5 and nbsucc1 >= 5):
        txt = '%d-D' % dim
    else:
        txt = str(int(nbsucc1))
        if nbsucc1 > 9:
            txt = '>9'

        tmp = str(int(nbsucc0))
        if nbsucc0 > 9:
            tmp = '>9'
        txt += '/' + tmp

    annot["label"] = txt

    # Do the rank-sum-test on the final values... (the larger the better)
    # Prepare the data:
    line0[1:] = numpy.power(line0[1:], -1.)
    line0[1:][numpy.isnan(line0[1:])] = -dataset0.finalfunvals[numpy.isnan(line0[1:])]
    line1[1:] = numpy.power(line1[1:], -1.)
    line1[1:][numpy.isnan(line1[1:])] = -dataset1.finalfunvals[numpy.isnan(line1[1:])]

    # one-tailed statistics: scipy.stats.mannwhitneyu, two-tailed statistics: scipy.stats.ranksums
    z, p = ranksums(line0[1:], line1[1:])

    annot["p"] = p

    return handles, annot

def generatePlot2(dataset0, dataset1, i, dim, minfvalue=None, nbtests=1):
    """Generate a graph.
    Differences with generatePlot:
      no dashed,
      no graph for the success probability
      annotations for the success probability
    """

    [data0, data1] = alignData(dataset0, dataset1)
    ert0 = computeERT(data0, dataset0.maxevals)
    ert1 = computeERT(data1, dataset1.maxevals)

    handles = []
    #resdata = []
    resplotdata = []
    isPlotFinished = False

    soliddata, data = generatelogERTData(ert0, dataset0, ert1, dataset1)
    plotdata, isPlotFinished = generatePlotData(soliddata, minfvalue)
    resplotdata.append(plotdata.copy())

    # Solid line while both algorithm have a finite ert
    plotdata, isPlotFinished = generatePlotData(data, minfvalue) # replace isPlotFinished
    resplotdata.append(plotdata.copy())
    h = plt.plot(plotdata[:, 0], plotdata[:, 1], color=colors[i],
                 linewidth=linewidth)
    handles.extend(h)

    if not isPlotFinished:
        # Marker for when an algorithm stopped
        h = plt.plot((plotdata[-1, 0], ), (plotdata[-1, 1], ),
                     marker=markers[i], markersize=6*linewidth,
                     markeredgewidth=linewidth, markeredgecolor=colors[i],
                     markerfacecolor='None', color=colors[i])

        handles.extend(h)

    #for the legend
    plt.plot((plotdata[-1, 0], ), (plotdata[-1, 1], ),
             marker=markers[i], markeredgecolor=colors[i],
             markerfacecolor='None', color=colors[i], label='%2d-D' % dim)

    # At this point one algorithm stopped.
    # Straight line for until the second algorithm stops
    if not isPlotFinished:
        data = generatePSData2(ert0, ert1, plotdata)
        if not data is None:
            plotdata, isPlotFinished = generatePlotData(data, minfvalue)
            resplotdata.append(plotdata.copy())

            if any(numpy.isfinite(plotdata[:, 0])):
                h = plt.plot(plotdata[:, 0], plotdata[:, 1],  color=colors[i],
                             linewidth=linewidth, markeredgecolor=colors[i],
                             ls='-')
                handles.extend(h)

    annot = {}
    annot["isPlotFinished"] = isPlotFinished
    annot["coord"] = plotdata[-1].copy()
    # Should correspond to the last value for which delta ftarget is larger than minfvalue

    tmp = data0[:, 0] <= annot["coord"][0]
    if any(tmp):
        line0 = data0[tmp, :][0]
    else:
        tmp = [annot["coord"][0]]
        tmp.extend([numpy.nan] * (numpy.shape(data0)[1] - 1))
        line0 = numpy.array(tmp)
    nbsucc0 = numpy.sum(numpy.isnan(line0[1:]) == False)

    tmp = data1[:, 0] <= annot["coord"][0]
    line1 = data1[-1]
    if any(tmp):
        line1 = data1[tmp, :][0]
    else:
        tmp = [annot["coord"][0]]
        tmp.extend([numpy.nan] * (numpy.shape(data1)[1] - 1))
        line1 = numpy.array(tmp)
    nbsucc1 = numpy.sum(numpy.isnan(line1[1:]) == False)

    if (nbsucc0 == 0 and nbsucc1 == 0) or (nbsucc0 > 5 and nbsucc1 >= 5):
        txt = '%d-D' % dim
    else:
        txt = str(int(nbsucc1))
        if nbsucc1 > 9:
            txt = '>9'

        tmp = str(int(nbsucc0))
        if nbsucc0 > 9:
            tmp = '>9'
        txt += '/' + tmp

    annot["label"] = txt

    # Do the rank-sum-test on the final values... (the larger the better)
    # Prepare the data:
    line0[1:] = numpy.power(line0[1:], -1.)
    line0[1:][numpy.isnan(line0[1:])] = -dataset0.finalfunvals[numpy.isnan(line0[1:])]
    line1[1:] = numpy.power(line1[1:], -1.)
    line1[1:][numpy.isnan(line1[1:])] = -dataset1.finalfunvals[numpy.isnan(line1[1:])]

    # one-tailed statistics: scipy.stats.mannwhitneyu, two-tailed statistics: scipy.stats.ranksums
    z, p = ranksums(line0[1:], line1[1:])

    annot["p"] = p

    return handles, annot

def annotate(annotations, minfvalue):
    # End of graph annotation:
    # Determine where to put the annotation
    handles = []
    nbtests = len(annotations)
    annotcoords = []
    axh = plt.gca()
    axh.set_yscale('log')
    ylim = plt.getp(axh, 'ylim')
    yrange = (numpy.ceil(numpy.log10(ylim[1]))
              - numpy.floor(numpy.log10(ylim[0])))
    space = numpy.power(10., yrange/40.)
    axh.set_yscale('linear')
    yrange = 2.

    for i, a in enumerate(annotations):
        # Set the annotation
        ha = "left"
        va = "center"
        annotcoord = a["coord"].copy()
        if a["isPlotFinished"]: # should be false if minfvalue is None
            annotcoord[0] = minfvalue * 10.**(-0.1)
            #annotcoord[1] should be moved depending on the values of the others...
        else:
            factor = 1.3 * yrange / 2.
            va = "top"
            if annotcoord[1] < 1:
                factor **= -1.
                va = "bottom"
            annotcoord[1] *= factor
            ha = "center"
        #a["annotcoord"] = annotcoord
        annotcoords.append(numpy.hstack((annotcoord, i)))
        a["ha"] = ha
        a["va"] = va

        # Set the correct line in data0 and data1
        if (nbtests * a["p"]) < 0.05:
            nbstars = -numpy.ceil(numpy.log10(nbtests * a["p"]))
            incr = 1.5
            tmp = a["coord"][0]
            if not minfvalue is None:
                tmp = max(a["coord"][0], minfvalue)

            xtmp = tmp * numpy.power(incr, numpy.arange(1., 1. + nbstars))
            # the additional slicing [0:int(nbstars)] is due to
            # numpy.arange(1., 1. - 0.1 * nbstars, -0.1) not having the right number
            # of elements due to numerical error
            ytmp = [a["coord"][1]] * nbstars
            try:
                h = plt.plot(xtmp, ytmp, marker='*', ls='', color='w',
                             markersize=5*linewidth, markeredgecolor='k',
                             markerfacecolor='None',
                             zorder=20, markeredgewidth = 0.4 * linewidth)
            except KeyError:
                #Version problem
                h = plt.plot(xtmp, ytmp, marker='+', ls='', color='w',
                             markersize=2.5*linewidth, markeredgecolor='k',
                             zorder=20, markeredgewidth = 0.2 * linewidth)
            #h = plt.plot(xtmp, ytmp, marker='*', ls='', markersize=4*linewidth,
                         #markeredgecolor='k', zorder=20)

            #plt.setp(h, "markerfacecolor", "none") #transparent
            handles.extend(h)

        #handles.append(plt.text(annotcoords[-1][0], annotcoords[-1][1],
                       #a["label"], fontsize=10,
                       #horizontalalignment=ha, verticalalignment=va))

    if len(annotcoords) > 1 and not minfvalue is None:
        annotcoords = numpy.vstack(annotcoords)
        # The following lines are used to spread out the annotations on the
        # right of the figure.
        closeannot = annotcoords[annotcoords[:, 0] == minfvalue * 10.**(-0.1), :]

        tmp = numpy.argsort(closeannot[:, 1])
        if len(tmp) > 0:
            closeannot = closeannot[tmp, :]
        for i in reversed(range(0, len(closeannot)/2)):
            while closeannot[i+1][1] / closeannot[i][1] < space:
                closeannot[i][1] /= numpy.sqrt(space)
        for i in range(int(round(len(closeannot)/2.)), len(closeannot)):
            while closeannot[i][1] / closeannot[i-1][1] < space:
                closeannot[i][1] *= numpy.sqrt(space)
        for i in closeannot:
            annotcoords[annotcoords[:, 2] == i[2], 1] = i[1]

    for i, a in enumerate(annotations):
        coords = annotcoords[i]
        handles.append(plt.text(coords[0], coords[1],
                       a["label"], fontsize=15,
                       horizontalalignment=a["ha"], verticalalignment=a["va"]))

    return handles

def main(dsList0, dsList1, outputdir, minfvalue = 1e-8, verbose=True):
    """From a list of IndexEntry, returns ERT figure."""

    plt.rc("axes", labelsize=20, titlesize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("font", size=20)
    plt.rc("legend", fontsize=20)

    dictFun0 = dsList0.dictByFunc()
    dictFun1 = dsList1.dictByFunc()

    for func in set.union(set(dictFun0), set(dictFun1)):
        annotations = []
        try:
            dictDim0 = dictFun0[func].dictByDim()
            dictDim1 = dictFun1[func].dictByDim()
        except KeyError:
            txt = ("Data on function f%d could not be found for " % (func) +
                   "both algorithms.")
            warnings.warn(txt)
            continue
        if isBenchmarkinfosFound:
            title = funInfos[func]
        else:
            title = ''

        filename = os.path.join(outputdir,'ppcmpfig_f%d' % (func))
        fig = plt.figure()
        dims = sorted(set.intersection(set(dictDim0), set(dictDim1)))
        #set_trace()
        for i, dim in enumerate((2, 3, 5, 10, 20, 40)):
            try:
                dataset0 = dictDim0[dim][0]
                dataset1 = dictDim1[dim][0]
                #if len(dictDim0[dim]) != 1 or len(dictDim1[dim]) != 1:
                    #warnings.warn('Could not find some data for f%d in %d-D.'
                                  #% (fun, dim))
                    #continue
            except KeyError:
                #warnings.warn('Could not find some data for f%d in %d-D.'
                              #% (fun, dim))
                continue

            # TODO: warn if there are not one element in each of those dictionaries
            h, a = generatePlot(dataset0, dataset1, i, dim, minfvalue, len(dims))
            annotations.append(a)

        annotate(annotations, minfvalue)
        legend = False
        if func in (1, 24, 101, 130):
            legend = True
        beautify(fig, filename, minfvalue, title=title,
                        fileFormat=figformat,
                        legend=legend, locLegend='best', verbose=verbose)
        #for i in h:
            #plt.setp(i,'color',colors[dim])
        #h = ppfig.createFigure(entry.arrayFullTab[:,[0,medianindex]], fig)
        #for i in h:
            #plt.setp(h,'color',colors[dim],'linestyle','--')
        #Do all this in createFigure?

    plt.rcdefaults()
