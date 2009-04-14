#! /usr/bin/env python

# Creates ENFEs and convergence figures for BBOB post-processing.

import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy
from pdb import set_trace
from bbob_pproc import bootstrap, pproc


colors = ('k', 'b', 'c', 'g', 'y', 'm', 'r', 'k', 'k', 'c', 'r', 'm')

#Get benchmark short infos.
funInfos = {}
isBenchmarkinfosFound = True
#infofile = os.path.join(os.path.split(__file__)[0], '..', '..',
                        #'benchmarkshortinfos.txt')
infofile = os.path.join(os.path.split(__file__)[0], 'benchmarkshortinfos.txt')

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
            lines.extend(plt.plot(data[tmp,0], yValues[tmp], label=label))
            lines.extend(plt.plot([data[tmp[-1],0]], [yValues[tmp[-1]]],
                                  marker='+'))
            #TODO: larger size.
        #else: ???

    return lines


def customizeFigure(figHandle, figureName = None, title='',
                    fileFormat=('png','eps'), labels=None,
                    legend=True, locLegend='best', verbose=True):
    """ Customize a figure by adding a legend, axis label, etc. At the
        end the figure is saved.

        Inputs:
        figHandle - handle to existing figure
        figureName - name of the output figure

        Optional Inputs:
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

    xmin, xmax = plt.xlim()
    plt.xlim(max(xmin, 1e-8), xmax)
    axisHandle.invert_xaxis()

    # Annotate figure
    if not labels is None: #Couldn't it be ''?
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


def generateData(indexEntry):
    res = []
    for i in indexEntry.hData:
        tmp = numpy.array(i[0])
        tmp2 = []
        for j in i[indexEntry.nbRuns+1:]:
            tmp2.append(j <= i[0])
        tmp = numpy.append(tmp, bootstrap.sp1(i[1:indexEntry.nbRuns + 1], 
                                              issuccessful=tmp2))
        tmp = numpy.append(tmp, bootstrap.prctile(i[1:indexEntry.nbRuns + 1], 
                                                  [0.5])[0])
        #set_trace()
        res.append(tmp)
    return numpy.vstack(res)

def computeERT(hdata):
    res = []
    nbRuns = (numpy.shape(hdata)[1]-1)/2
    for i in hdata:
        success = i[nbRuns+1:] <= i[0]
        tmp = [i[0]]
        tmp.extend(bootstrap.sp(i[1:nbRuns+1], issuccessful=success))
        res.append(tmp)
    #set_trace()
    return numpy.vstack(res)

def main(indexEntriesAlg0, indexEntriesAlg1, dimsOfInterest, outputdir,
         verbose=True):
    """From a list of IndexEntry, returns ERT figure."""

    plt.rc("axes", labelsize=20, titlesize=24)
    plt.rc("xtick", labelsize=20)
    plt.rc("ytick", labelsize=20)
    plt.rc("font", size=20)
    plt.rc("legend", fontsize=20)

    dictFunc0 = indexEntriesAlg0.dictByFunc()
    dictFunc1 = indexEntriesAlg1.dictByFunc()
    funcs = set.union(set(dictFunc0), set(dictFunc0))

    for func in funcs:
        dictFunc0[func] = dictFunc0[func].dictByDim()
        dictFunc1[func] = dictFunc1[func].dictByDim()
        if isBenchmarkinfosFound:
            title = funInfos[func]
        else:
            title = ''

        filename = os.path.join(outputdir,'cmpdata_f%d' % (func))
        fig = plt.figure()
        for i, dim in enumerate(dimsOfInterest):
            #compare dictFunc0[func][dim][0] and dictFunc1[func][dim][0]
            try:
                if len(dictFunc0[func][dim]) != 1 or len(dictFunc1[func][dim]) != 1:
                    warnings.warn('Could not find some data for f%d in %d-D.'
                                  % (func, dim))
                    #set_trace()
                    continue
            except KeyError:
                warnings.warn('Could not find some data for f%d in %d-D.'
                              % (func, dim))
                #set_trace()
                continue

            indexEntry0 = dictFunc0[func][dim][0]
            indexEntry1 = dictFunc1[func][dim][0]
            # align together and split again (the data are mixed together):
            res = pproc.alignData(pproc.HArrayMultiReader([indexEntry0.hData,
                                                           indexEntry1.hData]))
            idxM = (numpy.shape(res)[1]-1)/2
            idxCur = 1
            idxNext = idxCur+indexEntry0.nbRuns()
            data0 = res[:, numpy.r_[0, idxCur:idxNext]]
            data0 = numpy.hstack((data0,
                                  res[:, numpy.r_[idxM+idxCur:idxM+idxNext]]))
            data0 = computeERT(data0)
            data0 = data0[data0[:,0] <= min(indexEntry0.hData[0,0], indexEntry1.hData[0,0]),:]

            idxCur += indexEntry0.nbRuns()
            idxNext = idxCur+indexEntry1.nbRuns()
            data1 = res[:, numpy.r_[0, idxCur:idxNext]]
            data1 = numpy.hstack((data1,
                                  res[:, numpy.r_[idxM+idxCur:idxM+idxNext]]))
            data1 = computeERT(data1)
            data1 = data1[data1[:,0] <= min(indexEntry0.hData[0,0], indexEntry1.hData[0,0]),:]

            data = numpy.vstack((data0[:, 0],
                                 data1[:, 1]/data0[:, 1])).transpose()

            data = data[(data0[:, 2] > 0) * (data1[:, 2] > 0)]
            h = createFigure(data, label='%d-D' % dim, figHandle=fig)
            plt.setp(h, 'color', colors[i])
            plt.setp(h[0], 'label', '%d-D' % dim)

            if (data0[:, 2] == 0).any():
                #set_trace()
                tmp = data1[(data0[:, 2] == 0), 0:2]
                tmp = numpy.vstack([data1[(data0[:, 2] > 0) * (data1[:, 2] > 0)][-1, 0:2], tmp])
                #set_trace()
                tmp[:,1] = tmp[0,1] / tmp[:,1] * data[(data0[:, 2] > 0) * (data1[:, 2] > 0)][-1, 1]
                h = createFigure(tmp, figHandle=fig)
                plt.setp(h, 'color', colors[i])
            if (data1[:, 2] == 0).any():
                #set_trace()
                tmp = data0[(data1[:, 2] == 0), 0:2]
                tmp = numpy.vstack([data0[(data0[:, 2] > 0) * (data1[:, 2] > 0)][-1, 0:2], tmp])
                #set_trace()
                tmp[:,1] = tmp[:,1] / tmp[0,1] * data[(data0[:, 2] > 0) * (data1[:, 2] > 0)][-1, 1]
                h = createFigure(tmp, figHandle=fig)
                plt.setp(h, 'color', colors[i])

        legend = True #func in (1, 24, 101, 130)
        customizeFigure(fig, filename, title=title,
                        fileFormat=('png'), labels=['', ''],
                        legend=legend, locLegend='best', verbose=verbose)

        #for i in h:
            #plt.setp(i,'color',colors[dim])
        #h = ppfig.createFigure(entry.arrayFullTab[:,[0,medianindex]], fig)
        #for i in h:
            #plt.setp(h,'color',colors[dim],'linestyle','--')
        #Do all this in createFigure?

    plt.rcdefaults()
