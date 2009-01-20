#! /usr/bin/env python

# Creates ENFEs and convergence figures for BBOB post-processing.

import os
import sys
import matplotlib.pyplot as plt
import scipy
from pdb import set_trace
from bbob_pproc import bootstrap

plt.rc("axes", labelsize=20, titlesize=24)
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)
plt.rc("font", size=20)
plt.rc("legend", fontsize=20)
#valuesOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
#colors = {1.0:'b', 1.0e-2:'g', 1.0e-4:'r', 1.0e-6:'c', 1.0e-8:'m'} #TODO colormaps!
colors = ('b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm')  # should not be too short

#Either we read it in a file (flexibility) or we hard code it here.
funInfos = {}
isBenchmarkinfosFound = True
try:
    infofile = os.path.join(os.path.split(__file__)[0], '..', '..',
                            'benchmarkshortinfos')
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
    print 'Could not find benchmarkshortinfos file. '\
          'Titles in scaling figures will not be displayed.'


def createFigure(data, figHandle=None):
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
        tmp = scipy.where(scipy.isfinite(data[:,i]))[0]
        if tmp.size > 0:
            lines.extend(plt.plot(data[tmp,0], yValues[tmp]))
            lines.extend(plt.plot([data[tmp[-1],0]], [yValues[tmp[-1]]],
                                  marker='+'))
            #TODO: larger size.
        #else: ???

    return lines


def customizeFigure(figHandle, figureName = None, title='',
                    fileFormat=('png','eps'), labels=None, 
                    scale=['linear','linear'], legend=list(), locLegend='best',
                    verbose=True):
    """ Customize a figure by adding a legend, axis label, etc. At the
        end the figure is saved.

        Inputs:
        figHandle - handle to existing figure
        figureName - name of the output figure

        Optional Inputs:
        fileFormat - list of formats of output files
        labels - list with xlabel and ylabel
        scale - scale for x-axis and y-axis
        legend - if this list is not empty a legend will be generated from
                 the entries of this list
        locLegend - location of legend

    """

    # Input checking

    # Get axis handle and set scale for each axis
    axisHandle = figHandle.gca()
    axisHandle.set_xscale(scale[0])
    axisHandle.set_yscale(scale[1])

    tmp = axisHandle.get_xlim()
    # axisHandle.set_xlim(tmp[0], min(tmp[1], 40))  
    axisHandle.set_xlim(1.9, 45)  # TODO should become variable
    # Annotate figure
    if labels is None: #Couldn't it be ''?
        axisHandle.set_xlabel(labels[0])
        axisHandle.set_ylabel(labels[1])
    #axisHandle.invert_xaxis()

    # Grid options
    axisHandle.grid('True')
    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(scipy.log10(i)))
    axisHandle.set_yticklabels(tmp2)

    # Legend
    if len(legend) > 0:
        axisHandle.legend(legend,locLegend)
    axisHandle.set_title(title)

    # Save figure
    if not (figureName is None or fileFormat is None):
        if isinstance(fileFormat, basestring):
            plt.savefig(figureName + '.' + fileFormat, dpi = 120,
                        format = entry)
            if verbose:
                print 'Wrote figure in %s.' %(figureName + '.' + fileFormat)
        else:
            #TODO: that is if fileFormat is iterable.
            for entry in fileFormat:
                plt.savefig(figureName + '.' + entry, dpi = 120,
                            format = entry)
                if verbose:
                    print 'Wrote figure in %s.' %(figureName + '.' + entry)

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
        funcs.add(elem.funcId)
        dims.add(elem.dim)

    return sortByFunc


def generateData(indexEntry, targetFuncValue):
    """Returns data to be plotted from indexEntry and the target function value."""

    res = []

    for i in indexEntry.hData:
        if i[0] <= targetFuncValue:
            tmp = []
            for j in i[indexEntry.nbRuns+1:]:
                tmp.append(j <= i[0])
            res.extend(bootstrap.sp1(i[1:indexEntry.nbRuns + 1], 
                                     issuccessful=tmp))
            res.append(bootstrap.prctile(i[1:indexEntry.nbRuns + 1], 50)[0])
            break

    return scipy.array(res)


def main(indexEntries, valuesOfInterest, outputdir, verbose=True):
    """From a list of IndexEntry, returns a convergence and ENFEs figure vs dim

    """

    sortByFunc = sortIndexEntries(indexEntries)
    for func in sortByFunc:
        filename = os.path.join(outputdir,'ppdata_f%d' % (func))
        fig = plt.figure()

        for i in range(len(valuesOfInterest)):
            tmp = []
            for dim in sorted(sortByFunc[func]):
                tmp2 = generateData(sortByFunc[func][dim], valuesOfInterest[i])
                if len(tmp2) > 0:
                    tmp.append(scipy.append(dim, tmp2))

            if len(tmp) > 0:
                tmp = scipy.vstack(tmp)
                h = createFigure(tmp[:, [0, 1]], fig) #ENFEs
                for j in h:
                    plt.setp(j,'color',colors[i])

                h = createFigure(tmp[:,[0, -1]], fig) #median
                for j in h:
                    plt.setp(j,'color',colors[i],'linestyle','--')
                    #Do all this in createFigure

        #TODO: legend

        if isBenchmarkinfosFound:
            title = funInfos[func]
        else:
            title = ''

        customizeFigure(fig, filename, title=title,
                        fileFormat=('eps','png'), labels=['', ''],
                        scale=['log','log'], locLegend='best',
                        verbose=verbose)

        plt.close(fig)

    # TODO: how do we make a user define what color or line style?
