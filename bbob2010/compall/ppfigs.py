#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates ERTs and convergence figures for multiple algorithms."""

import os
import sys
import matplotlib.pyplot as plt
import numpy
from pdb import set_trace
from bbob_pproc import bootstrap, bestalg, pproc
from bbob_pproc.ppfig import saveFigure
from bbob_pproc.dataoutput import algPlotInfos

# styles = [{'color': 'k', 'marker': 'o', 'markeredgecolor': 'k'},
#           {'color': 'b'},
#           {'color': 'c', 'marker': 'v', 'markeredgecolor': 'c'},
#           {'color': 'g'},
#           {'color': 'y', 'marker': '^', 'markeredgecolor': 'y'},
#           {'color': 'm'},
#           {'color': 'r', 'marker': 's', 'markeredgecolor': 'r'}] # sort of rainbow style
styles = [{'marker': 'o', 'linestyle': '-', 'color': 'b'},
          {'marker': 'd', 'linestyle': '-', 'color': 'g'},
          {'marker': 's', 'linestyle': '-', 'color': 'r'},
          {'marker': 'v', 'linestyle': '-', 'color': 'c'},
          {'marker': '*', 'linestyle': '-', 'color': 'm'},
          {'marker': 'h', 'linestyle': '-', 'color': 'y'},
          {'marker': '^', 'linestyle': '-', 'color': 'k'},
          {'marker': 'p', 'linestyle': '-', 'color': 'b'},
          {'marker': 'H', 'linestyle': '-', 'color': 'g'},
          {'marker': '<', 'linestyle': '-', 'color': 'r'},
          {'marker': 'D', 'linestyle': '-', 'color': 'c'},
          {'marker': '>', 'linestyle': '-', 'color': 'm'},
          {'marker': '1', 'linestyle': '-', 'color': 'y'},
          {'marker': '2', 'linestyle': '-', 'color': 'k'},
          {'marker': '3', 'linestyle': '-', 'color': 'b'},
          {'marker': '4', 'linestyle': '-', 'color': 'g'}]

show_algorithms = []
fontsize = 20.0
legend = False

#Get benchmark short infos.
funInfos = {}
figformat = ('eps', 'pdf') # Controls the output when using the main method
isBenchmarkinfosFound = True
infofile = os.path.join(os.path.split(__file__)[0], '..', '..',
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
          'Titles in figures will not be displayed.'

def plotLegend(handles, maxval=None):
    """Display right-side legend.
    
    Sorted from smaller to larger y-coordinate values.
    
    """

    ys = {}
    lh = 0 # Number of labels to display on the right
    if not maxval:
        maxval = []
        for h in handles:
            x2 = []
            y2 = []
            for i in h:
                x2.append(plt.getp(i, "xdata"))
            x2 = numpy.sort(numpy.hstack(x2))
            maxval.append(max(x2))
        maxval = max(maxval)

    for h in handles:
        x2 = []
        y2 = []

        for i in h:
            x2.append(plt.getp(i, "xdata"))
            y2.append(plt.getp(i, "ydata"))

        x2 = numpy.array(numpy.hstack(x2))
        y2 = numpy.array(numpy.hstack(y2))
        tmp = numpy.argsort(x2)
        x2 = x2[tmp]
        y2 = y2[tmp]
        h = h[-1]

        # ybis is used to sort in case of ties
        try:
            tmp = x2 <= maxval
            y = y2[tmp][-1]
            ybis = y2[tmp][y2[tmp] < y]
            if len(ybis) > 0:
                ybis = ybis[-1]
            else:
                ybis = y2[tmp][-2]
            ys.setdefault(y, {}).setdefault(ybis, []).append(h)
            lh += 1
        except IndexError:
            pass

    if len(show_algorithms) > 0:
        lh = min(lh, len(show_algorithms))

    if lh <= 1:
        lh = 2

    ymin, ymax = plt.ylim()
    xmin, xmax = plt.xlim()

    i = 0 # loop over the elements of ys
    for j in sorted(ys.keys()):
        for k in sorted(ys[j].keys()):
            #enforce best 2009 comes first in case of equality
            tmp = []
            for h in ys[j][k]:
                if plt.getp(h, 'label') == 'best 2009':
                    tmp.insert(0, h)
                else:
                    tmp.append(h)
            #tmp.reverse()
            ys[j][k] = tmp

            for h in ys[j][k]:
                if (not plt.getp(h, 'label').startswith('_line') and
                    (len(show_algorithms) == 0 or
                     plt.getp(h, 'label') in show_algorithms)):
                    y = 0.02 + i * 0.96/(lh-1)
                    # transform y in the axis coordinates
                    #inv = plt.gca().transLimits.inverted()
                    #legx, ydat = inv.transform((.9, y))
                    #leglabx, ydat = inv.transform((.92, y))
                    #set_trace()

                    ydat = 10**(y * numpy.log10(ymax/ymin)) * ymin
                    legx = 10**(.85 * numpy.log10(xmax/xmin)) * xmin
                    leglabx = 10**(.87 * numpy.log10(xmax/xmin)) * xmin
                    tmp = {}
                    for attr in ('lw', 'ls', 'marker',
                                 'markeredgewidth', 'markerfacecolor',
                                 'markeredgecolor', 'markersize', 'zorder'):
                        tmp[attr] = plt.getp(h, attr)
                    plt.plot((maxval, legx), (j, ydat),
                             color=plt.getp(h, 'markeredgecolor'), **tmp)

                    plt.text(leglabx, ydat,
                             plt.getp(h, 'label'), horizontalalignment="left",
                             verticalalignment="center", size=fontsize)
                    i += 1

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    if maxval:
        plt.axvline(maxval, color='k')

def beautify(rightlegend=False):
    """ Customize a figure by adding a legend, axis label, etc and save to a file.
        Is identical to beautify except for the linear and quadratic scaling
        lines which are quadratic and cubic

        Keyword arguments:
        rightlegend -- if True, makes some space on the right for legend

    """

    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_xscale("log")
    try:
        axisHandle.set_yscale("log")
    except OverflowError:
        set_trace()

    # Grid options
    axisHandle.grid(True)

    ymin, ymax = plt.ylim()

    # quadratic and cubic "grid"
    plt.plot((2,200), (1, 1e2), 'k:', zorder=-1)
    plt.plot((2,200), (1, 1e4), 'k:', zorder=-1)
    plt.plot((2,200), (1e3, 1e5), 'k:', zorder=-1)  
    plt.plot((2,200), (1e3, 1e7), 'k:', zorder=-1)
    plt.plot((2,200), (1e6, 1e8), 'k:', zorder=-1)  
    plt.plot((2,200), (1e6, 1e10), 'k:', zorder=-1)

    plt.ylim(ymin=10**-0.2, ymax=ymax) # Set back the default maximum.

    # ticks on axes
    #axisHandle.invert_xaxis()
    dimticklist = (2, 3, 4, 5, 10, 20, 40)  # TODO: should become input arg at some point? 
    dimannlist = (2, 3, '', 5, 10, 20, 40)  # TODO: should become input arg at some point? 
    # TODO: All these should depend on (xlim, ylim)

    axisHandle.set_xticks(dimticklist)
    axisHandle.set_xticklabels([str(n) for n in dimannlist])

    # axes limites
    if rightlegend:
        plt.xlim(1.8, 101) # 101 is 10 ** (numpy.log10(45/1.8)*1.25) * 1.8
    else:
        plt.xlim(1.8, 45)                # Should depend on xmin and xmax

    tmp = axisHandle.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    axisHandle.set_yticklabels(tmp2)

def generateData(dataSet, target):
    """Returns an array of results to be plotted.

    Oth column is ert, 1st is the success rate, 2nd the number of successes,
    3rd the mean of the number of function evaluations, and 4th the median
    of number of function evaluations of successful runs or numpy.nan.
    """

    res = []

    data = dataSet.detEvals([target])[0]
    succ = (numpy.isnan(data) == False)
    data[numpy.isnan(data)] = dataSet.maxevals[numpy.isnan(data)]
    res.extend(bootstrap.sp(data, issuccessful=succ, allowinf=False))
    res.append(numpy.mean(data))
    if res[2] > 0:
        res.append(bootstrap.prctile(data[succ], 50)[0])
    else:
        res.append(numpy.nan)

    return res

def main(dictAlg, sortedAlgs, target, outputdir, verbose=True):
    """From a DataSetList, returns figures showing the scaling: ERT/dim vs dim.
    
    One function and one target per figure.
    
    """

    dictFunc = pproc.dictAlgByFun(dictAlg)

    for f in dictFunc:
        filename = os.path.join(outputdir,'ppfigs_f%03d' % (f))
        handles = []
        for alg in sortedAlgs:
            dictDim = dictFunc[f][alg].dictByDim()

            #Collect data
            dimert = []
            ert = []
            dimnbsucc = []
            ynbsucc = []
            nbsucc = []
            dimmaxevals = []
            maxevals = []
            dimmedian = []
            medianfes = []
            infos = set()
            for dim in sorted(dictDim):
                entry = dictDim[dim][0]
                infos.add((entry.algId, entry.comment))
                data = generateData(entry, target)
                if data[2] == 0: # No success
                    dimmaxevals.append(dim)
                    maxevals.append(float(data[3])/dim)
                else:
                    dimmedian.append(dim)
                    medianfes.append(data[4]/dim)
                    dimert.append(dim)
                    ert.append(float(data[0])/dim)
                    if data[1] < 1.:
                        dimnbsucc.append(dim)
                        ynbsucc.append(float(data[0])/dim)
                        nbsucc.append('%d' % data[2])

            if not infos: # empty for any reason...
                continue
            infos = infos.pop()
            # Draw lines
            tmp = plt.plot(dimert, ert, marker='.', markersize=30,
                          **algPlotInfos[infos])
            plt.setp(tmp[0], markeredgecolor=plt.getp(tmp[0], 'color'))
            if dimmaxevals:
                tmp = plt.plot(dimmaxevals, maxevals, **algPlotInfos[infos])
                plt.setp(tmp[0],  marker='x', markersize=20,
                         markeredgecolor=plt.getp(tmp[0], 'color'))
            handles.append(tmp)
            #tmp2 = plt.plot(dimmedian, medianfes, ls='', marker='+',
            #               markersize=30, markeredgewidth=5,
            #               markeredgecolor=plt.getp(tmp, 'color'))[0]
            #for i, n in enumerate(nbsucc):
            #    plt.text(dimnbsucc[i], numpy.array(ynbsucc[i])*1.85, n,
            #             verticalalignment='bottom',
            #             horizontalalignment='center')

        if not bestalg.bestalgentries2009:
            bestalg.loadBBOB2009()

        bestalgdata = []
        dimbestalg = list(df[0] for df in bestalg.bestalgentries2009 if df[1] == f)
        dimbestalg.sort()
        dimbestalg2 = []
        for d in dimbestalg:
            entry = bestalg.bestalgentries2009[(d, f)]
            tmp = entry.detERT([target])[0]
            if numpy.isfinite(tmp):
                bestalgdata.append(float(tmp)/d)
                dimbestalg2.append(d)

        tmp = plt.plot(dimbestalg2, bestalgdata, color='wheat', linewidth=10,
                       marker='d', markersize=25, markeredgecolor='wheat',
                       label='best 2009', zorder=-1)
        handles.append(tmp)

        if isBenchmarkinfosFound:
            title = funInfos[f]
            plt.gca().set_title(title)

        beautify(rightlegend=True)

        plotLegend(handles)

        saveFigure(filename, figFormat=figformat, verbose=verbose)

        plt.close()

def main2(dictAlg, sortedAlgs, target, outputdir, verbose=True):
    """From a DataSetList, returns figures showing the scaling: ERT/dim vs dim.
    
    Differ from main method by the line style policy: in the case of main2
    the variable styles (defined in header) is used instead of
    bbob_pproc.dataoutput.algPlotInfos.
    One function and one target per figure.
    
    """

    dictFunc = pproc.dictAlgByFun(dictAlg)

    for f in dictFunc:
        filename = os.path.join(outputdir,'ppfigs_f%03d' % (f))
        handles = []
        for i, alg in enumerate(sortedAlgs):
            dictDim = dictFunc[f][alg].dictByDim()

            #Collect data
            dimert = []
            ert = []
            dimnbsucc = []
            ynbsucc = []
            nbsucc = []
            dimmaxevals = []
            maxevals = []
            dimmedian = []
            medianfes = []
            for dim in sorted(dictDim):
                entry = dictDim[dim][0]
                data = generateData(entry, target)
                if data[2] == 0: # No success
                    dimmaxevals.append(dim)
                    maxevals.append(float(data[3])/dim)
                else:
                    dimmedian.append(dim)
                    medianfes.append(data[4]/dim)
                    dimert.append(dim)
                    ert.append(float(data[0])/dim)
                    if data[1] < 1.:
                        dimnbsucc.append(dim)
                        ynbsucc.append(float(data[0])/dim)
                        nbsucc.append('%d' % data[2])

            # Draw lines
            tmp = plt.plot(dimert, ert, markersize=30, **styles[i])
                           #label=alg, )
            plt.setp(tmp[0], markeredgecolor=plt.getp(tmp[0], 'color'))
            # For legend
            tmp = plt.plot([], [], markersize=12., label=alg, **styles[i])
            plt.setp(tmp[0], markeredgecolor=plt.getp(tmp[0], 'color'))

            if dimmaxevals:
                tmp = plt.plot(dimmaxevals, maxevals, **styles[i])
                plt.setp(tmp[0], markersize=20, #label=alg,
                         markeredgecolor=plt.getp(tmp[0], 'color'),
                         markerfacecolor='None')

            handles.append(tmp)
            #tmp2 = plt.plot(dimmedian, medianfes, ls='', marker='+',
            #               markersize=30, markeredgewidth=5,
            #               markeredgecolor=plt.getp(tmp, 'color'))[0]
            #for i, n in enumerate(nbsucc):
            #    plt.text(dimnbsucc[i], numpy.array(ynbsucc[i])*1.85, n,
            #             verticalalignment='bottom',
            #             horizontalalignment='center')

        if not bestalg.bestalgentries2009:
            bestalg.loadBBOB2009()

        bestalgdata = []
        dimbestalg = list(df[0] for df in bestalg.bestalgentries2009 if df[1] == f)
        dimbestalg.sort()
        dimbestalg2 = []
        for d in dimbestalg:
            entry = bestalg.bestalgentries2009[(d, f)]
            tmp = entry.detERT([target])[0]
            if numpy.isfinite(tmp):
                bestalgdata.append(float(tmp)/d)
                dimbestalg2.append(d)

        tmp = plt.plot(dimbestalg2, bestalgdata, color='wheat', linewidth=10,
                       marker='d', markersize=25, markeredgecolor='wheat', zorder=-1)
                       #label='best 2009', 
        handles.append(tmp)

        if isBenchmarkinfosFound:
            title = funInfos[f]
            plt.gca().set_title(title)

        beautify(rightlegend=legend)

        if legend:
            plotLegend(handles)
        else:
            if f in (1, 24, 101, 130):
                plt.legend()

        saveFigure(filename, figFormat=figformat, verbose=verbose)

        plt.close()

