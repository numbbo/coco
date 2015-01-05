#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates ERTs and convergence figures for multiple algorithms."""
from __future__ import absolute_import
import os
import matplotlib.pyplot as plt
import numpy
from pdb import set_trace
from .. import toolsdivers, toolsstats, bestalg, pproc, genericsettings
from ..ppfig import saveFigure
from ..pptex import color_to_latex, marker_to_latex, writeLabels

# styles = [{'color': 'k', 'marker': 'o', 'markeredgecolor': 'k'},
#           {'color': 'b'},
#           {'color': 'c', 'marker': 'v', 'markeredgecolor': 'c'},
#           {'color': 'g'},
#           {'color': 'y', 'marker': '^', 'markeredgecolor': 'y'},
#           {'color': 'm'},
#           {'color': 'r', 'marker': 's', 'markeredgecolor': 'r'}] # sort of rainbow style

ftarget_default = 1e-8
show_significance = 0.01  # for zero nothing is shown
scaling_figure_caption_start_fixed = (r"""Expected running time (\ERT\ in number of $f$-evaluations 
                as $\log_{10}$ value), divided by dimension for target function value $BBOBPPFIGSFTARGET$ 
                versus dimension. Slanted grid lines indicate quadratic scaling with the dimension. """
                )
scaling_figure_caption_start_rlbased = (r"""Expected running time (\ERT\ in number of $f$-evaluations 
                as $\log_{10}$ value) divided by dimension versus dimension. The target function value 
                is chosen such that the REFERENCE_ALGORITHM artificial algorithm just failed to achieve 
                an \ERT\ of $BBOBPPFIGSFTARGET\times\DIM$. """
                )
scaling_figure_caption_end = (
                r"Different symbols " +
                r"correspond to different algorithms given in the legend of #1. " +
                r"Light symbols give the maximum number of function evaluations from the longest trial " + 
                r"divided by dimension. " +
                (r"Black stars indicate a statistically better result compared to all other algorithms " +
                 r"with $p<0.01$ and Bonferroni correction number of dimensions (six).  ") 
                         if show_significance else ''
                )

ecdfs_figure_caption_standard = (
                r"Bootstrapped empirical cumulative distribution of the number " +
                r"of objective function evaluations divided by dimension " +
                r"(FEvals/DIM) for 50 targets in $10^{[-8..2]}$ for all "+
                r"functions and subgroups in #1-D. The ``best 2009'' line "+
                r"corresponds to the best \ERT\ observed during BBOB 2009 " +
                r"for each single target."
                )

ecdfs_figure_caption_rlbased = (
                r"Bootstrapped empirical cumulative distribution of the number " +
                r"of objective function evaluations divided by dimension " +
                r"(FEvals/DIM) for all functions and subgroups in #1-D." +
                r" The targets are chosen from $10^{[-8..2]}$ " +
                r"such that the REFERENCE_ALGORITHM artificial algorithm just " +
                r"not reached them within a given budget of $k$ $\times$ DIM, " +
                r"with $k\in \{0.5, 1.2, 3, 10, 50\}$. " +
                r"The ``best 2009'' line " +
                r"corresponds to the best \ERT\ observed during BBOB 2009 " +
                r"for each selected target."
                )


styles = genericsettings.line_styles
def fix_styles(number, styles=styles):
    """a short hack to fix length of styles"""
    m = len(styles) 
    while len(styles) < number:
        styles.append(styles[len(styles) % m])
    for i in xrange(len(styles)):
        styles[i].update({'linewidth': 5 - min([2, i/3.0]),  # thinner lines over thicker lines
                          'markeredgewidth': 6 - min([2, i / 2.0]),
                          'markerfacecolor': 'None'})
refcolor = 'wheat'

show_algorithms = []
fontsize = 10.0
legend = False

#Get benchmark short infos.
infofile = os.path.join(os.path.split(__file__)[0], '..', 'benchmarkshortinfos.txt')
try:
    funInfos = {}
    f = open(infofile,'r')
    for line in f:
        if len(line) == 0 or line.startswith('%') or line.isspace() :
            continue
        funcId, funcInfo = line[0:-1].split(None,1)
        funInfos[int(funcId)] = funcId + ' ' + funcInfo
    f.close()
except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    print 'Could not find file', infofile, \
          'Titles in figures will not be displayed.'
          
def scaling_figure_caption(target):
    # need to be used in rungenericmany.py!?
    assert len(target) == 1
    if isinstance(target, pproc.RunlengthBasedTargetValues):
        s = scaling_figure_caption_start_rlbased.replace('BBOBPPFIGSFTARGET', 
                                                         toolsdivers.number_to_latex(target.label(0)))
        s = s.replace('REFERENCE_ALGORITHM', target.reference_algorithm)
    else:
        s = scaling_figure_caption_start_fixed.replace('BBOBPPFIGSFTARGET', 
                                                       toolsdivers.number_to_latex(target.label(0)))
    s += scaling_figure_caption_end
    return s

def ecdfs_figure_caption(target):
    assert len(target) == 1
    if isinstance(target, pproc.RunlengthBasedTargetValues):
        s = ecdfs_figure_caption_rlbased.replace('REFERENCE_ALGORITHM', 
                                                         target.reference_algorithm)
    else:
        s = ecdfs_figure_caption_standard
    return s


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

def beautify(legend=False, rightlegend=False):
    """Customize figure format.

    adding a legend, axis label, etc

    :param bool legend: if True, display a box legend
    :param bool rightlegend: if True, makes some space on the right for
                             legend

    """
    # Get axis handle and set scale for each axis
    axisHandle = plt.gca()
    axisHandle.set_xscale("log")
    try:
        axisHandle.set_yscale("log")
    except OverflowError:
        set_trace()

    # Grid options
    axisHandle.yaxis.grid(True)

    ymin, ymax = plt.ylim()

    # quadratic "grid"
    plt.plot((2,200), (1, 1e2), 'k:', zorder=-1)
    # plt.plot((2,200), (1, 1e4), 'k:', zorder=-1)
    plt.plot((2,200), (1e3, 1e5), 'k:', zorder=-1)
    # plt.plot((2,200), (1e3, 1e7), 'k:', zorder=-1)
    plt.plot((2,200), (1e6, 1e8), 'k:', zorder=-1)
    # plt.plot((2,200), (1e6, 1e10), 'k:', zorder=-1)

    plt.ylim(ymin=10**-0.2, ymax=ymax) # Set back the default maximum.

    # ticks on axes
    #axisHandle.invert_xaxis()
    dimticklist = (2, 3, 5, 10, 20, 40)  # TODO: should become input arg at some point? 
    dimannlist = (2, 3, 5, 10, 20, 40)  # TODO: should become input arg at some point? 
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

    if legend:
        plt.legend(loc=0, numpoints=1)

def generateData(dataSet, target):
    """Returns an array of results to be plotted.

    Oth column is ert, 1st is the success rate, 2nd the number of
    successes, 3rd the mean of the number of function evaluations, and
    4th the median of number of function evaluations of successful runs
    or numpy.nan.

    """
    res = []

    data = dataSet.detEvals([target])[0]
    succ = (numpy.isnan(data) == False)
    data[numpy.isnan(data)] = dataSet.maxevals[numpy.isnan(data)]
    res.extend(toolsstats.sp(data, issuccessful=succ, allowinf=False))
    res.append(numpy.mean(data))
    if res[2] > 0:
        res.append(toolsstats.prctile(data[succ], 50)[0])
    else:
        res.append(numpy.nan)
    res[3] = numpy.max(dataSet.maxevals)
    return res

def main(dictAlg, sortedAlgs=None, target=ftarget_default, outputdir='ppdata', verbose=True):
    """From a DataSetList, returns figures showing the scaling: ERT/dim vs dim.
    
    One function and one target per figure.
    
    ``target`` can be a scalar, a list with one element or a 
    ``pproc.TargetValues`` instance with one target.
    
    ``sortedAlgs`` is a list of string-identifies (folder names)
    
    """
    # target becomes a TargetValues "list" with one element
    target = pproc.TargetValues.cast([target] if numpy.isscalar(target) else target)
    latex_commands_filename = os.path.join(outputdir, 'bbob_pproc_commands.tex')
    assert isinstance(target, pproc.TargetValues) 
    if len(target) != 1:
        raise ValueError('only a single target can be managed in ppfigs, ' + str(len(target)) + ' targets were given')
    
    dictFunc = pproc.dictAlgByFun(dictAlg)
    if sortedAlgs is None:
        sortedAlgs = sorted(dictAlg.keys())
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    for f in dictFunc:
        filename = os.path.join(outputdir,'ppfigs_f%03d' % (f))
        handles = []
        fix_styles(len(sortedAlgs))  # 
        for i, alg in enumerate(sortedAlgs):
            dictDim = dictFunc[f][alg].dictByDim()  # this does not look like the most obvious solution

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
                assert len(dictDim[dim]) == 1
                entry = dictDim[dim][0]
                data = generateData(entry, target((f, dim))[0]) # TODO: here we might want a different target for each function
                if 1 < 3 or data[2] == 0: # No success
                    dimmaxevals.append(dim)
                    maxevals.append(float(data[3])/dim)
                if data[2] > 0:
                    dimmedian.append(dim)
                    medianfes.append(data[4]/dim)
                    dimert.append(dim)
                    ert.append(float(data[0])/dim)
                    if data[1] < 1.:
                        dimnbsucc.append(dim)
                        ynbsucc.append(float(data[0])/dim)
                        nbsucc.append('%d' % data[2])

            # Draw lines
            if 1 < 3:  # new version
                # omit the line if a point in between is missing
                for idim in range(len(dimert)):
                    # plot line only if next dim < 2.1*dim (a hack)
                    if idim < len(dimert) - 1 and dimert[idim + 1] < 2.1 * dimert[idim]:
                        tmp = plt.plot(dimert[idim:idim+2], ert[idim:idim+2], **styles[i]) #label=alg, )
                    else:  # plot remaining single points (some twice)
                        tmp = plt.plot(dimert[idim], ert[idim], **styles[i]) #label=alg, )
                    plt.setp(tmp[0], markeredgecolor=plt.getp(tmp[0], 'color'))
            else:  # to be removed
                tmp = plt.plot(dimert, ert, **styles[i]) #label=alg, )
                plt.setp(tmp[0], markeredgecolor=plt.getp(tmp[0], 'color'))

            # For legend
            # tmp = plt.plot([], [], label=alg.replace('..' + os.sep, '').strip(os.sep), **styles[i])
            tmp = plt.plot([], [], label=alg.split(os.sep)[-1], **styles[i])
            plt.setp(tmp[0], markersize=12.,
                     markeredgecolor=plt.getp(tmp[0], 'color'))

            if dimmaxevals:
                tmp = plt.plot(dimmaxevals, maxevals, **styles[i])
                plt.setp(tmp[0], markersize=20, #label=alg,
                         markeredgecolor=plt.getp(tmp[0], 'color'),
                         markeredgewidth=1, 
                         markerfacecolor='None', linestyle='None')
                
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
            tmp = entry.detERT(target((f, d)))[0]
            if numpy.isfinite(tmp):
                bestalgdata.append(float(tmp)/d)
                dimbestalg2.append(d)

        tmp = plt.plot(dimbestalg2, bestalgdata, color=refcolor, linewidth=10,
                       marker='d', markersize=25, markeredgecolor=refcolor, zorder=-1
                       #label='best 2009', 
                       )
        handles.append(tmp)
        
        if show_significance: # plot significance-stars
            xstar, ystar = [], []
            dims = sorted(pproc.dictAlgByDim(dictFunc[f]))
            for i, dim in enumerate(dims):
                datasets = pproc.dictAlgByDim(dictFunc[f])[dim]
                assert all([len(datasets[ialg]) == 1 for ialg in sortedAlgs if datasets[ialg]])
                dsetlist =  [datasets[ialg][0] for ialg in sortedAlgs if datasets[ialg]]
                if len(dsetlist) > 1:
                    arzp, arialg = toolsstats.significance_all_best_vs_other(dsetlist, target((f, dim)))
                    if arzp[0][1] * len(dims) < show_significance:
                        ert = dsetlist[arialg[0]].detERT(target((f, dim)))[0]
                        if ert < numpy.inf: 
                            xstar.append(dim)
                            ystar.append(ert/dim)

            plt.plot(xstar, ystar, 'k*', markerfacecolor=None, markeredgewidth=2, markersize=0.5*styles[0]['markersize'])
        if funInfos:
            plt.gca().set_title(funInfos[f])

        isLegend = False
        if legend:
            plotLegend(handles)
        elif 1 < 3:
            if f in (1, 24, 101, 130) and len(sortedAlgs) < 6: # 6 elements at most in the boxed legend
                isLegend = True

        beautify(legend=isLegend, rightlegend=legend)

        plt.text(plt.xlim()[0], plt.ylim()[0], 'target ' + target.label_name() + ': ' + target.label(0))  # TODO: check

        saveFigure(filename, verbose=verbose)

        plt.close()

    # generate commands in tex file:
    try:
        abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        alg_definitions = []
        for i in range(len(sortedAlgs)):
            symb = r'{%s%s}' % (color_to_latex(styles[i]['color']),
                                marker_to_latex(styles[i]['marker']))
            alg_definitions.append((', ' if i > 0 else '') + '%s:%s' % (symb, '\\algorithm' + abc[i % len(abc)]))
        toolsdivers.prepend_to_file(latex_commands_filename, 
                [#'\\providecommand{\\bbobppfigsftarget}{\\ensuremath{10^{%s}}}' 
                 #       % target.loglabel(0), # int(numpy.round(numpy.log10(target))),
                '\\providecommand{\\bbobppfigslegend}[1]{',
                scaling_figure_caption(target), 
                'Legend: '] + alg_definitions + ['}']
                )
        toolsdivers.prepend_to_file(latex_commands_filename, 
                ['\\providecommand{\\bbobECDFslegend}[1]{',
                ecdfs_figure_caption(target), '}']
                )


        if verbose:
            print 'Wrote commands and legend to %s' % filename

        # this is obsolete (however check templates)
        filename = os.path.join(outputdir,'ppfigs.tex') 
        f = open(filename, 'w')
        f.write('% Do not modify this file: calls to post-processing software'
                + ' will overwrite any modification.\n')
        f.write('Legend: ')
        
        for i in range(0, len(sortedAlgs)):
            symb = r'{%s%s}' % (color_to_latex(styles[i]['color']),
                                marker_to_latex(styles[i]['marker']))
            f.write((', ' if i > 0 else '') + '%s:%s' % (symb, writeLabels(sortedAlgs[i])))
        f.close()    
        if verbose:
            print '(obsolete) Wrote legend in %s' % filename
    except IOError:
        raise


        handles.append(tmp)

        if funInfos:
            plt.gca().set_title(funInfos[f])

        beautify(rightlegend=legend)

        if legend:
            plotLegend(handles)
        else:
            if f in (1, 24, 101, 130):
                plt.legend()

        saveFigure(filename, figFormat=genericsettings.fig_formats, verbose=verbose)

        plt.close()

