#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates aRTs and convergence figures for multiple algorithms."""
from __future__ import absolute_import
import os
import matplotlib.pyplot as plt
import numpy
import warnings
from pdb import set_trace
from .. import toolsdivers, toolsstats, bestalg, pproc, genericsettings, htmldesc, ppfigparam
from .. import testbedsettings
from ..ppfig import saveFigure
from ..pptex import color_to_latex, marker_to_latex, marker_to_html, writeLabels

# styles = [{'color': 'k', 'marker': 'o', 'markeredgecolor': 'k'},
#           {'color': 'b'},
#           {'color': 'c', 'marker': 'v', 'markeredgecolor': 'c'},
#           {'color': 'g'},
#           {'color': 'y', 'marker': '^', 'markeredgecolor': 'y'},
#           {'color': 'm'},
#           {'color': 'r', 'marker': 's', 'markeredgecolor': 'r'}] # sort of rainbow style

show_significance = 0.01  # for zero nothing is shown

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


def prepare_scaling_figure_caption():

    scaling_figure_caption_start_fixed = (r"""Average running time (\aRT\ in number of $f$-evaluations
                    as $\log_{10}$ value), divided by dimension for target function value $BBOBPPFIGSFTARGET$
                    versus dimension. Slanted grid lines indicate quadratic scaling with the dimension. """
                                          )

    scaling_figure_caption_start_rlbased = (r"""Average running time (\aRT\ in number of $f$-evaluations
                        as $\log_{10}$ value) divided by dimension versus dimension. The target function value
                        is chosen such that the REFERENCE_ALGORITHM artificial algorithm just failed to achieve
                        an \aRT\ of $BBOBPPFIGSFTARGET\times\DIM$. """
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

    scaling_figure_caption_fixed = scaling_figure_caption_start_fixed + scaling_figure_caption_end
    scaling_figure_caption_rlbased = scaling_figure_caption_start_rlbased + scaling_figure_caption_end

    if testbedsettings.current_testbed.name == testbedsettings.testbed_name_bi:
        # NOTE: no runlength-based targets supported yet
        figure_caption = scaling_figure_caption_fixed
    elif testbedsettings.current_testbed.name == testbedsettings.testbed_name_single or \
         testbedsettings.current_testbed.name == testbedsettings.testbed_name_cons:
        if genericsettings.runlength_based_targets:
            figure_caption = scaling_figure_caption_rlbased
        else:
            figure_caption = scaling_figure_caption_fixed
    else:
        warnings.warn("Current settings do not support ppfigdim caption.")

    return figure_caption


def scaling_figure_caption(for_html = False):

    if for_html:
        figure_caption = htmldesc.getValue('##bbobppfigslegend' +
                                           testbedsettings.current_testbed.scenario + '##')
    else:
        figure_caption = prepare_scaling_figure_caption()

    target = testbedsettings.current_testbed.ppfigs_ftarget
    target = pproc.TargetValues.cast([target] if numpy.isscalar(target) else target)
    assert len(target) == 1

    if genericsettings.runlength_based_targets:
        figure_caption = figure_caption.replace('REFERENCE_ALGORITHM',
                                                target.reference_algorithm)
        figure_caption = figure_caption.replace('REFERENCEALGORITHM',
                                                target.reference_algorithm)

    figure_caption = figure_caption.replace('BBOBPPFIGSFTARGET',
                                            toolsdivers.number_to_latex(target.label(0)))

    return figure_caption


def prepare_ecdfs_figure_caption():

    best2009text = (
                r"The ``best 2009'' line " +
                r"corresponds to the best \aRT\ observed during BBOB 2009 " +
                r"for each selected target."
                )
    ecdfs_figure_caption_standard = (
                r"Bootstrapped empirical cumulative distribution of the number " +
                r"of objective function evaluations divided by dimension " +
                r"(FEvals/DIM) for $BBOBPPFIGSFTARGET$ " +
                r"targets with target precision in BBOBPPFIGSTARGETRANGE " +
                r"for all functions and subgroups in #1-D. "
                )
    ecdfs_figure_caption_rlbased = (
                r"Bootstrapped empirical cumulative distribution of the number " +
                r"of objective function evaluations divided by dimension " +
                r"(FEvals/DIM) for all functions and subgroups in #1-D." +
                r" The targets are chosen from BBOBPPFIGSTARGETRANGE " +
                r"such that the REFERENCE_ALGORITHM artificial algorithm just " +
                r"not reached them within a given budget of $k$ $\times$ DIM, " +
                r"with $k\in \{0.5, 1.2, 3, 10, 50\}$. "
                )

    if testbedsettings.current_testbed.name == testbedsettings.testbed_name_bi:
        # NOTE: no runlength-based targets supported yet
        figure_caption = ecdfs_figure_caption_standard
    elif testbedsettings.current_testbed.name == testbedsettings.testbed_name_single or \
         testbedsettings.current_testbed.name == testbedsettings.testbed_name_cons:
        if genericsettings.runlength_based_targets:
            figure_caption = ecdfs_figure_caption_rlbased + best2009text
        else:
            figure_caption = ecdfs_figure_caption_standard + best2009text
    else:
        warnings.warn("Current settings do not support ppfigdim caption.")

    return figure_caption


def ecdfs_figure_caption(for_html = False, dimension = 0):

    if for_html:
        key = '##bbobECDFslegend%s%d##' % (testbedsettings.current_testbed.scenario, dimension)
        caption = htmldesc.getValue(key)
    else:
        caption = prepare_ecdfs_figure_caption()

    target = testbedsettings.current_testbed.ppfigs_ftarget
    target = pproc.TargetValues.cast([target] if numpy.isscalar(target) else target)
    assert len(target) == 1

    caption = caption.replace('BBOBPPFIGSTARGETRANGE',
                              str(testbedsettings.current_testbed.pprldmany_target_range_latex))

    if genericsettings.runlength_based_targets:
        caption = caption.replace('REFERENCE_ALGORITHM', target.reference_algorithm)
        caption = caption.replace('REFERENCEALGORITHM', target.reference_algorithm)
    else:
        caption = caption.replace('BBOBPPFIGSFTARGET',
                                  str(len(testbedsettings.current_testbed.pprldmany_target_values)))

    return caption


def get_ecdfs_single_fcts_caption():
    ''' For the moment, only the bi-objective case is covered! '''
    s = (r"""Empirical cumulative distribution of simulated (bootstrapped) runtimes in number
         of objective function evaluations divided by dimension (FEvals/DIM) for the $""" +
            str(len(testbedsettings.current_testbed.pprldmany_target_values)) +
            r"$ targets " + 
            str(testbedsettings.current_testbed.pprldmany_target_range_latex) +
            r" for functions $f_1$ to $f_{16}$ and all dimensions. "
            )
    return s

def get_ecdfs_all_groups_caption():
    ''' For the moment, only the bi-objective case is covered! '''
#    s = (r"Bootstrapped empirical cumulative distribution of the number " +
#            r"of objective function evaluations divided by dimension " +
#            r"(FEvals/DIM) for " +
    s = (r"""Empirical cumulative distribution of simulated (bootstrapped) runtimes, measured in number
         of objective function evaluations, divided by dimension (FEvals/DIM) for the $""" +
            str(len(testbedsettings.current_testbed.pprldmany_target_values)) +
            r"$ targets " + 
            str(testbedsettings.current_testbed.pprldmany_target_range_latex) +
            r" for all function groups and all dimensions. The aggregation" +
            r" over all 55 functions is shown in the last plot."
            )
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

    # quadratic slanted "grid"
    if 1 < 3:
        for i in xrange(-2, 7, 1 if ymax < 1e5 else 2):
            plt.plot((0.2, 20000), (10**i, 10**(i + 5)), 'k:',
                     linewidth=0.5)  # grid should be on top
    else:  # to be removed
        plt.plot((2,200), (1, 1e2), 'k:', zorder=-1)  # -1 -> plotted below?
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
        toolsdivers.legend(loc=0, numpoints=1)

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

def main(dictAlg, htmlFilePrefix, isBiobjective, sortedAlgs=None, outputdir='ppdata', verbose=True):
    """From a DataSetList, returns figures showing the scaling: aRT/dim vs dim.
    
    One function and one target per figure.
    
    ``target`` can be a scalar, a list with one element or a 
    ``pproc.TargetValues`` instance with one target.
    
    ``sortedAlgs`` is a list of string-identifies (folder names)
    
    """
    # target becomes a TargetValues "list" with one element
    target = testbedsettings.current_testbed.ppfigs_ftarget
    target = pproc.TargetValues.cast([target] if numpy.isscalar(target) else target)
    latex_commands_filename = os.path.join(outputdir, 'bbob_pproc_commands.tex')
    assert isinstance(target, pproc.TargetValues) 
    if len(target) != 1:
        raise ValueError('only a single target can be managed in ppfigs, ' + str(len(target)) + ' targets were given')
    
    funInfos = ppfigparam.read_fun_infos()    

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
            algorithmName = toolsdivers.str_to_latex(toolsdivers.strip_pathname1(alg))
            tmp = plt.plot([], [], label = algorithmName, **styles[i])
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

        bestalgentries = bestalg.load_best_algorithm()

        if bestalgentries:        
            bestalgdata = []
            dimbestalg = list(df[0] for df in bestalgentries if df[1] == f)
            dimbestalg.sort()
            dimbestalg2 = []
            for d in dimbestalg:
                entry = bestalgentries[(d, f)]
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
        
        fontSize = genericsettings.getFontSize(funInfos.values())
        if f in funInfos.keys():
            plt.gca().set_title(funInfos[f], fontsize=fontSize)

        functions_with_legend = testbedsettings.current_testbed.functions_with_legend
        isLegend = False
        if legend:
            plotLegend(handles)
        elif 1 < 3:
            if f in functions_with_legend and len(sortedAlgs) < 6: # 6 elements at most in the boxed legend
                isLegend = True

        beautify(legend=isLegend, rightlegend=legend)

        # bottom labels with #instances and type of targets:
        infotext = ''
        algorithms_with_data = [a for a in dictAlg.keys() if dictAlg[a] != []]
        for alg in algorithms_with_data:
            infotext += '%d, ' % len((dictFunc[f][alg])[0].instancenumbers)
        infotext = infotext.rstrip(', ')
        infotext += ' instances'
        plt.text(plt.xlim()[0], plt.ylim()[0]+0.5, infotext, fontsize=14)  # TODO: check
        plt.text(plt.xlim()[0], plt.ylim()[0], 
                 'target ' + target.label_name() + ': ' + target.label(0),
                 fontsize=14)  # TODO: check

        saveFigure(filename, verbose=verbose)

        plt.close()

    
    htmlFile = os.path.join(outputdir, htmlFilePrefix + '.html')
    # generate commands in tex file:
    try:
        abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        alg_definitions = []
        alg_definitions_html = ''
        for i in range(len(sortedAlgs)):
            symb = r'{%s%s}' % (color_to_latex(styles[i]['color']),
                                marker_to_latex(styles[i]['marker']))
            symb_html = '<span style="color:%s;">%s</span>' % (styles[i]['color'], marker_to_html(styles[i]['marker']))
            
            alg_definitions.append((', ' if i > 0 else '') + '%s: %s' % (symb, '\\algorithm' + abc[i % len(abc)]))
            alg_definitions_html += (', ' if i > 0 else '') + '%s: %s' % (symb_html, toolsdivers.str_to_latex(toolsdivers.strip_pathname1(sortedAlgs[i])))
        toolsdivers.prepend_to_file(latex_commands_filename, 
                [#'\\providecommand{\\bbobppfigsftarget}{\\ensuremath{10^{%s}}}' 
                 #       % target.loglabel(0), # int(numpy.round(numpy.log10(target))),
                '\\providecommand{\\bbobppfigslegend}[1]{',
                scaling_figure_caption(),
                'Legend: '] + alg_definitions + ['}']
                )
        toolsdivers.prepend_to_file(latex_commands_filename, 
                ['\\providecommand{\\bbobECDFslegend}[1]{',
                ecdfs_figure_caption(), '}']
                )

        toolsdivers.replace_in_file(htmlFile, '##bbobppfigslegend##', scaling_figure_caption(True) + 'Legend: ' + alg_definitions_html)
        toolsdivers.replace_in_file(htmlFile, '##bbobECDFslegend5##', ecdfs_figure_caption(True, 5))
        toolsdivers.replace_in_file(htmlFile, '##bbobECDFslegend20##', ecdfs_figure_caption(True, 20))

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

        if f in funInfos.keys():
            plt.gca().set_title(funInfos[f], fontsize=fontSize)

        beautify(rightlegend=legend)

        if legend:
            plotLegend(handles)
        else:
            if f in functions_with_legend:
                toolsdivers.legend()

        saveFigure(filename, figFormat=genericsettings.getFigFormats(), verbose=verbose)

        plt.close()

