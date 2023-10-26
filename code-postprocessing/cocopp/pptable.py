#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for generating tables used by rungeneric1.py.

The generated tables give the ERT and in brackets the 10th to 90th
percentile range divided by two of 100 simulated runs divided by the
ERT of a reference algorithm (given in the respective first row and as
indicated in testbedsettings.py)
for different target precisions for different functions. If the reference
algorithm did not reach the target precision, the absolute values are
given.

The median number of conducted function evaluations is given in
*italics*, if no run reached 1e-7.
#succ is the number of trials that reached the target precision 1e-8
**Bold** entries are statistically significantly better (according to
the rank-sum test) compared to the given reference algorithm, with
p = 0.05 or p = 1e-k where k > 1 is the number following the
\downarrow symbol, with Bonferroni correction by the number of
functions.

"""
from __future__ import absolute_import, print_function

import os
import warnings
import numpy as np
from . import genericsettings, bestalg, toolsstats, pproc
from . import testbedsettings
from .pptex import tableLaTeX, writeFEvals2, writeFEvalsMaxPrec
from .toolsstats import significancetest
from .toolsdivers import prepend_to_file
from . import captions

# def tablespec(targets):
# 
#     i = 0
#     tspec = {'col%d' % i: {'what': 'fname', 'header': r'$\Delta f$', 'format': None}}
#     for t in targets:
#         i =  i + 1
#         tspec.update({'col%d' % i: {'what': 'ERT ratio for df=%e' % t,
#                                     'header': r'\multicolumn{2}{@{}c@{}}{1e%+d}' % (int(np.log10(t)),
#                                     'format': writeFEval}})
#     i = i + 1
#     tspec.update({'col%d' %  i: {'what': 'nb of success', 'header': r'\#succ',
#                                  'format': '%d'}})


def get_table_caption():
    """ Sets table caption, based on the testbedsettings.current_testbed
        and genericsettings.runlength_based_targets.
    """

    table_caption_start = (r"""%
        Expected runtime (\ERT\ in number of """
        + testbedsettings.current_testbed.string_evals
        + r""") divided by the \ERT\ of !!THE-REF-ALG!! in #1. This \ERT\
        ratio and, in braces as dispersion measure, the half difference between 90 and
        10\%-tile of !!BOOTSTRAPPED!! run lengths appear in the second row of each cell,  
        the best \ERT\
        """)
    table_caption_rlbased = (r"""%
        in the first. The different target """
        + ("!!DF!!-" if not testbedsettings.current_testbed.has_constraints else "precision ")
        + r"""values are shown in the top row. 
        \#succ is the number of trials that reached the (final) target $!!FOPT!! 
        + !!HARDEST-TARGET-LATEX!!$.
        """)
    table_caption_fixedtargets = (r"""%
        (preceded by the target """
        + ("!!DF!!-" if not testbedsettings.current_testbed.has_constraints else "precision ")
        + r"""value in \textit{italics}) in the first. 
        \#succ is the number of trials that reached the target value of the last column.
        """)
    table_caption_rest = r"""%
        The median number of conducted evaluations is additionally given in 
        \textit{italics}, if the target in the last column was never reached. 
        \textbf{Bold} entries are statistically significantly better (according to
        the rank-sum test) compared to !!THE-REF-ALG!!, with
        $p = 0.05$ or $p = 10^{-k}$ when the number $k > 1$ is following the
        $\downarrow$ symbol, with Bonferroni correction by the number of
        functions (!!TOTAL-NUM-OF-FUNCTIONS!!).
        """
    table_caption_no_reference_algorithm = (r"""%
        Expected runtime (\ERT) to reach given targets, measured
        in number of """
        + testbedsettings.current_testbed.string_evals
        + r""" in #1. For each function, the \ERT\ 
        and, in braces as dispersion measure, the half difference between 10 and 
        90\%-tile of !!BOOTSTRAPPED!! runtimes is shown for the different
        target """
        + ("!!DF!!-" if not testbedsettings.current_testbed.has_constraints else "precision ")
        + r"""values as shown in the top row. 
        \#succ is the number of trials that reached the last target 
        $!!FOPT!! + !!HARDEST-TARGET-LATEX!!$.
        The median number of conducted evaluations is additionally given in 
        \textit{italics}, if the target in the last column was never reached. 
        """)

    table_caption = None
    if (testbedsettings.current_testbed.reference_algorithm_filename == '' or
            testbedsettings.current_testbed.reference_algorithm_filename is None):
        # all testbeds without provided reference algorithm
        table_caption = table_caption_no_reference_algorithm
    else:
        if genericsettings.runlength_based_targets:
            table_caption = table_caption_start + table_caption_rlbased + table_caption_rest
        else:
            table_caption = table_caption_start + table_caption_fixedtargets + table_caption_rest

    return captions.replace(table_caption)


def main(dsList, dims_of_interest, outputdir, latex_commands_file):
    """Generate a table of ratio ERT/ERTref vs target precision.

    1 table per dimension will be generated.

    Rank-sum tests table on "Final Data Points" for only one algorithm.
    that is, for example, using 1/#fevals(ftarget) if ftarget was
    reached and -f_final otherwise as input for the rank-sum test, where
    obviously the larger the better.

    """
    # TODO: check that it works for any reference algorithm?
    # in the following the reference algorithm is the one given in
    # bestalg.bestalgentries which is the virtual best of BBOB
    dictDim = dsList.dictByDim()
    testbed = testbedsettings.current_testbed

    targetf = testbed.pptable_ftarget
    targetsOfInterest = testbed.pptable_targetsOfInterest

    refalgentries = bestalg.load_reference_algorithm(testbed.reference_algorithm_filename)

    if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
        header = [r'\#FEs/D']
        header_html = ['<thead>\n<tr>\n<th>#FEs/D</th>\n']
        for i in targetsOfInterest.labels():
            header.append(r'\multicolumn{2}{c}{%s}' % i)
            header_html.append('<td>%s</td>\n' % i)

    else:
        header = [r'$\Delta f$']
        header_html = ['<thead>\n<tr>\n<th>&#916; f</th>\n']
        for i in targetsOfInterest.target_values:
            header.append(r'\multicolumn{2}{c}{1e%+d}' % (int(np.log10(i))))
            header_html.append('<td>1e%+d</td>\n' % (int(np.log10(i))))

    header.append(r'\multicolumn{2}{|@{}r@{}}{\#succ}')
    header_html.append('<td>#succ</td>\n</tr>\n</thead>\n')

    for d in dims_of_interest:
        tableHtml = header_html[:]
        try:
            dictFunc = dictDim[d].dictByFunc()
        except KeyError:
            continue
        funcs = set(dictFunc.keys())
        nbtests = float(len(funcs))  # #funcs tests times one algorithm

        tableHtml.append('<tbody>\n')
        for f in sorted(funcs):
            table = []
            extraeol = []

            tableHtml.append('<tr>\n')
            curline = [r'${\bf f_{%d}}$' % f]
            curlineHtml = ['<th><b>f<sub>%d</sub></b></th>\n' % f]

            # generate all data for ranksum test
            assert len(dictFunc[f]) == 1
            entry = dictFunc[f][0]  # take the first element
            ertdata = entry.detERT(targetsOfInterest((f, d)))

            if refalgentries:
                refalgentry = refalgentries[(d, f)]
                refalgdata = refalgentry.detERT(targetsOfInterest((f, d)))
                if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
                    # write ftarget:fevals
                    for i in range(len(refalgdata[:-1])):
                        temp = "%.1e" % targetsOfInterest((f, d))[i]
                        if temp[-2] == "0":
                            temp = temp[:-2] + temp[-1]
                        curline.append(r'\multicolumn{2}{@{}c@{}}{\textit{%s}:%s \quad}'
                                       % (temp, writeFEvalsMaxPrec(refalgdata[i], 2)))
                        curlineHtml.append('<td><i>%s</i>:%s</td>\n'
                                           % (temp, writeFEvalsMaxPrec(refalgdata[i], 2)))
                    temp = "%.1e" % targetsOfInterest((f, d))[-1]
                    if temp[-2] == "0":
                        temp = temp[:-2] + temp[-1]
                    curline.append(r'\multicolumn{2}{@{}c|@{}}{\textit{%s}:%s \quad}'
                                   % (temp, writeFEvalsMaxPrec(refalgdata[i], 2)))
                    curlineHtml.append('<td><i>%s</i>:%s</td>\n'
                                       % (temp, writeFEvalsMaxPrec(refalgdata[i], 2)))
                    # success
                    targetf = targetsOfInterest((f, d))[-1]

                else:
                    # write #fevals of the reference alg
                    for i in refalgdata[:-1]:
                        curline.append(r'\multicolumn{2}{@{}c@{}}{%s \quad}'
                                       % writeFEvalsMaxPrec(i, 2))
                        curlineHtml.append('<td>%s</td>\n' % writeFEvalsMaxPrec(i, 2))
                    curline.append(r'\multicolumn{2}{@{}c|@{}}{%s}'
                                   % writeFEvalsMaxPrec(refalgdata[-1], 2))
                    curlineHtml.append('<td>%s</td>\n' % writeFEvalsMaxPrec(refalgdata[-1], 2))

                # write the success ratio for the reference alg
                successful_runs, all_runs = refalgentry.get_success_ratio(targetf)
                curline.append('%d' % successful_runs)
                curline.append('/%d' % all_runs)
                curlineHtml.append('<td>%d/%d</td>\n' % (successful_runs, all_runs))

                curlineHtml = [i.replace('$\infty$', '&infin;') for i in curlineHtml]
                table.append(curline[:])
                tableHtml.extend(curlineHtml[:])
                tableHtml.append('</tr>\n')
                extraeol.append('')

                testresrefvs1 = significancetest(refalgentry, entry,
                                                 targetsOfInterest((f, d)))

                tableHtml.append('<tr>\n')
                # for nb, entry in enumerate(entries):
                # curline = [r'\algshort\hspace*{\fill}']
                curline = ['']
                curlineHtml = ['<th></th>\n']

            # data = entry.detERT(targetsOfInterest)
            evals = entry.detEvals(targetsOfInterest((f, d)))
            dispersion = []
            data = []
            for i in evals:
                succ = np.isfinite(i)  # was: (np.isnan(i) == False)
                tmp = i.copy()
                tmp[np.logical_not(succ)] = entry.maxevals[np.logical_not(succ)]
                # set_trace()
                # TODO: what is the difference between data and ertdata?
                data.append(toolsstats.sp(tmp, issuccessful=succ)[0])
                # if not any(succ):
                #   set_trace()
                if any(succ):
                    tmp2 = toolsstats.drawSP(tmp[succ], tmp[succ == False],
                                             (10, 50, 90), entry.bootstrap_sample_size())[0]
                    dispersion.append((tmp2[-1] - tmp2[0]) / 2.)
                else:
                    dispersion.append(None)
            if data != ertdata:
                # comment before computeERT was called unconditionally at the end of DataSet.__init__:
                # warning comes only in balance_instances=True setting only for bfgs but not for randsearch or bipop
                # ertdata values are consistently slightly larger than data values, after calling entry.computeERTfromEvals
                # the ertdata values become the same as the data values
                warnings.warn("data != ertdata " + str((entry,
                                                        getattr(entry, 'instancenumbers',
                                                                '`instancenumbers` attribute is missing'),
                                                        entry.instance_multipliers, data, ertdata)))
                # assert data == ertdata
            for i, ert in enumerate(data):
                alignment = 'c'
                if i == len(data) - 1:  # last element
                    alignment = 'c|'

                nbstars = 0
                if refalgentries:
                    z, p = testresrefvs1[i]
                    if ert - refalgdata[i] < 0. and not np.isinf(refalgdata[i]):
                        evals = entry.detEvals([targetsOfInterest((f, d))[i]])[0]
                        evals[np.isnan(evals)] = entry.maxevals[np.isnan(evals)]
                        refevals = refalgentry.detEvals([targetsOfInterest((f, d))[i]])
                        refevals, refalgalg = (refevals[0][0], refevals[1][0])
                        refevals[np.isnan(refevals)] = refalgentry.maxevals[refalgalg][np.isnan(refevals)]
                        evals = np.array(sorted(evals))[0:min(len(evals), len(refevals))]
                        refevals = np.array(sorted(refevals))[0:min(len(evals), len(refevals))]

                    # The conditions for significance are now that ERT < ERT_ref and
                    # all(sorted(FEvals_ref) > sorted(FEvals_current)).
                    if ((nbtests * p) < 0.05 and ert - refalgdata[i] < 0. and
                            z < 0. and (np.isinf(refalgdata[i]) or all(evals < refevals))):
                        nbstars = -np.ceil(np.log10(nbtests * p))
                is_bold = False
                if nbstars > 0:
                    is_bold = True

                if refalgentries and np.isinf(refalgdata[i]):  # if the reference algorithm did not solve the problem
                    tmp = writeFEvalsMaxPrec(float(ert), 2)
                    if not np.isinf(ert):
                        if refalgentries:
                            tmpHtml = '<i>%s</i>' % (tmp)
                            tmp = r'\textit{%s}' % (tmp)
                        else:
                            tmpHtml = tmp

                        if is_bold:
                            tmp = r'\textbf{%s}' % tmp
                            tmpHtml = '<b>%s</b>' % tmpHtml
                    else:
                        tmpHtml = tmp
                    tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                  % (alignment, tmp))
                    tableentryHtml = ('%s' % tmpHtml)
                else:
                    # Formatting
                    tmp = float(ert) / refalgdata[i] if refalgentries else float(ert)
                    assert not np.isnan(tmp)
                    tableentry = writeFEvalsMaxPrec(tmp, 2)
                    tableentryHtml = writeFEvalsMaxPrec(tmp, 2)

                    if np.isinf(tmp) and i == len(data) - 1:
                        tableentry = (
                                    tableentry + r'\textit{%s}' % writeFEvals2(np.median(entry.maxevals).astype(int), 2,
                                                                               3))
                        tableentryHtml = (
                                    tableentryHtml + ' <i>%s</i>' % writeFEvals2(np.median(entry.maxevals).astype(int),
                                                                                 2, 4))
                        if is_bold:
                            tableentry = r'\textbf{%s}' % tableentry
                            tableentryHtml = '<b>%s</b>' % tableentryHtml
                        elif 11 < 3:  # and significance0vs1 < 0:
                            tableentry = r'\textit{%s}' % tableentry
                            tableentryHtml = '<i>%s</i>' % tableentryHtml
                        tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                      % (alignment, tableentry))
                    elif tableentry.find('e') > -1 or (np.isinf(tmp) and i != len(data) - 1):
                        if is_bold:
                            tableentry = r'\textbf{%s}' % tableentry
                            tableentryHtml = '<b>%s</b>' % tableentryHtml
                        elif 11 < 3:  # and significance0vs1 < 0:
                            tableentry = r'\textit{%s}' % tableentry
                            tableentryHtml = '<i>%s</i>' % tableentryHtml
                        tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                      % (alignment, tableentry))
                    else:
                        tmp = tableentry.split('.', 1)
                        tmpHtml = tableentryHtml.split('.', 1)
                        if is_bold:
                            tmp = list(r'\textbf{%s}' % i for i in tmp)
                            tmpHtml = list('<b>%s</b>' % i for i in tmpHtml)
                        elif 11 < 3:  # and significance0vs1 < 0:
                            tmp = list(r'\textit{%s}' % i for i in tmp)
                            tmpHtml = list('<i>%s</i>' % i for i in tmpHtml)
                        tableentry = ' & .'.join(tmp)
                        tableentryHtml = '.'.join(tmpHtml)
                        if len(tmp) == 1:
                            tableentry += '&'

                superscript = ''
                superscriptHtml = ''

                if nbstars > 0:
                    # tmp = '\hspace{-.5ex}'.join(nbstars * [r'\star'])
                    if z > 0:
                        superscript = r'\uparrow'  # * nbstars
                        superscriptHtml = '&uarr;'
                    else:
                        superscript = r'\downarrow'  # * nbstars
                        superscriptHtml = '&darr;'
                    if nbstars > 1:
                        superscript += str(int(min((9, nbstars))))
                        superscriptHtml += str(int(min(9, nbstars)))
                        # superscript += str(int(nbstars))

                # if superscript or significance0vs1:
                    # s = ''
                    # if significance0vs1 > 0:
                       # s = '\star'
                    # if significance0vs1 > 1:
                       # s += str(significance0vs1)
                    # s = r'$^{' + s + superscript + r'}$'

                    # if tableentry.endswith('}'):
                        # tableentry = tableentry[:-1] + s + r'}'
                    # else:
                        # tableentry += s

                if dispersion[i]:
                    if refalgentries and not np.isinf(refalgdata[i]):
                        tmp = writeFEvalsMaxPrec(dispersion[i] / refalgdata[i], 1)
                    else:
                        tmp = writeFEvalsMaxPrec(dispersion[i], 1)
                    s = r'(%s)' % tmp
                    if tableentry.endswith('}'):
                        tableentry = tableentry[:-1] + s + r'}'
                    else:
                        tableentry += s
                    tableentryHtml += (' (%s)' % tmp)

                if superscript:
                    s = r'$^{' + superscript + r'}$'
                    shtml = '<sup>' + superscriptHtml + '</sup>'

                    if tableentry.endswith('}'):
                        tableentry = tableentry[:-1] + s + r'}'
                    else:
                        tableentry += s
                    tableentryHtml += shtml

                tableentryHtml = tableentryHtml.replace('$\infty$', '&infin;')
                curlineHtml.append('<td>%s</td>\n' % tableentryHtml)
                curline.append(tableentry)

                # curline.append(tableentry)
                # if dispersion[i] is None or np.isinf(refalgdata[i]):
                    # curline.append('')
                # else:
                    # tmp = writeFEvalsMaxPrec(dispersion[i]/refalgdata[i], 2)
                    # curline.append('(%s)' % tmp)

            tmp = entry.evals[entry.evals[:, 0] <= targetf, 1:]
            try:
                tmp = tmp[0]
                curline.append('%d' % np.sum(np.isnan(tmp) == False))
                curlineHtml.append('<td>%d' % np.sum(np.isnan(tmp) == False))
            except IndexError:
                curline.append('%d' % 0)
                curlineHtml.append('<td>%d' % 0)
            curline.append('/%d' % entry.nbRuns())
            curlineHtml.append('/%d</td>\n' % entry.nbRuns())

            table.append(curline[:])
            tableHtml.extend(curlineHtml[:])
            tableHtml.append('</tr>\n')
            extraeol.append(r'\hline')

            extraeol[-1] = ''

            output_file = os.path.join(outputdir, 'pptable_f%03d_%02dD.tex' % (f, d))
            if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
                spec = r'@{}c@{}|' + '*{%d}{@{ }r@{}@{}l@{}}' % len(targetsOfInterest) + '|@{}r@{}@{}l@{}'
            else:
                spec = r'@{}c@{}|' + '*{%d}{@{}r@{}@{}l@{}}' % len(targetsOfInterest) + '|@{}r@{}@{}l@{}'
            # res = r'\providecommand{\algshort}{%s}' % alg1 + '\n'
            res = tableLaTeX(table, spec=spec, extra_eol=extraeol, add_begin_tabular=False, add_end_tabular=False)
            f = open(output_file, 'w')
            f.write(res)
            f.close()

        res = ("").join(str(item) for item in tableHtml)
        res = '<table>\n%s</table>\n' % res

        filename = os.path.join(outputdir, 'pptable.html')
        lines = []
        html_string = '<!--pptableHtml_%d-->' % d
        with open(filename) as infile:
            for line in infile:
                if html_string in line:
                    lines.append(res)
                lines.append(line)

        with open(filename, 'w') as outfile:
            for line in lines:
                outfile.write(line)

        if genericsettings.verbose:
            print("Table written in %s" % output_file)

    if len(dims_of_interest) > 0:
        extraeol = [r'\hline']
        res = tableLaTeX([header], spec=spec, extra_eol=extraeol, add_end_tabular=False)
        prepend_to_file(latex_commands_file, ['\\providecommand{\\pptableheader}{', res, '}'])

        res = tableLaTeX([], spec=spec, add_begin_tabular=False)
        prepend_to_file(latex_commands_file, ['\\providecommand{\\pptablefooter}{', res, '}'])
