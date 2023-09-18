#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for the generation of TeX tables."""
from __future__ import absolute_import, print_function

import os
import sys
import warnings
import numpy

from .. import genericsettings, bestalg, toolsstats, pproc, ppfigparam, testbedsettings, captions, ppfig
from ..pptex import writeFEvals2, writeFEvalsMaxPrec, tableXLaTeX, numtotext
from ..toolsstats import significancetest, significance_all_best_vs_other, best_alg_indices
from ..toolsdivers import str_to_latex, strip_pathname1, strip_pathname3, replace_in_file, get_version_label, prepend_to_file


def get_table_caption():
    """Sets table caption.
    
       Based on the testbedsettings.current_testbed
       and genericsettings.runlength_based_targets.
    """

    table_caption_one = (r"""%
        Expected runtime (\ERT\ in number of """
        + testbedsettings.current_testbed.string_evals
        + r""") divided by the respective !!BEST-ERT!! (when finite) in
        #1.
        This \ERT\ ratio and, in braces as dispersion measure, the half difference between
        10 and 90\%-tile of bootstrapped run lengths appear for each algorithm and 
        """)
    table_caption_one_noreference = (r"""%
        Expected runtime (\ERT) to reach given targets, measured
        in number of """
        + testbedsettings.current_testbed.string_evals
        + r""", in #1. For each function, the \ERT\ 
        and, in braces as dispersion measure, the half difference between 10 and 
        90\%-tile of (bootstrapped) runtimes is shown for the different
        target """
        + ("!!DF!!-" if not testbedsettings.current_testbed.has_constraints else "precision ")
        + r"""values as shown in the top row. 
        \#succ is the number of trials that reached the last target
        $!!FOPT!! + """ + testbedsettings.current_testbed.hardesttargetlatex + r"""$.
        """)
    table_caption_two1 = r"""%
        target, the corresponding reference \ERT\
        in the first row. The different target !!DF!!-values are shown in the top row.
        \#succ is the number of trials that reached the (final) target
        $!!FOPT!! + """ + testbedsettings.current_testbed.hardesttargetlatex + r"""$.
        """
    table_caption_two2 = r"""%
        run-length based target, the corresponding reference \ERT\
        (preceded by the target !!DF!!-value in \textit{italics}) in the first row. 
        \#succ is the number of trials that reached the target value of the last column.
        """

    table_caption_rest = (r"""%
        The median number of conducted function evaluations is additionally given in 
        \textit{italics}, if the target in the last column was never reached.
        Entries, succeeded by a star, are statistically significantly better (according to
        the rank-sum test) when compared to all other algorithms of the table, with
        $p = 0.05$ or $p = 10^{-k}$ when the number $k$ following the star is larger
        than 1, with Bonferroni correction by the number of functions (!!TOTAL-NUM-OF-FUNCTIONS!!). """ +
                (r"""A ${}$ signifies the number of trials that were worse than the ERT of !!THE-REF-ALG!! """
                 r"""shown only when less than 10 percent were worse and the ERT was better."""
                 .format(significance_vs_ref_symbol)
                    if not (testbedsettings.current_testbed.reference_algorithm_filename == '' or
                            testbedsettings.current_testbed.reference_algorithm_filename is None)
                 else "") + r"""Best results are printed in bold.
        """)

    table_caption = None
    if (testbedsettings.current_testbed.reference_algorithm_filename == '' or
            testbedsettings.current_testbed.reference_algorithm_filename is None):
        # NOTE: no runlength-based targets supported yet
        table_caption = table_caption_one_noreference + table_caption_rest
    else:
        if genericsettings.runlength_based_targets:
            table_caption = table_caption_one + table_caption_two2 + table_caption_rest
        else:
            table_caption = table_caption_one + table_caption_two1 + table_caption_rest

    return captions.replace(table_caption)

show_number_of_better_runs_threshold = 0.9  # compared to reference/best algorithm

table_column_width = 100  # used with <td style="min-width:%dpx"> % ...
table_first_column_width = 250  # in 'px', aligns the second column in the table

with_table_heading = False  # in case the page is long enough

allmintarget = {}
allmedtarget = {}

significance_vs_others_symbol = r"\star"
significance_vs_others_symbol_html = r"&#9733;"
significance_vs_ref_symbol = r"\uparrow"
significance_vs_ref_symbol_html = r"&uarr;"
maxfloatrepr = 10000.
samplesize = genericsettings.simulated_runlength_bootstrap_sample_size
precfloat = 2
precscien = 2
precdispersion = 1  # significant digits for dispersion


def cite(alg_name, is_noise_free, is_noisy):
    """Returns the citation key associated to the algorithm name.

    Hard coded while no other solution is found.

    """
    res = []
    # The names of the algorithms must correspond to the name of the folder
    # containing the data. The citations keys must be in bbob.bib.
    if is_noise_free:
        if alg_name == "ALPS-GA":
            res.append("Hornby:2009")
        if alg_name in ("AMaLGaM IDEA", "iAMaLGaM IDEA"):
            res.append("DBLP:conf/gecco/BosmanGT09")
        if alg_name == "BayEDAcG":
            res.append("DBLP:conf/gecco/Gallagher09")
        if alg_name == "BFGS":
            res.append("DBLP:conf/gecco/Ros09")
        if alg_name == "Cauchy EDA":
            res.append("DBLP:conf/gecco/Posik09")
        if alg_name == "BIPOP-CMA-ES":
            res.append("DBLP:conf/gecco/Hansen09")
        if alg_name == "(1+1)-CMA-ES":
            res.append("DBLP:conf/gecco/AugerH09")
        if alg_name == "DASA":
            res.append("DBLP:conf/gecco/KorosecS09")
        if alg_name == "DEPSO":
            res.append("DBLP:conf/gecco/Garcia-NietoAA09")
        if alg_name == "DIRECT":
            res.append("DBLP:conf/gecco/Posik09a")
        if alg_name == "EDA-PSO":
            res.append("DBLP:conf/gecco/El-AbdK09")
        if alg_name == "CMA-EGS":
            res.append("Finck:2009")
        if alg_name == "G3-PCX":
            res.append("DBLP:conf/gecco/Posik09b")
        if alg_name == "simple GA":
            res.append("DBLP:conf/gecco/Nicolau09")
        if alg_name == "GLOBAL":
            res.append("Pal:2009a")
        if alg_name in ("LSfminbnd", "LSstep"):
            res.append("DBLP:conf/gecco/Posik09c")
        if alg_name == "MA-LS-Chain":
            res.append("DBLP:conf/gecco/MolinaLH09")
        if alg_name == "MCS":
            res.append("Huyer:2009b")
        if alg_name == "NELDER (Han)":
            res.append("DBLP:conf/gecco/Hansen09b")
        if alg_name == "NELDER (Doe)":
            res.append("DBLP:conf/gecco/DoerrFSW09")
        if alg_name in ("NEWUOA", "avg NEWUOA", "full NEWUOA"):
            res.append("DBLP:conf/gecco/Ros09b")
        if alg_name == "(1+1)-ES":
            res.append("DBLP:conf/gecco/Auger09")
        if alg_name == "POEMS":
            res.append("DBLP:conf/gecco/Kubalik09a")
        if alg_name == "PSO":
            res.append("DBLP:conf/gecco/El-AbdK09a")
        if alg_name == "PSO\_Bounds":
            res.append("DBLP:conf/gecco/El-AbdK09b")
        if alg_name == "Monte Carlo":
            res.append("DBLP:conf/gecco/AugerR09")
        if alg_name == "Rosenbrock":
            res.append("DBLP:conf/gecco/Posik09d")
        if alg_name == "IPOP-SEP-CMA-ES":
            res.append("DBLP:conf/gecco/Ros09d")
        if alg_name == "VNS (Garcia)":
            res.append("DBLP:conf/gecco/Garcia-MartinezL09")
    if is_noisy:
        if alg_name == "ALPS-GA":
            res.append("Hornby:2009a")
        elif alg_name in ("AMaLGaM IDEA", "iAMaLGaM IDEA"):
            res.append("DBLP:conf/gecco/BosmanGT09a")
        elif alg_name in ("avg NEWUOA", "full NEWUOA", "NEWUOA"):
            res.append("DBLP:conf/gecco/Ros09c")
        elif alg_name == "BayEDAcG":
            res.append("DBLP:conf/gecco/Gallagher09a")
        elif alg_name == "BFGS":
            res.append("DBLP:conf/gecco/Ros09a")
        elif alg_name == "BIPOP-CMA-ES":
            res.append("DBLP:conf/gecco/Hansen09a")
        elif alg_name == "(1+1)-CMA-ES":
            res.append("DBLP:conf/gecco/AugerH09a")
        elif alg_name == "DASA":
            res.append("DBLP:conf/gecco/KorosecS09a")
        elif alg_name == "DEPSO":
            res.append("DBLP:conf/gecco/Garcia-NietoAA09a")
        elif alg_name == "EDA-PSO":
            res.append("DBLP:conf/gecco/El-AbdK09")
        elif alg_name == "CMA-EGS":
            res.append("Finck:2009a")
        elif alg_name == "GLOBAL":
            res.append("Pal:2009")
        elif alg_name == "MA-LS-Chain":
            res.append("DBLP:conf/gecco/MolinaLH09a")
        elif alg_name == "MCS":
            res.append("Huyer:2009a")
        elif alg_name == "(1+1)-ES":
            res.append("DBLP:conf/gecco/Auger09a")
        elif alg_name == "PSO":
            res.append("DBLP:conf/gecco/El-AbdK09a")
        elif alg_name == "PSO\_Bounds":
            res.append("DBLP:conf/gecco/El-AbdK09b")
        elif alg_name == "Monte Carlo":
            res.append("DBLP:conf/gecco/AugerR09a")
        elif alg_name == "IPOP-SEP-CMA-ES":
            res.append("DBLP:conf/gecco/Ros09e")
        elif alg_name == "SNOBFIT":
            res.append("Huyer:2009")
        elif alg_name == "VNS (Garcia)":
            res.append("DBLP:conf/gecco/Garcia-MartinezL09a")

    if res:
        res = r"\cite{%s}" % (", ".join(res))
        # set_trace()
    else:
        # res = r"\cite{add_an_entry_for_%s_in_bbob.bib}" % algName
        res = ""
    return res


def get_top_indices_of_columns(table, max_rank=None):
    """For each column, returns a list of the maxRank-ranked elements.

    This list may have a length larger than maxRank in the case of ties.

    """
    if max_rank is None:
        max_rank = numpy.shape(table)[0]

    ranked = []  # the length of ranked will be the number of columns in table.
    trans_table = numpy.transpose(table)
    for line in trans_table:
        sid = line.argsort()  # returns the sorted index of the elements of line
        prev_value = None
        rank = []
        for idx in sid:
            if line[idx] == prev_value:  # tie
                continue
            prev_value = line[idx]
            rank.extend(numpy.where(line == prev_value)[0])
            if len(rank) >= max_rank:
                break
        ranked.append(rank)

    return ranked


# TODO: function_headings argument need to be tested, default should be changed according to templates
def main(dict_alg, sorted_algs, output_dir='.', function_targets_line=True, latex_commands_file=''):  # [1, 13, 101]
    """Generate one table per func with results of multiple algorithms."""
    """Difference with the first version:

    * numbers aligned using the decimal separator
    * premices for dispersion measure
    * significance test against best algorithm
    * table width...

    Takes ``pptable_targetsOfInterest`` from testbedsetting's Testbed instance
    as "input argument" to compute the desired target values.
    ``pptable_targetsOfInterest`` might be configured via config.
    
    """

    # TODO: method is long, terrible to read, split if possible

    testbed = testbedsettings.current_testbed
    targets_of_interest = testbed.pptablemany_targetsOfInterest

    refalgentries = bestalg.load_reference_algorithm(testbed.reference_algorithm_filename)

    plotting_style_list = ppfig.get_plotting_styles(sorted_algs, True)
    sorted_algs = plotting_style_list[0].algorithm_list

    # Sort data per dimension and function
    dict_data = {}
    dsListperAlg = list(dict_alg[i] for i in sorted_algs)
    for n, entries in enumerate(dsListperAlg):
        tmpdictdim = entries.dictByDim()
        for d in tmpdictdim:
            tmpdictfun = tmpdictdim[d].dictByFunc()
            for f in tmpdictfun:
                dict_data.setdefault((d, f), {})[n] = tmpdictfun[f]

    nbtests = len(dict_data)

    fun_infos = ppfigparam.read_fun_infos()

    tables_header = []
    additional_commands = []
    for df in sorted(dict_data):
        # Generate one table per df
        # first update targets for each dimension-function pair if needed:
        targets = targets_of_interest((df[1], df[0]))
        if isinstance(targets_of_interest, pproc.RunlengthBasedTargetValues):
            targetf = targets[-1]
        else:
            targetf = testbed.pptable_ftarget

        # reference algorithm
        if refalgentries:
            refalgentry = refalgentries[df]
            refalgert = refalgentry.detERT(targets)

        # Process the data
        # The following variables will be lists of elements each corresponding
        # to an algorithm
        algnames = []
        # algdata = []
        algerts = []
        algevals = []
        algdisp = []
        algnbsucc = []
        algnbruns = []
        algmedmaxevals = []
        algmedfinalfunvals = []
        algtestres = []
        algentries = []

        for n in sorted(dict_data[df].keys()):
            entries = dict_data[df][n]
            # the number of datasets for a given dimension and function (df)
            # should be strictly 1. TODO: find a way to warn
            # TODO: do this checking before... why wasn't it triggered by ppperprof?
            if len(entries) > 1:
                print(entries)
                txt = ("There is more than a single entry associated with "
                       "folder %s on %d-D f%d." % (sorted_algs[n], df[0], df[1]))
                raise Exception(txt)

            entry = entries[0]
            algentries.append(entry)

            algnames.append(sorted_algs[n])

            evals = entry.detEvals(targets)
            # tmpdata = []
            tmpdisp = []
            tmpert = []
            for i, e in enumerate(evals):
                succ = (numpy.isnan(e) == False)
                ec = e.copy()  # note: here was the previous bug (changes made in e also appeared in evals !)
                ec[succ == False] = entry.maxevals[succ == False]
                ert = toolsstats.sp(ec, issuccessful=succ)[0]
                # tmpdata.append(ert/refalgert[i])
                if succ.any():
                    tmp = toolsstats.drawSP(ec[succ], entry.maxevals[succ == False],
                                            [10, 50, 90], samplesize=entry.bootstrap_sample_size(samplesize))[0]
                    tmpdisp.append((tmp[-1] - tmp[0]) / 2.)
                else:
                    tmpdisp.append(numpy.nan)
                tmpert.append(ert)
            algerts.append(tmpert)
            algevals.append(evals)
            # algdata.append(tmpdata)
            algdisp.append(tmpdisp)
            algmedmaxevals.append(numpy.median(entry.maxevals))
            algmedfinalfunvals.append(numpy.median(entry.finalfunvals))
            # algmedmaxevals.append(numpy.median(entry.maxevals)/df[0])
            # algmedfinalfunvals.append(numpy.median(entry.finalfunvals))

            if refalgentries:
                nbs = []  # number of worse runs to show for each target
                for target in targets:
                    nbs.append(None)  # overwrite when "significance" is found
                    ert = entry.detERT([target])[0]
                    if not numpy.isfinite(ert):
                        continue
                    ref_ert = refalgentry.detERT([target])[0]
                    if ert >= ref_ert:
                        continue
                    nbetter = entry._number_of_better_runs(target, ref_ert)
                    if nbetter > show_number_of_better_runs_threshold * entry.nbRuns():
                        nbs[-1] = entry.nbRuns() - nbetter  # number of worse runs
                        assert nbs[-1] >= 0
                algtestres.append(nbs)

            # determine success probability for Df = 1e-8
            e = entry.detEvals((targetf,))[0]
            algnbsucc.append(numpy.sum(numpy.isnan(e) == False))
            algnbruns.append(len(e))

        # Process over all data
        # find best values...

        nalgs = len(dict_data[df])
        maxRank = 1 + numpy.floor(0.14 * nalgs)  # number of algs to be displayed in bold

        isBoldArray = []  # Point out the best values
        algfinaldata = []  # Store median function values/median number of function evaluations
        tmptop = get_top_indices_of_columns(algerts, max_rank=maxRank)
        for i, erts in enumerate(algerts):
            tmp = []
            for j, ert in enumerate(erts):  # algi targetj
                tmp.append(i in tmptop[j] or (refalgentries and nalgs > 7 and algerts[i][j] <= 3. * refalgert[j]))
            isBoldArray.append(tmp)
            algfinaldata.append((algmedfinalfunvals[i], algmedmaxevals[i]))

        # significance test of best given algorithm against all others
        significance_versus_others, best_alg_idx = significance_all_best_vs_other(
            algentries, targets, best_alg_indices(algerts, algmedfinalfunvals))
        # Create the table
        table = []
        tableHtml = []
        spec = r'@{}c@{}|*{%d}{@{\,}r@{}X@{\,}}|@{}r@{}@{}l@{}' % (
        len(targets_of_interest))  # in case StrLeft not working: replaced c@{} with l@{ }
        spec = r'@{}c@{}|*{%d}{@{}r@{}X@{}}|@{}r@{}@{}l@{}' % (
        len(targets_of_interest))  # in case StrLeft not working: replaced c@{} with l@{ }
        extraeol = []

        # Generate header lines
        if with_table_heading:
            header = fun_infos[df[1]] if df[1] in fun_infos.keys() else 'f%d' % df[1]
            table.append([r'\multicolumn{%d}{@{\,}c@{\,}}{{\textbf{%s}}}'
                          % (2 * len(targets_of_interest) + 2, header)])
            extraeol.append('')

        # generate line with displayed quality indicator and targets:
        if function_targets_line is True or (function_targets_line and df[1] in function_targets_line):
            if isinstance(targets_of_interest, pproc.RunlengthBasedTargetValues):
                curline = [r'\#FEs/D']
                counter = 1
                for i in targets_of_interest.labels():
                    curline.append(r'\multicolumn{2}{@{}c@{}}{%s}' % i)
                    counter += 1
            else:
                if (testbed.name in [testbedsettings.suite_name_bi,
                                     testbedsettings.suite_name_bi_ext,
                                     testbedsettings.suite_name_bi_mixint]):
                    curline = [r'$\Df$']
                else:
                    curline = [r'$\Delta f_\mathrm{opt}$']
                counter = 1
                for t in targets:
                    curline.append(r'\multicolumn{2}{@{\,}l@{\,}}{%s}'
                                   % writeFEvals2(t, precision=1, isscientific=True))
                    counter += 1
                #                curline.append(r'\multicolumn{2}{@{\,}l@{}|}{%s}'
                #                            % writeFEvals2(targets_of_interest[-1], precision=1, isscientific=True))
            if (testbed.name in [testbedsettings.suite_name_bi,
                                 testbedsettings.suite_name_bi_ext,
                                 testbedsettings.suite_name_bi_mixint]):
                curline.append(r'\multicolumn{2}{|@{}l@{}}{\begin{rotate}{30}\#succ\end{rotate}}')
            else:
                curline.append(r'\multicolumn{2}{|@{}l@{}}{\#succ}')
            tables_header = curline[:]

        # do the same for the HTML output, but all the time:
        curlineHtml = []
        if isinstance(targets_of_interest, pproc.RunlengthBasedTargetValues):
            curlineHtml = ['<thead>\n<tr>\n<th>#FEs/D<br>REPLACEH</th>\n']
            counter = 1
            for i in targets_of_interest.labels():
                curlineHtml.append('<td style="min-width:%dpx">%s<br>REPLACE%d</td>\n' % (
                    table_column_width, i, counter))
                counter += 1
        else:
            if (testbed.name in [testbedsettings.suite_name_bi,
                                 testbedsettings.suite_name_bi_ext,
                                 testbedsettings.suite_name_bi_mixint]):
                curlineHtml = ['<thead>\n<tr>\n<th>&#916; HV<sub>ref</sub><br>REPLACEH</th>\n']
            else:
                curlineHtml = ['<thead>\n<tr>\n<th>&#916; f<sub>opt</sub><br>REPLACEH</th>\n']
            counter = 1
            for t in targets:
                curlineHtml.append(
                    '<td style="min-width:%dpx">%s<br>REPLACE%d</td>\n' % (
                        table_column_width, writeFEvals2(t, precision=1, isscientific=True), counter))
                counter += 1
        curlineHtml.append('<td>#succ<br>REPLACEF</td>\n</tr>\n</thead>\n')
 
        extraeol.append(r'\hline')
        #        extraeol.append(r'\hline\arrayrulecolor{tableShade}')

        # line with function name and potential ERT values of reference algorithm
        curline = [r'\textbf{f%d}' % df[1]]
        replaceValue = '<b>f%d, %d-D</b>' % (df[1], df[0])
        curlineHtml = [item.replace('REPLACEH', replaceValue) for item in curlineHtml]

        ### write the header row
        if refalgentries:
            if isinstance(targets_of_interest, pproc.RunlengthBasedTargetValues):
                # write ftarget:fevals
                counter = 1
                for i in range(len(refalgert[:-1])):
                    temp = "%.1e" % targets_of_interest((df[1], df[0]))[i]
                    if temp[-2] == "0":
                        temp = temp[:-2] + temp[-1]
                    curline.append(r'\multicolumn{2}{@{}c@{}}{\textit{%s}:%s \quad}'
                                   % (temp, writeFEvalsMaxPrec(refalgert[i], 2)))
                    replaceValue = '<i>%s</i>:%s' % (temp, writeFEvalsMaxPrec(refalgert[i], 2))
                    curlineHtml = [item.replace('REPLACE%d' % counter, replaceValue) for item in curlineHtml]
                    counter += 1

                temp = "%.1e" % targets_of_interest((df[1], df[0]))[-1]
                if temp[-2] == "0":
                    temp = temp[:-2] + temp[-1]
                curline.append(r'\multicolumn{2}{@{}c@{}|}{\textit{%s}:%s }'
                               % (temp, writeFEvalsMaxPrec(refalgert[-1], 2)))
                replaceValue = '<i>%s</i>:%s' % (temp, writeFEvalsMaxPrec(refalgert[-1], 2))
                curlineHtml = [item.replace('REPLACE%d' % counter, replaceValue) for item in curlineHtml]
            else:
                # write #fevals of the reference alg
                counter = 1
                for i in refalgert[:-1]:
                    curline.append(r'\multicolumn{2}{@{}c@{}}{%s \quad}'
                                   % writeFEvalsMaxPrec(i, 2))
                    curlineHtml = [item.replace('REPLACE%d' % counter, writeFEvalsMaxPrec(i, 2)) for item in
                                   curlineHtml]
                    counter += 1
                curline.append(r'\multicolumn{2}{@{}c@{}|}{%s}'
                               % writeFEvalsMaxPrec(refalgert[-1], 2))
                curlineHtml = [item.replace('REPLACE%d' % counter, writeFEvalsMaxPrec(refalgert[-1], 2)) for item in
                               curlineHtml]

            # write the success ratio for the reference alg
            successful_runs, all_runs = refalgentry.get_success_ratio(targetf)
            curline.append('%d' % successful_runs)
            curline.append('/%d' % all_runs)
            replaceValue = '%d/%d' % (successful_runs, all_runs)
            curlineHtml = [item.replace('REPLACEF', replaceValue) for item in curlineHtml]

        else:  # if not refalgentries
            curline.append(r'\multicolumn{%d}{@{}c@{}|}{} & ' % (2 * (len(targets_of_interest))))
            for counter in range(1, len(targets_of_interest) + 1):
                curlineHtml = [item.replace('REPLACE%d' % counter, '&nbsp;') for item in curlineHtml]
            curlineHtml = [item.replace('REPLACEF', '&nbsp;') for item in curlineHtml]

        curlineHtml = [i.replace('$\infty$', '&infin;') for i in curlineHtml]
        table.append(curline[:])
        tableHtml.extend(curlineHtml[:])
        tableHtml.append('<tbody>\n')
        extraeol.append('')

        ### write a row for each algorithm
        additional_commands = ['\\providecommand{\\ntables}{%d}' % len(targets_of_interest)]
        for i, alg in enumerate(algnames):
            tableHtml.append('<tr>\n')
            # algname, entries, irs, line, line2, succ, runs, testres1alg in zip(algnames,
            # data, dispersion, isBoldArray, isItalArray, nbsucc, nbruns, testres):
            command_name = r'\alg%stables' % numtotext(i)
            #            header += r'\providecommand{%s}{{%s}{}}' % (command_name, str_to_latex(strip_pathname(alg)))
            if df[0] == testbedsettings.current_testbed.tabDimsOfInterest[0]:
                additional_commands.append('\\providecommand{%s}{\\StrLeft{%s}{\\ntables}}' %
                                           (command_name, str_to_latex(strip_pathname1(alg))))
            curline = [command_name + r'\hspace*{\fill}']  # each list element becomes a &-separated table entry?
            curlineHtml = ['<th style="width:%dpx">%s</th>\n' % (
                table_first_column_width, str_to_latex(strip_pathname3(alg)))]

            zipToEnumerate = zip(algerts[i], algdisp[i], isBoldArray[i], algtestres[i]) if refalgentries else zip(
                algerts[i], algdisp[i], isBoldArray[i])

            for j, tmp in enumerate(zipToEnumerate):  # j is target index
                if refalgentries:
                    ert, dispersion, isBold, testres = tmp
                else:
                    ert, dispersion, isBold = tmp

                alignment = '@{\,}l@{\,}'
                if j == len(algerts[i]) - 1:
                    alignment = '@{\,}l@{\,}|'

                # create superscript star for significance against all other algorithms
                str_significance_subsup = ''
                str_significance_subsup_html = ''
                if (len(best_alg_idx) > 0 and len(significance_versus_others) > 0 and
                            i == best_alg_idx[j] and nbtests * significance_versus_others[j][1] < 0.05):
                    logp = -numpy.ceil(numpy.log10(nbtests * significance_versus_others[j][1]))
                    logp = numpy.min((9, logp))  # not messing up the format and handling inf
                    str_significance_subsup = r"^{%s%s}" % (
                    significance_vs_others_symbol, str(int(logp)) if logp > 1 else '')
                    str_significance_subsup_html = '<sup>%s%s</sup>' % (
                    significance_vs_others_symbol_html, str(int(logp)) if logp > 1 else '')

                # create subscript arrow for significance against reference
                if refalgentries:
                    if testres is not None:
                        # tmp2[-1] += r'$^{%s}$' % superscript
                        nb = str(int(testres + 1/2))  # rounded number of runs that were worse
                        str_significance_subsup += r'_{%s%s}' % (significance_vs_ref_symbol, nb)
                        str_significance_subsup_html += '<sub>%s%s</sub>' % (significance_vs_ref_symbol_html, nb)

                if str_significance_subsup:
                    str_significance_subsup = '$%s$' % str_significance_subsup

                # was in case of ert=inf and refalgert[j]=inf:
                # curline.append(r'\multicolumn{2}{%s}{.}' % alignment)
                # curlineHtml.append('<td>&nbsp;</td>')
                # continue

                # write "raw" ERT when reference is inf:
                if numpy.isfinite(ert) and refalgentries and numpy.isinf(refalgert[j]):
                    tableentry = r'\textbf{%s}' % writeFEvalsMaxPrec(algerts[i][j], 2)
                    tableentryHtml = '<b>%s</b>' % writeFEvalsMaxPrec(algerts[i][j], 2)
                    if dispersion and numpy.isfinite(dispersion):
                        tableentry += r'\mbox{\tiny (%s)}' % writeFEvalsMaxPrec(dispersion, precdispersion)
                        tableentryHtml += ' (%s)' % writeFEvalsMaxPrec(dispersion, precdispersion)
                    curline.append(r'\multicolumn{2}{%s}{%s}%s' % (
                        alignment, tableentry, str_significance_subsup))
                    curlineHtml.append('<td sorttable_customkey=\"%f\">%s%s</td>\n' % (
                        algerts[i][j], tableentryHtml, str_significance_subsup_html))
                    continue

                denom = 1
                if refalgentries and numpy.isfinite(refalgert[j]):
                    denom = refalgert[j]
                data = ert / denom

                # format display of variable data
                tmp = writeFEvalsMaxPrec(data, precfloat, maxfloatrepr=maxfloatrepr)
                tmpHtml = writeFEvalsMaxPrec(data, precfloat, maxfloatrepr=maxfloatrepr)
                sortKey = data
                if data >= maxfloatrepr or data < 0.01:  # either inf or scientific notation
                    if numpy.isinf(data) and j == len(algerts[i]) - 1:
                        tmp += r'\,\textit{%s}' % writeFEvalsMaxPrec(algfinaldata[i][1], 0,
                                                                        maxfloatrepr=maxfloatrepr)
                        tmpHtml += '<i>%s</i>' % writeFEvalsMaxPrec(algfinaldata[i][1], 0,
                                                                    maxfloatrepr=maxfloatrepr)
                        sortKey = algfinaldata[i][1]
                    else:
                        tmp = writeFEvalsMaxPrec(data, precscien, maxfloatrepr=data)
                        if isBold:
                            tmpHtml = '<b>%s</b>' % tmp
                            tmp = r'\textbf{%s}' % tmp

                    if not numpy.isnan(dispersion):
                        tmpdisp = dispersion / denom
                        if tmpdisp >= maxfloatrepr or tmpdisp < 0.005:  # TODO: hack
                            tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=tmpdisp)
                        else:
                            tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=maxfloatrepr)
                        tmp += r'\mbox{\tiny (%s)}' % tmpdisp
                        tmpHtml += ' (%s)' % tmpdisp
                    curline.append(r'\multicolumn{2}{%s}{%s%s}' % (alignment, tmp, str_significance_subsup))
                    if (numpy.isinf(sortKey)):
                        sortKey = sys.maxsize
                    curlineHtml.append('<td sorttable_customkey=\"%f\">%s%s</td>' % (
                    sortKey, tmpHtml, str_significance_subsup_html))
                else:
                    tmp2 = tmp.split('.', 1)
                    if len(tmp2) < 2:
                        tmp2.append('')
                    else:
                        tmp2[-1] = '.' + tmp2[-1]
                    if isBold:
                        tmp3 = []
                        tmp3html = []
                        for k in tmp2:
                            tmp3.append(r'\textbf{%s}' % k)
                            tmp3html.append('<b>%s</b>' % k)
                        tmp2 = tmp3
                        tmp2html = tmp3html
                    else:
                        tmp2html = []
                        tmp2html.extend(tmp2)
                    if not numpy.isnan(dispersion):
                        tmpdisp = dispersion / denom
                        if tmpdisp >= maxfloatrepr or tmpdisp < 0.01:
                            tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=tmpdisp)
                        else:
                            tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=maxfloatrepr)
                        tmp2[-1] += (r'\mbox{\tiny (%s)}' % (tmpdisp))
                        tmp2html[-1] += ' (%s)' % tmpdisp
                    tmp2[-1] += str_significance_subsup
                    tmp2html[-1] += str_significance_subsup_html
                    curline.extend(tmp2)
                    tmp2html = ("").join(str(item) for item in tmp2html)
                    curlineHtml.append('<td sorttable_customkey=\"%f\">%s</td>' % (data, tmp2html))

            curline.append('%d' % algnbsucc[i])
            curline.append('/%d' % algnbruns[i])
            table.append(curline)
            curlineHtml.append(
                '<td sorttable_customkey=\"%d\">%d/%d</td>\n' % (algnbsucc[i], algnbsucc[i], algnbruns[i]))
            curlineHtml = [i.replace('$\infty$', '&infin;') for i in curlineHtml]
            tableHtml.extend(curlineHtml[:])
            extraeol.append('')

        # Write table
        res = tableXLaTeX(table, spec=spec, extra_eol=extraeol, add_begin_tabular=False, add_end_tabular=False)
        try:
            filename = os.path.join(output_dir, 'pptables_f%03d_%02dD.tex' % (df[1], df[0]))
            f = open(filename, 'w')
            if with_table_heading:
                f.write(header + '\n')
            f.write(res)

            res = "".join(str(item) for item in tableHtml)
            res = '\n<table class=\"sortable\" >\n%s</table>\n<p/>\n' % res

            if True:
                filename = os.path.join(output_dir, genericsettings.pptables_file_name + '.html')

                lines = []
                html_string = '<!--pptablesHtml_%d-->' % df[0]
                with open(filename) as infile:
                    for line in infile:
                        if html_string in line:
                            lines.append(res)
                        lines.append(line)

                with open(filename, 'w') as outfile:
                    for line in lines:
                        outfile.write(line)
                
                replace_in_file(filename, '??COCOVERSION??', '<br />Data produced with COCO %s' % (get_version_label(None)))

            if genericsettings.verbose:
                print('Wrote table in %s' % filename)
        except:
            raise
        else:
            f.close()
            # TODO: return status

    if len(additional_commands) > 0:
        for command in additional_commands:
            prepend_to_file(latex_commands_file, [command])
    if len(tables_header) > 0 and df[0] == min(df):
        extraeol = [r'\hline']
        res = tableXLaTeX([tables_header], spec=spec, extra_eol=extraeol, add_end_tabular=False)
        prepend_to_file(latex_commands_file, ['\\providecommand{\\pptablesheader}{', res, '}'])

        res = tableXLaTeX([], spec=spec, add_begin_tabular=False)
        prepend_to_file(latex_commands_file, ['\\providecommand{\\pptablesfooter}{', res, '}'])
