#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for the generation of TeX tables."""
from __future__ import absolute_import

import os, sys
from pdb import set_trace
import warnings
import numpy
from .. import genericsettings, bestalg, toolsstats, pproc, ppfigparam, testbedsettings
from ..pptex import writeFEvals2, writeFEvalsMaxPrec, tableXLaTeX, numtotext
from ..toolsstats import significancetest, significance_all_best_vs_other
from ..toolsdivers import str_to_latex, strip_pathname1, replace_in_file, get_version_label

"""
See Section Comparison Tables in
http://tao.lri.fr/tiki-index.php?page=BBOC+Data+presentation

"""


def get_table_caption():
    """ Sets table caption, based on the testbedsettings.current_testbed
        and genericsettings.runlength_based_targets.
        
        TODO: \hvref and \fopt should be defined via the current_testbed, 
        preferably with a single latex command. 
    """

    table_caption_one = r"""%
        Average running time (\aRT\ in number of function 
        evaluations) divided by the respective best \aRT\ measured during BBOB-2009 in
        #1.
        The \aRT\ and in braces, as dispersion measure, the half difference between 
        10 and 90\%-tile of bootstrapped run lengths appear for each algorithm and 
        """
    table_caption_two1 = r"""%
        target, the corresponding best \aRT\
        in the first row. The different target \Df-values are shown in the top row.
        \#succ is the number of trials that reached the (final) target
        $\fopt + """ + testbedsettings.current_testbed.hardesttargetlatex + r"""$.
        """
    table_caption_two2 = r"""%
        run-length based target, the corresponding best \aRT\
        (preceded by the target \Df-value in \textit{italics}) in the first row. 
        \#succ is the number of trials that reached the target value of the last column.
        """
    table_caption_one_bi = r"""%
        Average runtime (\aRT) to reach given targets, measured
        in number of function evaluations, in #1. For each function, the \aRT\ 
        and, in braces as dispersion measure, the half difference between 10 and 
        90\%-tile of (bootstrapped) runtimes is shown for the different
        target \DI-values as shown in the top row. 
        \#succ is the number of trials that reached the last target
        $\hvref + """ + testbedsettings.current_testbed.hardesttargetlatex + r"""$.
        """
    table_caption_rest = (r"""%
        The median number of conducted function evaluations is additionally given in 
        \textit{italics}, if the target in the last column was never reached. 
        Entries, succeeded by a star, are statistically significantly better (according to
        the rank-sum test) when compared to all other algorithms of the table, with
        $p = 0.05$ or $p = 10^{-k}$ when the number $k$ following the star is larger
        than 1, with Bonferroni correction of #2. """ +
                          (r"""A $\downarrow$ indicates the same tested against the best
        algorithm of BBOB-2009. """
                           if not (testbedsettings.current_testbed.name in [testbedsettings.testbed_name_bi, testbedsettings.testbed_name_bi_ext, testbedsettings.testbed_name_largescale])
                           else "") + r"""Best results are printed in bold.
        """ + r"""\cocoversion""")

    if testbedsettings.current_testbed.name in [testbedsettings.testbed_name_bi,
                                                testbedsettings.testbed_name_bi_ext,
                                                testbedsettings.testbed_name_largescale]:
        # Wassim: TODO: the large-scale suite is, for now, put here. The case where there is no reference algorithm should be treated for the tables too
        # NOTE: no runlength-based targets supported yet
        table_caption = table_caption_one_bi + table_caption_rest
    elif testbedsettings.current_testbed.name in [testbedsettings.testbed_name_single]:
        if genericsettings.runlength_based_targets:
            table_caption = table_caption_one + table_caption_two2 + table_caption_rest
        else:
            table_caption = table_caption_one + table_caption_two1 + table_caption_rest
    else:
        warnings.warn("Current settings do not support pptables caption.")

    return table_caption


with_table_heading = False  # in case the page is long enough

allmintarget = {}
allmedtarget = {}

significance_vs_others_symbol = r"\star"
significance_vs_others_symbol_html = r"&#9733;"
significance_vs_ref_symbol = r"\downarrow"
significance_vs_ref_symbol_html = r"&darr;"
maxfloatrepr = 10000.
samplesize = genericsettings.simulated_runlength_bootstrap_sample_size
precfloat = 2
precscien = 2
precdispersion = 1  # significant digits for dispersion


def cite(algName, isNoisefree, isNoisy):
    """Returns the citation key associated to the algorithm name.

    Hard coded while no other solution is found.

    """
    res = []
    # The names of the algorithms must correspond to the name of the folder
    # containing the data. The citations keys must be in bbob.bib.
    if isNoisefree:
        if algName == "ALPS-GA":
            res.append("Hornby:2009")
        if algName in ("AMaLGaM IDEA", "iAMaLGaM IDEA"):
            res.append("DBLP:conf/gecco/BosmanGT09")
        if algName == "BayEDAcG":
            res.append("DBLP:conf/gecco/Gallagher09")
        if algName == "BFGS":
            res.append("DBLP:conf/gecco/Ros09")
        if algName == "Cauchy EDA":
            res.append("DBLP:conf/gecco/Posik09")
        if algName == "BIPOP-CMA-ES":
            res.append("DBLP:conf/gecco/Hansen09")
        if algName == "(1+1)-CMA-ES":
            res.append("DBLP:conf/gecco/AugerH09")
        if algName == "DASA":
            res.append("DBLP:conf/gecco/KorosecS09")
        if algName == "DEPSO":
            res.append("DBLP:conf/gecco/Garcia-NietoAA09")
        if algName == "DIRECT":
            res.append("DBLP:conf/gecco/Posik09a")
        if algName == "EDA-PSO":
            res.append("DBLP:conf/gecco/El-AbdK09")
        if algName == "CMA-EGS":
            res.append("Finck:2009")
        if algName == "G3-PCX":
            res.append("DBLP:conf/gecco/Posik09b")
        if algName == "simple GA":
            res.append("DBLP:conf/gecco/Nicolau09")
        if algName == "GLOBAL":
            res.append("Pal:2009a")
        if algName in ("LSfminbnd", "LSstep"):
            res.append("DBLP:conf/gecco/Posik09c")
        if algName == "MA-LS-Chain":
            res.append("DBLP:conf/gecco/MolinaLH09")
        if algName == "MCS":
            res.append("Huyer:2009b")
        if algName == "NELDER (Han)":
            res.append("DBLP:conf/gecco/Hansen09b")
        if algName == "NELDER (Doe)":
            res.append("DBLP:conf/gecco/DoerrFSW09")
        if algName in ("NEWUOA", "avg NEWUOA", "full NEWUOA"):
            res.append("DBLP:conf/gecco/Ros09b")
        if algName == "(1+1)-ES":
            res.append("DBLP:conf/gecco/Auger09")
        if algName == "POEMS":
            res.append("DBLP:conf/gecco/Kubalik09a")
        if algName == "PSO":
            res.append("DBLP:conf/gecco/El-AbdK09a")
        if algName == "PSO\_Bounds":
            res.append("DBLP:conf/gecco/El-AbdK09b")
        if algName == "Monte Carlo":
            res.append("DBLP:conf/gecco/AugerR09")
        if algName == "Rosenbrock":
            res.append("DBLP:conf/gecco/Posik09d")
        if algName == "IPOP-SEP-CMA-ES":
            res.append("DBLP:conf/gecco/Ros09d")
        if algName == "VNS (Garcia)":
            res.append("DBLP:conf/gecco/Garcia-MartinezL09")
    if isNoisy:
        if algName == "ALPS-GA":
            res.append("Hornby:2009a")
        elif algName in ("AMaLGaM IDEA", "iAMaLGaM IDEA"):
            res.append("DBLP:conf/gecco/BosmanGT09a")
        elif algName in ("avg NEWUOA", "full NEWUOA", "NEWUOA"):
            res.append("DBLP:conf/gecco/Ros09c")
        elif algName == "BayEDAcG":
            res.append("DBLP:conf/gecco/Gallagher09a")
        elif algName == "BFGS":
            res.append("DBLP:conf/gecco/Ros09a")
        elif algName == "BIPOP-CMA-ES":
            res.append("DBLP:conf/gecco/Hansen09a")
        elif algName == "(1+1)-CMA-ES":
            res.append("DBLP:conf/gecco/AugerH09a")
        elif algName == "DASA":
            res.append("DBLP:conf/gecco/KorosecS09a")
        elif algName == "DEPSO":
            res.append("DBLP:conf/gecco/Garcia-NietoAA09a")
        elif algName == "EDA-PSO":
            res.append("DBLP:conf/gecco/El-AbdK09")
        elif algName == "CMA-EGS":
            res.append("Finck:2009a")
        elif algName == "GLOBAL":
            res.append("Pal:2009")
        elif algName == "MA-LS-Chain":
            res.append("DBLP:conf/gecco/MolinaLH09a")
        elif algName == "MCS":
            res.append("Huyer:2009a")
        elif algName == "(1+1)-ES":
            res.append("DBLP:conf/gecco/Auger09a")
        elif algName == "PSO":
            res.append("DBLP:conf/gecco/El-AbdK09a")
        elif algName == "PSO\_Bounds":
            res.append("DBLP:conf/gecco/El-AbdK09b")
        elif algName == "Monte Carlo":
            res.append("DBLP:conf/gecco/AugerR09a")
        elif algName == "IPOP-SEP-CMA-ES":
            res.append("DBLP:conf/gecco/Ros09e")
        elif algName == "SNOBFIT":
            res.append("Huyer:2009")
        elif algName == "VNS (Garcia)":
            res.append("DBLP:conf/gecco/Garcia-MartinezL09a")

    if res:
        res = r"\cite{%s}" % (", ".join(res))
        # set_trace()
    else:
        # res = r"\cite{add_an_entry_for_%s_in_bbob.bib}" % algName
        res = ""
    return res


def getTopIndicesOfColumns(table, maxRank=None):
    """For each column, returns a list of the maxRank-ranked elements.

    This list may have a length larger than maxRank in the case of ties.

    """
    if maxRank is None:
        maxRank = numpy.shape(table)[0]

    ranked = []  # the length of ranked will be the number of columns in table.
    ttable = numpy.transpose(table)
    for line in ttable:
        sid = line.argsort()  # returns the sorted index of the elements of line
        prevValue = None
        rank = []
        for idx in sid:
            if line[idx] == prevValue:  # tie
                continue
            prevValue = line[idx]
            rank.extend(numpy.where(line == prevValue)[0])
            if len(rank) >= maxRank:
                break
        ranked.append(rank)

    return ranked


# TODO: function_headings argument need to be tested, default should be changed according to templates
def main(dictAlg, sortedAlgs, outputdir='.', function_targets_line=True):  # [1, 13, 101]
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
    targetsOfInterest = testbed.pptablemany_targetsOfInterest

    bestalgentries = bestalg.load_reference_algorithm(testbed.reference_algorithm_filename)

    # Sort data per dimension and function
    dictData = {}
    dsListperAlg = list(dictAlg[i] for i in sortedAlgs)
    for n, entries in enumerate(dsListperAlg):
        tmpdictdim = entries.dictByDim()
        for d in tmpdictdim:
            tmpdictfun = tmpdictdim[d].dictByFunc()
            for f in tmpdictfun:
                dictData.setdefault((d, f), {})[n] = tmpdictfun[f]

    nbtests = len(dictData)

    funInfos = ppfigparam.read_fun_infos()

    firstFunction = True

    for df in sorted(dictData):
        # Generate one table per df
        # first update targets for each dimension-function pair if needed:
        targets = targetsOfInterest((df[1], df[0]))
        if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
            targetf = targets[-1]
        else:
            targetf = testbed.pptable_ftarget

        # reference algorithm
        if bestalgentries:
            refalgentry = bestalgentries[df]
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

        for n in sorted(dictData[df].keys()):
            entries = dictData[df][n]
            # the number of datasets for a given dimension and function (df)
            # should be strictly 1. TODO: find a way to warn
            # TODO: do this checking before... why wasn't it triggered by ppperprof?
            if len(entries) > 1:
                print entries
                txt = ("There is more than a single entry associated with "
                       "folder %s on %d-D f%d." % (sortedAlgs[n], df[0], df[1]))
                raise Exception(txt)

            entry = entries[0]
            algentries.append(entry)

            algnames.append(sortedAlgs[n])

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
                                            [10, 50, 90], samplesize=samplesize)[0]
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

            if bestalgentries:
                algtestres.append(significancetest(refalgentry, entry, targets))

            # determine success probability for Df = 1e-8
            e = entry.detEvals((targetf,))[0]
            algnbsucc.append(numpy.sum(numpy.isnan(e) == False))
            algnbruns.append(len(e))

        # Process over all data
        # find best values...

        nalgs = len(dictData[df])
        maxRank = 1 + numpy.floor(0.14 * nalgs)  # number of algs to be displayed in bold

        isBoldArray = []  # Point out the best values
        algfinaldata = []  # Store median function values/median number of function evaluations
        tmptop = getTopIndicesOfColumns(algerts, maxRank=maxRank)
        for i, erts in enumerate(algerts):
            tmp = []
            for j, ert in enumerate(erts):  # algi targetj
                tmp.append(i in tmptop[j] or (bestalgentries and nalgs > 7 and algerts[i][j] <= 3. * refalgert[j]))
            isBoldArray.append(tmp)
            algfinaldata.append((algmedfinalfunvals[i], algmedmaxevals[i]))

        # significance test of best given algorithm against all others
        best_alg_idx = numpy.array(algerts).argsort(0)[0, :]  # indexed by target index
        significance_versus_others = significance_all_best_vs_other(algentries, targets, best_alg_idx)[0]

        # Create the table
        table = []
        tableHtml = []
        spec = r'@{}c@{}|*{%d}{@{\,}r@{}X@{\,}}|@{}r@{}@{}l@{}' % (
        len(targetsOfInterest))  # in case StrLeft not working: replaced c@{} with l@{ }
        spec = r'@{}c@{}|*{%d}{@{}r@{}X@{}}|@{}r@{}@{}l@{}' % (
        len(targetsOfInterest))  # in case StrLeft not working: replaced c@{} with l@{ }
        extraeol = []

        # Generate header lines
        if with_table_heading:
            header = funInfos[df[1]] if df[1] in funInfos.keys() else 'f%d' % df[1]
            table.append([r'\multicolumn{%d}{@{\,}c@{\,}}{{\textbf{%s}}}'
                          % (2 * len(targetsOfInterest) + 2, header)])
            extraeol.append('')

        # generate line with displayed quality indicator and targets:
        if function_targets_line is True or (function_targets_line and df[1] in function_targets_line):
            if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
                curline = [r'\#FEs/D']
                counter = 1
                for i in targetsOfInterest.labels():
                    curline.append(r'\multicolumn{2}{@{}c@{}}{%s}' % i)
                    counter += 1
            else:
                if (testbed.name == testbedsettings.testbed_name_bi):
                    curline = [r'$\Df$']
                else:
                    curline = [r'$\Delta f_\mathrm{opt}$']
                counter = 1
                for t in targets:
                    curline.append(r'\multicolumn{2}{@{\,}l@{\,}}{%s}'
                                   % writeFEvals2(t, precision=1, isscientific=True))
                    counter += 1
                #                curline.append(r'\multicolumn{2}{@{\,}l@{}|}{%s}'
                #                            % writeFEvals2(targetsOfInterest[-1], precision=1, isscientific=True))
            if (testbed.name == testbedsettings.testbed_name_bi):
                curline.append(r'\multicolumn{2}{|@{}l@{}}{\begin{rotate}{30}\#succ\end{rotate}}')
            else:
                curline.append(r'\multicolumn{2}{|@{}l@{}}{\#succ}')
            table.append(curline)

        # do the same for the HTML output, but all the time:
        curlineHtml = []
        if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
            curlineHtml = ['<thead>\n<tr>\n<th>#FEs/D<br>REPLACEH</th>\n']
            counter = 1
            for i in targetsOfInterest.labels():
                curlineHtml.append('<td>%s<br>REPLACE%d</td>\n' % (i, counter))
                counter += 1
        else:
            if (testbed.name == testbedsettings.testbed_name_bi):
                curlineHtml = ['<thead>\n<tr>\n<th>&#916; HV<sub>ref</sub><br>REPLACEH</th>\n']
            else:
                curlineHtml = ['<thead>\n<tr>\n<th>&#916; f<sub>opt</sub><br>REPLACEH</th>\n']
            counter = 1
            for t in targets:
                curlineHtml.append(
                    '<td>%s<br>REPLACE%d</td>\n' % (writeFEvals2(t, precision=1, isscientific=True), counter))
                counter += 1
        curlineHtml.append('<td>#succ<br>REPLACEF</td>\n</tr>\n</thead>\n')
 
        extraeol.append(r'\hline')
        #        extraeol.append(r'\hline\arrayrulecolor{tableShade}')
        
        # line with function name and potential aRT values of bestalg
        curline = [r'\textbf{f%d}' % df[1]]
        replaceValue = '<b>f%d</b>' % df[1]
        curlineHtml = [item.replace('REPLACEH', replaceValue) for item in curlineHtml]
        
        if bestalgentries:
            if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
                # write ftarget:fevals
                counter = 1
                for i in xrange(len(refalgert[:-1])):
                    temp = "%.1e" % targetsOfInterest((df[1], df[0]))[i]
                    if temp[-2] == "0":
                        temp = temp[:-2] + temp[-1]
                    curline.append(r'\multicolumn{2}{@{}c@{}}{\textit{%s}:%s \quad}'
                                   % (temp, writeFEvalsMaxPrec(refalgert[i], 2)))
                    replaceValue = '<i>%s</i>:%s' % (temp, writeFEvalsMaxPrec(refalgert[i], 2))
                    curlineHtml = [item.replace('REPLACE%d' % counter, replaceValue) for item in curlineHtml]
                    counter += 1

                temp = "%.1e" % targetsOfInterest((df[1], df[0]))[-1]
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

        else:  # if not bestalgentries
            curline.append(r'\multicolumn{%d}{@{}c@{}|}{} & ' % (2 * (len(targetsOfInterest))))
            for counter in range(1, len(targetsOfInterest) + 1):
                curlineHtml = [item.replace('REPLACE%d' % counter, '&nbsp;') for item in curlineHtml]
            curlineHtml = [item.replace('REPLACEF', '&nbsp;') for item in curlineHtml]

        curlineHtml = [i.replace('$\infty$', '&infin;') for i in curlineHtml]
        table.append(curline[:])
        tableHtml.extend(curlineHtml[:])
        tableHtml.append('<tbody>\n')
        extraeol.append('')

        header = r'\providecommand{\ntables}{%d}' % len(targetsOfInterest)
        for i, alg in enumerate(algnames):
            tableHtml.append('<tr>\n')
            # algname, entries, irs, line, line2, succ, runs, testres1alg in zip(algnames,
            # data, dispersion, isBoldArray, isItalArray, nbsucc, nbruns, testres):
            commandname = r'\alg%stables' % numtotext(i)
            #            header += r'\providecommand{%s}{{%s}{}}' % (commandname, str_to_latex(strip_pathname(alg)))
            header += r'\providecommand{%s}{\StrLeft{%s}{\ntables}}' % (commandname, str_to_latex(strip_pathname1(alg)))
            curline = [commandname + r'\hspace*{\fill}']  # each list element becomes a &-separated table entry?
            curlineHtml = ['<th>%s</th>\n' % str_to_latex(strip_pathname1(alg))]

            zipToEnumerate = zip(algerts[i], algdisp[i], isBoldArray[i], algtestres[i]) if bestalgentries else zip(
                algerts[i], algdisp[i], isBoldArray[i])

            for j, tmp in enumerate(zipToEnumerate):  # j is target index
                if bestalgentries:
                    ert, dispersion, isBold, testres = tmp
                else:
                    ert, dispersion, isBold = tmp

                alignment = '@{\,}l@{\,}'
                if j == len(algerts[i]) - 1:
                    alignment = '@{\,}l@{\,}|'

                data = ert / refalgert[j] if bestalgentries else ert
                # write star for significance against all other algorithms
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

                if bestalgentries:
                    # moved out of the above else: this was a bug!?
                    z, p = testres
                    if (nbtests * p) < 0.05 and data < 1. and z < 0.:
                        if not numpy.isinf(refalgert[j]):
                            tmpevals = algevals[i][j].copy()
                            tmpevals[numpy.isnan(tmpevals)] = algentries[i].maxevals[numpy.isnan(tmpevals)]
                            bestevals = refalgentry.detEvals(targets)
                            bestevals, bestalgalg = (bestevals[0][0], bestevals[1][0])
                            bestevals[numpy.isnan(bestevals)] = refalgentry.maxevals[bestalgalg][numpy.isnan(bestevals)]
                            tmpevals = numpy.array(sorted(tmpevals))[0:min(len(tmpevals), len(bestevals))]
                            bestevals = numpy.array(sorted(bestevals))[0:min(len(tmpevals), len(bestevals))]

                        # The conditions are now that aRT < aRT_best and
                        # all(sorted(FEvals_best) > sorted(FEvals_current)).
                        if numpy.isinf(refalgert[j]) or all(tmpevals < bestevals):
                            nbstars = -numpy.ceil(numpy.log10(nbtests * p))
                            # tmp2[-1] += r'$^{%s}$' % superscript
                            str_significance_subsup += r'_{%s%s}' % (significance_vs_ref_symbol,
                                                                     str(int(nbstars)) if nbstars > 1 else '')
                            str_significance_subsup_html = '<sub>%s%s</sub>' % (significance_vs_ref_symbol_html,
                                                                                str(int(
                                                                                    nbstars)) if nbstars > 1 else '')
                if str_significance_subsup:
                    str_significance_subsup = '$%s$' % str_significance_subsup

                # format number in variable data
                if numpy.isnan(data):
                    curline.append(r'\multicolumn{2}{%s}{.}' % alignment)
                    curlineHtml.append('<td>&nbsp;</td>')
                else:
                    if bestalgentries and numpy.isinf(refalgert[j]):
                        tableentry = r'\textbf{%s}' % writeFEvalsMaxPrec(algerts[i][j], 2)
                        tableentryHtml = '<b>%s</b>' % writeFEvalsMaxPrec(algerts[i][j], 2)
                        if dispersion and numpy.isfinite(dispersion):
                            tableentry += r'\mbox{\tiny (%s)}' % writeFEvalsMaxPrec(dispersion, precdispersion)
                            tableentryHtml += ' (%s)' % writeFEvalsMaxPrec(dispersion, precdispersion)

                        curline.append(r'\multicolumn{2}{%s}{%s}%s'
                                       % (alignment,
                                          tableentry,
                                          str_significance_subsup))

                        curlineHtml.append('<td sorttable_customkey=\"%f\">%s%s</td>\n'
                                           % (algerts[i][j],
                                              tableentryHtml,
                                              str_significance_subsup_html))
                        continue

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
                            tmpdisp = dispersion / refalgert[j] if bestalgentries else dispersion
                            if tmpdisp >= maxfloatrepr or tmpdisp < 0.005:  # TODO: hack
                                tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=tmpdisp)
                            else:
                                tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=maxfloatrepr)
                            tmp += r'\mbox{\tiny (%s)}' % tmpdisp
                            tmpHtml += ' (%s)' % tmpdisp
                        curline.append(r'\multicolumn{2}{%s}{%s%s}' % (alignment, tmp, str_significance_subsup))
                        if (numpy.isinf(sortKey)):
                            sortKey = sys.maxint
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
                            tmpdisp = dispersion / refalgert[j] if bestalgentries else dispersion
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
        res = tableXLaTeX(table, spec=spec, extraeol=extraeol)
        try:
            filename = os.path.join(outputdir, 'pptables_f%03d_%02dD.tex' % (df[1], df[0]))
            f = open(filename, 'w')
            f.write(header + '\n')
            f.write(res)

            res = ("").join(str(item) for item in tableHtml)
            res = '\n<table class=\"sortable\" style=\"width:800px \">\n%s</table>\n<p/>\n' % res
            if firstFunction:
                res = '<br><p><b>%d-D</b></p>' % df[0] + res
                firstFunction = False

            if True:
                filename = os.path.join(outputdir, genericsettings.pptables_file_name + '.html')
                lines = []
                with open(filename) as infile:
                    for line in infile:
                        if '<!--' + 'pptablesHtml' + '-->' in line:
                            lines.append(res)
                        lines.append(line)

                with open(filename, 'w') as outfile:
                    for line in lines:
                        outfile.write(line)
                
                replace_in_file(filename, '??COCOVERSION??', '<br />Data produced with COCO %s' % (get_version_label(None)))


            if genericsettings.verbose:
                print 'Wrote table in %s' % filename
        except:
            raise
        else:
            f.close()
            # TODO: return status
