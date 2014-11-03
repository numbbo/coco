#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for the generation of TeX tables."""
from __future__ import absolute_import

import os
from pdb import set_trace
import warnings
import numpy
from bbob_pproc import genericsettings, bestalg, toolsstats, pproc
from bbob_pproc.pptex import writeFEvals, writeFEvals2, writeFEvalsMaxPrec, tableXLaTeX, numtotext
from bbob_pproc.toolsstats import significancetest, significance_all_best_vs_other
from bbob_pproc.pproc import DataSetList
from bbob_pproc.toolsdivers import prepend_to_file, str_to_latex, strip_pathname2
from bbob_pproc.pplogloss import detf
#import bbob_pproc.pproc as pproc

"""
See Section Comparison Tables in
http://tao.lri.fr/tiki-index.php?page=BBOC+Data+presentation

"""

table_caption_one = r"""%
    Expected running time (ERT in number of function 
    evaluations) divided by the respective best ERT measured during BBOB-2009 in
    #1.
    The ERT and in braces, as dispersion measure, the half difference between 90 and 
    10\%-tile of bootstrapped run lengths appear for each algorithm and 
    """
table_caption_two1 = r"""%
    target, the corresponding best ERT
    in the first row. The different target \Df-values are shown in the top row. 
    \#succ is the number of trials that reached the (final) target $\fopt + 10^{-8}$.
    """
table_caption_two2 = r"""%
    run-length based target, the corresponding best ERT
    (preceded by the target \Df-value in \textit{italics}) in the first row. 
    \#succ is the number of trials that reached the target value of the last column.
    """
table_caption_rest = r"""%
    The median number of conducted function evaluations is additionally given in 
    \textit{italics}, if the target in the last column was never reached. 
    Entries, succeeded by a star, are statistically significantly better (according to
    the rank-sum test) when compared to all other algorithms of the table, with
    $p = 0.05$ or $p = 10^{-k}$ when the number $k$ following the star is larger
    than 1, with Bonferroni correction by the number of instances.
    """
tables_many_legend = table_caption_one + table_caption_two1 + table_caption_rest
tables_many_expensive_legend = table_caption_one + table_caption_two2 + table_caption_rest

targetsOfInterest = pproc.TargetValues((10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7))

with_table_heading = False  # in case the page is long enough

allmintarget = {}
allmedtarget = {}

infofile = os.path.join(os.path.split(__file__)[0], '..', 'benchmarkshortinfos.txt')
try:
    funInfos = {}
    f = open(infofile, 'r')
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

significance_vs_others_symbol = r"\star"
significance_vs_ref_symbol = r"\downarrow"
maxfloatrepr = 10000.
samplesize = genericsettings.simulated_runlength_bootstrap_sample_size
targetf = 1e-8
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
        #set_trace()
    else:
        #res = r"\cite{add_an_entry_for_%s_in_bbob.bib}" % algName
        res = ""
    return res

def getTopIndicesOfColumns(table, maxRank=None):
    """For each column, returns a list of the maxRank-ranked elements.

    This list may have a length larger than maxRank in the case of ties.

    """
    if maxRank is None:
        maxRank = numpy.shape(table)[0]

    ranked = [] # the length of ranked will be the number of columns in table.
    ttable = numpy.transpose(table)
    for line in ttable:
        sid = line.argsort() # returns the sorted index of the elements of line
        prevValue = None
        rank = []
        for idx in sid:
            if line[idx] == prevValue: # tie
                continue
            prevValue = line[idx]
            rank.extend(numpy.where(line == prevValue)[0])
            if len(rank) >= maxRank:
                break
        ranked.append(rank)

    return ranked

# TODO: function_headings argument need to be tested, default should be changed according to templates
def main(dictAlg, sortedAlgs, outputdir='.', verbose=True, function_targets_line=True):  # [1, 13, 101]
    """Generate one table per func with results of multiple algorithms."""
    """Difference with the first version:

    * numbers aligned using the decimal separator
    * premices for dispersion measure
    * significance test against best algorithm
    * table width...

    Takes ``targetsOfInterest`` from this file as "input argument" to compute
    the desired target values. ``targetsOfInterest`` might be configured via 
    config.
    
    """

    # TODO: method is long, terrible to read, split if possible

    if not bestalg.bestalgentries2009:
        bestalg.loadBBOB2009()

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

    for df in dictData:
        # Generate one table per df
        # first update targets for each dimension-function pair if needed:
        targets = targetsOfInterest((df[1], df[0]))            
        targetf = targets[-1]
        
        # best 2009
        refalgentry = bestalg.bestalgentries2009[df]
        refalgert = refalgentry.detERT(targets)
        refalgevals = (refalgentry.detEvals((targetf, ))[0][0])
        refalgnbruns = len(refalgevals)
        refalgnbsucc = numpy.sum(numpy.isnan(refalgevals) == False)

        # Process the data
        # The following variables will be lists of elements each corresponding
        # to an algorithm
        algnames = []
        #algdata = []
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
                txt = ("There is more than a single entry associated with "
                       "folder %s on %d-D f%d." % (sortedAlgs[n], df[0], df[1]))
                raise Exception(txt)

            entry = entries[0]
            algentries.append(entry)

            algnames.append(sortedAlgs[n])

            evals = entry.detEvals(targets)
            #tmpdata = []
            tmpdisp = []
            tmpert = []
            for i, e in enumerate(evals):
                succ = (numpy.isnan(e) == False)
                ec = e.copy() # note: here was the previous bug (changes made in e also appeared in evals !)
                ec[succ == False] = entry.maxevals[succ == False]
                ert = toolsstats.sp(ec, issuccessful=succ)[0]
                #tmpdata.append(ert/refalgert[i])
                if succ.any():
                    tmp = toolsstats.drawSP(ec[succ], entry.maxevals[succ == False],
                                           [10, 50, 90], samplesize=samplesize)[0]
                    tmpdisp.append((tmp[-1] - tmp[0])/2.)
                else:
                    tmpdisp.append(numpy.nan)
                tmpert.append(ert)
            algerts.append(tmpert)
            algevals.append(evals)
            #algdata.append(tmpdata)
            algdisp.append(tmpdisp)
            algmedmaxevals.append(numpy.median(entry.maxevals))
            algmedfinalfunvals.append(numpy.median(entry.finalfunvals))
            #algmedmaxevals.append(numpy.median(entry.maxevals)/df[0])
            #algmedfinalfunvals.append(numpy.median(entry.finalfunvals))

            algtestres.append(significancetest(refalgentry, entry, targets))

            # determine success probability for Df = 1e-8
            e = entry.detEvals((targetf ,))[0]
            algnbsucc.append(numpy.sum(numpy.isnan(e) == False))
            algnbruns.append(len(e))

        # Process over all data
        # find best values...
            
        nalgs = len(dictData[df])
        maxRank = 1 + numpy.floor(0.14 * nalgs)  # number of algs to be displayed in bold

        isBoldArray = [] # Point out the best values
        algfinaldata = [] # Store median function values/median number of function evaluations
        tmptop = getTopIndicesOfColumns(algerts, maxRank=maxRank)
        for i, erts in enumerate(algerts):
            tmp = []
            for j, ert in enumerate(erts):  # algi targetj
                tmp.append(i in tmptop[j] or (nalgs > 7 and algerts[i][j] <= 3. * refalgert[j]))
            isBoldArray.append(tmp)
            algfinaldata.append((algmedfinalfunvals[i], algmedmaxevals[i]))

        # significance test of best given algorithm against all others
        best_alg_idx = numpy.array(algerts).argsort(0)[0, :]  # indexed by target index
        significance_versus_others = significance_all_best_vs_other(algentries, targets, best_alg_idx)[0]
                
        # Create the table
        table = []
        spec = r'@{}c@{}|*{%d}{@{\,}r@{}X@{\,}}|@{}r@{}@{}l@{}' % (len(targets)) # in case StrLeft not working: replaced c@{} with l@{ }
        spec = r'@{}c@{}|*{%d}{@{}r@{}X@{}}|@{}r@{}@{}l@{}' % (len(targets)) # in case StrLeft not working: replaced c@{} with l@{ }
        extraeol = []

        # Generate header lines
        if with_table_heading:
            header = funInfos[df[1]] if funInfos else 'f%d' % df[1]
            table.append([r'\multicolumn{%d}{@{\,}c@{\,}}{{\textbf{%s}}}'
                          % (2 * len(targets) + 2, header)])
            extraeol.append('')

        if function_targets_line is True or (function_targets_line and df[1] in function_targets_line):
            if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
                curline = [r'\#FEs/D']
                for i in targetsOfInterest.labels():
                    curline.append(r'\multicolumn{2}{@{}c@{}}{%s}'
                                % i) 
                                
            else:
                curline = [r'$\Delta f_\mathrm{opt}$']
                for t in targets:
                    curline.append(r'\multicolumn{2}{@{\,}X@{\,}}{%s}'
                                % writeFEvals2(t, precision=1, isscientific=True))
#                curline.append(r'\multicolumn{2}{@{\,}X@{}|}{%s}'
#                            % writeFEvals2(targets[-1], precision=1, isscientific=True))
            curline.append(r'\multicolumn{2}{@{}l@{}}{\#succ}')
            table.append(curline)
        extraeol.append(r'\hline')
#        extraeol.append(r'\hline\arrayrulecolor{tableShade}')

        curline = [r'ERT$_{\text{best}}$'] if with_table_heading else [r'\textbf{f%d}' % df[1]] 
        if isinstance(targetsOfInterest, pproc.RunlengthBasedTargetValues):
            # write ftarget:fevals
            for i in xrange(len(refalgert[:-1])):
                temp="%.1e" %targetsOfInterest((df[1], df[0]))[i]
                if temp[-2]=="0":
                    temp=temp[:-2]+temp[-1]
                curline.append(r'\multicolumn{2}{@{}c@{}}{\textit{%s}:%s \quad}'
                                   % (temp,writeFEvalsMaxPrec(refalgert[i], 2)))
            temp="%.1e" %targetsOfInterest((df[1], df[0]))[-1]
            if temp[-2]=="0":
                temp=temp[:-2]+temp[-1]
            curline.append(r'\multicolumn{2}{@{}c@{}|}{\textit{%s}:%s }'
                               % (temp,writeFEvalsMaxPrec(refalgert[-1], 2))) 
        else:            
            # write #fevals of the reference alg
            for i in refalgert[:-1]:
                curline.append(r'\multicolumn{2}{@{}c@{}}{%s \quad}'
                                   % writeFEvalsMaxPrec(i, 2))
            curline.append(r'\multicolumn{2}{@{}c@{}|}{%s}'
                               % writeFEvalsMaxPrec(refalgert[-1], 2))
    
        # write the success ratio for the reference alg
        tmp2 = numpy.sum(numpy.isnan(refalgevals) == False) # count the nb of success
        curline.append('%d' % (tmp2))
        if tmp2 > 0:
            curline.append('/%d' % len(refalgevals))

        table.append(curline[:])
        extraeol.append('')

        #for i, gna in enumerate(zip((1, 2, 3), ('bla', 'blo', 'bli'))):
            #print i, gna, gno
            #set_trace()
        # Format data
        #if df == (5, 17):
            #set_trace()

        header = r'\providecommand{\ntables}{7}'
        for i, alg in enumerate(algnames):
            #algname, entries, irs, line, line2, succ, runs, testres1alg in zip(algnames,
            #data, dispersion, isBoldArray, isItalArray, nbsucc, nbruns, testres):
            commandname = r'\alg%stables' % numtotext(i)
#            header += r'\providecommand{%s}{{%s}{}}' % (commandname, str_to_latex(strip_pathname(alg)))
            header += r'\providecommand{%s}{\StrLeft{%s}{\ntables}}' % (commandname, str_to_latex(strip_pathname2(alg)))
            curline = [commandname + r'\hspace*{\fill}']  # each list element becomes a &-separated table entry?

            for j, tmp in enumerate(zip(algerts[i], algdisp[i],  # j is target index
                                        isBoldArray[i], algtestres[i])):
                ert, dispersion, isBold, testres = tmp
                alignment = '@{\,}X@{\,}'
                if j == len(algerts[i]) - 1:
                    alignment = '@{\,}X@{\,}|'

                data = ert/refalgert[j]
                # write star for significance against all other algorithms
                str_significance_subsup = ''
                if (len(best_alg_idx) > 0 and len(significance_versus_others) > 0 and 
                    i == best_alg_idx[j] and nbtests * significance_versus_others[j][1] < 0.05):
                    logp = -numpy.ceil(numpy.log10(nbtests * significance_versus_others[j][1]))
                    str_significance_subsup =  r"^{%s%s}" % (significance_vs_others_symbol, str(int(logp)) if logp > 1 else '')

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

                    #The conditions are now that ERT < ERT_best and
                    # all(sorted(FEvals_best) > sorted(FEvals_current)).
                    if numpy.isinf(refalgert[j]) or all(tmpevals < bestevals):
                        nbstars = -numpy.ceil(numpy.log10(nbtests * p))
                        # tmp2[-1] += r'$^{%s}$' % superscript
                        str_significance_subsup += r'_{%s%s}' % (significance_vs_ref_symbol, 
                                                                 str(int(nbstars)) if nbstars > 1 else '')
                if str_significance_subsup:
                    str_significance_subsup = '$%s$' % str_significance_subsup

                # format number in variable data
                if numpy.isnan(data):
                    curline.append(r'\multicolumn{2}{%s}{.}' % alignment)
                else:
                    if numpy.isinf(refalgert[j]):
                        curline.append(r'\multicolumn{2}{%s}{\textbf{%s}\mbox{\tiny (%s)}%s}'
                                       % (alignment,
                                          writeFEvalsMaxPrec(algerts[i][j], 2),
                                          writeFEvalsMaxPrec(dispersion, precdispersion), 
                                          str_significance_subsup))
                        continue

                    tmp = writeFEvalsMaxPrec(data, precfloat, maxfloatrepr=maxfloatrepr)
                    if data >= maxfloatrepr or data < 0.01: # either inf or scientific notation
                        if numpy.isinf(data) and j == len(algerts[i]) - 1:
                            tmp += r'\,\textit{%s}' % writeFEvalsMaxPrec(algfinaldata[i][1], 0, maxfloatrepr=maxfloatrepr)
                        else:
                            tmp = writeFEvalsMaxPrec(data, precscien, maxfloatrepr=data)
                            if isBold:
                                tmp = r'\textbf{%s}' % tmp

                        if not numpy.isnan(dispersion):
                            tmpdisp = dispersion/refalgert[j]
                            if tmpdisp >= maxfloatrepr or tmpdisp < 0.005: # TODO: hack
                                tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=tmpdisp)
                            else:
                                tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=maxfloatrepr)
                            tmp += r'\mbox{\tiny (%s)}' % tmpdisp
                        curline.append(r'\multicolumn{2}{%s}{%s%s}' % (alignment, tmp, str_significance_subsup))
                    else:
                        tmp2 = tmp.split('.', 1)
                        if len(tmp2) < 2:
                            tmp2.append('')
                        else:
                            tmp2[-1] = '.' + tmp2[-1]
                        if isBold:
                            tmp3 = []
                            for k in tmp2:
                                tmp3.append(r'\textbf{%s}' % k)
                            tmp2 = tmp3
                        if not numpy.isnan(dispersion):
                            tmpdisp = dispersion/refalgert[j]
                            if tmpdisp >= maxfloatrepr or tmpdisp < 0.01:
                                tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=tmpdisp)
                            else:
                                tmpdisp = writeFEvalsMaxPrec(tmpdisp, precdispersion, maxfloatrepr=maxfloatrepr)
                            tmp2[-1] += (r'\mbox{\tiny (%s)}' % (tmpdisp))
                        tmp2[-1] += str_significance_subsup
                        curline.extend(tmp2)
                                        
            curline.append('%d' % algnbsucc[i])
            curline.append('/%d' % algnbruns[i])
            table.append(curline)
            extraeol.append('')

        # Write table
        res = tableXLaTeX(table, spec=spec, extraeol=extraeol)
        try:
            filename = os.path.join(outputdir, 'pptables_f%03d_%02dD.tex' % (df[1], df[0]))
            f = open(filename, 'w')
            f.write(header + '\n')
            f.write(res)
            if verbose:
                print 'Wrote table in %s' % filename
        except:
            raise
        else:
            f.close()
        # TODO: return status

