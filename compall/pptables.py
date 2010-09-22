#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for the generation of TeX tables.
See Section Comparison Tables in
http://tao.lri.fr/tiki-index.php?page=BBOC+Data+presentation
"""

from __future__ import absolute_import

import os
from pdb import set_trace
import warnings
import numpy
from bbob_pproc import bestalg, bootstrap
from bbob_pproc.pptex import writeFEvals, writeFEvals2, writeFEvalsMaxPrec, writeLabels, tableLaTeX, numtotext
from bbob_pproc.bootstrap import prctile
from bbob_pproc.dataoutput import algPlotInfos
from bbob_pproc.pproc import DataSetList, significancetest
from bbob_pproc.pplogloss import detf

allmintarget = {}
allmedtarget = {}

funInfos = {}
isBenchmarkinfosFound = True
infofile = os.path.join(os.path.split(__file__)[0], '..', 'benchmarkshortinfos.txt')
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

maxfloatrepr = 10000.
samplesize = 3000
targetf = 1e-8

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

def sortColumns(table, maxRank=None):
    """For each column in table, returns a list of the maxRank-ranked elements.
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

def main(dictAlg, sortedAlgs, targets, outputdir='.', verbose=True):
    """Generate one table per function showing results of multiple algorithms.

    Difference with the first version:
    * numbers aligned using the decimal separator
    * premices for dispersion measure
    * significance test against best algorithm
    * table width...
    """

    # TODO: method is long, split if possible

    if not bestalg.bestalgentries2009:
        bestalg.loadBBOB2009()

    # Sort data per dimension and function
    dictData = {}
    dsListperAlg = list(dictAlg[i] for i in sortedAlgs)
    for entries in dsListperAlg:
        tmpdictdim = entries.dictByDim()
        for d in tmpdictdim:
            tmpdictfun = tmpdictdim[d].dictByFunc()
            for f in tmpdictfun:
                dictData.setdefault((d, f), []).append(tmpdictfun[f])

    nbtests = len(dictData)

    for df in dictData:
        # Generate one table per df

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
        algert = []
        algevals = []
        algdisp = []
        algnbsucc = []
        algnbruns = []
        algmedmaxevals = []
        algmedfinalfunvals = []
        algtestres = []
        algentry = []

        for n, entries in enumerate(dictData[df]):
            # the number of datasets for a given dimension and function (df)
            # should be strictly 1. TODO: find a way to warn
            # TODO: do this checking before... why wasn't it triggered by ppperprof?
            # TODO: could len(entries) be 0 as well?
            if len(entries) > 1:
                txt = ("There is more than a single entry associated with "
                      + "folder %s on %d-D f%d." % (sortedAlgs[n], df[0], df[1]))
                raise Exception(txt)

            entry = entries[0]
            algentry.append(entry)

            algnames.append(sortedAlgs[n])

            evals = entry.detEvals(targets)
            #tmpdata = []
            tmpdisp = []
            tmpert = []
            for i, e in enumerate(evals):
                succ = (numpy.isnan(e) == False)
                e[succ == False] = entry.maxevals[succ == False]
                ert = bootstrap.sp(e, issuccessful=succ)[0]
                #tmpdata.append(ert/refalgert[i])
                if succ.any():
                    tmp = bootstrap.drawSP(e[succ], entry.maxevals[succ == False],
                                           [10, 50, 90], samplesize=samplesize)[0]
                    tmpdisp.append((tmp[-1] - tmp[0])/2.)
                else:
                    tmpdisp.append(numpy.nan)
                tmpert.append(ert)
            algert.append(tmpert)
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
        if len(dictData[df]) <= 3:
            maxRank = 1
        else:
            maxRank = 3

        isBoldArray = [] # Point out the best values
        algfinaldata = [] # Store median function values/median number of function evaluations
        tmparray = sortColumns(algert, maxRank=3)
        for i, line in enumerate(algert):
            tmp = []
            for j, e in enumerate(line):
                tmp.append(i in tmparray[j] or algert[i][j] <= 3. * refalgert[j])
            isBoldArray.append(tmp)
            algfinaldata.append((algmedfinalfunvals[i], algmedmaxevals[i]))

        # Create the table
        table = []
        spec = r'@{}c@{}|*{%d}{@{\,}r@{}l@{\,}}|@{}r@{}@{}l@{}' % (len(targets))
        extraeol = []

        # Generate header lines
        if isBenchmarkinfosFound:
            header = funInfos[df[1]]
        else:
            header = 'f%d' % df[1]
        table.append([r'\multicolumn{%d}{@{\,}c@{\,}}{{\normalsize \textbf{%s}}}'
                      % (2 * len(targets) + 2, header)])
        extraeol.append('')

        curline = [r'$\Delta$ftarget']
        for t in targets[0:-1]:
            curline.append(r'\multicolumn{2}{@{\,}c@{\,}}{%s}'
                           % writeFEvals2(t, precision=1, isscientific=True))
        curline.append(r'\multicolumn{2}{@{\,}c@{}|}{%s}'
                       % writeFEvals2(targets[-1], precision=1, isscientific=True))
        curline.append(r'\multicolumn{2}{@{}l@{}}{\#succ}')
        table.append(curline)
        extraeol.append('')

        curline = [r'ERT$_{\text{best}}$']
        for i in refalgert[0:-1]:
            curline.append(r'\multicolumn{2}{@{\,}c@{\,}}{%s}'
                           % writeFEvalsMaxPrec(float(i), 2))
        curline.append(r'\multicolumn{2}{@{\,}c@{\,}|}{%s}'
                       % writeFEvalsMaxPrec(float(refalgert[-1]), 2))
        curline.append('%d' % refalgnbsucc)
        if refalgnbsucc:
            curline.append('/%d' % refalgnbruns)
        #curline.append(curline[0])
        table.append(curline)
        extraeol.append(r'\hline')

        #for i, gna in enumerate(zip((1, 2, 3), ('bla', 'blo', 'bli'))):
            #print i, gna, gno
            #set_trace()
        # Format data
        #if df == (5, 17):
            #set_trace()

        header = ''
        for i, alg in enumerate(algnames):
        #algname, entries, irs, line, line2, succ, runs, testres1alg in zip(algnames,
                   #data, dispersion, isBoldArray, isItalArray, nbsucc, nbruns, testres):
            commandname = r'\alg%stables' % numtotext(i)
            header += r'\providecommand{%s}{%s}' % (commandname, writeLabels(alg))
            curline = [commandname + r'\hspace*{\fill}']

            for j, tmp in enumerate(zip(algert[i], algdisp[i],
                                        isBoldArray[i], algtestres[i])):
                ert, dispersion, isBold, testres = tmp

                alignment = '@{\,}c@{\,}'
                if j == len(algert[i]) - 1:
                    alignment = '@{\,}c@{\,}|'

                data = ert/refalgert[j]

                # format number in variable data
                if numpy.isnan(data):
                    curline.append(r'\multicolumn{2}{%s}{.}' % alignment)
                else:
                    if numpy.isinf(refalgert[j]):
                        curline.append(r'\multicolumn{2}{%s}{\textbf{%s}${\scriptscriptstyle (%s)}$}'
                                       % (alignment,
                                          writeFEvalsMaxPrec(algert[i][j], 2),
                                          writeFEvalsMaxPrec(dispersion, 2)))
                        continue

                    tmp = writeFEvalsMaxPrec(data, 2, maxfloatrepr=maxfloatrepr)
                    if data >= maxfloatrepr: # either inf or scientific notation
                        if numpy.isinf(data) and j == len(algert[i]) - 1:
                            tmp += r'\,\textit{%s}' % writeFEvalsMaxPrec(algfinaldata[i][1], 0)
                        else:
                            if isBold:
                                tmp = r'\textbf{%s}' % tmp

                        if not numpy.isnan(dispersion):
                            tmp += r'${\scriptscriptstyle (%s)}$' % writeFEvalsMaxPrec(dispersion/refalgert[j], 2)
                        curline.append(r'\multicolumn{2}{%s}{%s}' % (alignment, tmp))
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
                            tmp2[-1] += (r'${\scriptscriptstyle (%s)}$' % writeFEvalsMaxPrec(dispersion/refalgert[j], 2))

                        z, p = testres
                        if data < 1. and not numpy.isinf(refalgert[j]):
                            tmpevals = algevals[i][j].copy()
                            tmpevals[numpy.isnan(tmpevals)] = algentry[i].maxevals[numpy.isnan(tmpevals)]
                            bestevals = refalgentry.detEvals([targets[j]])
                            bestevals, bestalgalg = (bestevals[0][0], bestevals[1][0])
                            bestevals[numpy.isnan(bestevals)] = refalgentry.maxevals[bestalgalg][numpy.isnan(bestevals)]
                            tmpevals = numpy.array(sorted(tmpevals))[0:min(len(tmpevals), len(bestevals))]
                            bestevals = numpy.array(sorted(bestevals))[0:min(len(tmpevals), len(bestevals))]

                        #The conditions are now that ERT < ERT_best and
                        # all(sorted(FEvals_best) > sorted(FEvals_current)).
                        if ((nbtests * p) < 0.05 and data < 1.
                            and z < 0.
                            and (numpy.isinf(refalgert[j])
                                 or all(tmpevals < bestevals))):
                            nbstars = -numpy.ceil(numpy.log10(nbtests * p))
                            superscript = r'\downarrow' #* nbstars
                            if nbstars > 1:
                                superscript += str(int(nbstars))
                            tmp2[-1] += r'$^{%s}$' % superscript

                        curline.extend(tmp2)

            curline.append('%d' % algnbsucc[i])
            curline.append('/%d' % algnbruns[i])
            table.append(curline)
            extraeol.append('')

        # Write table
        res = tableLaTeX(table, spec=spec, extraeol=extraeol)
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

