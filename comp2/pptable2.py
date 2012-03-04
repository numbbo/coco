#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Rank-sum tests table on "Final Data Points".

That is, for example, using 1/#fevals(ftarget) if ftarget was reached
and -f_final otherwise as input for the rank-sum test, where obviously
the larger the better.

One table per function and dimension.

"""
from __future__ import absolute_import

import os, warnings
import numpy
import matplotlib.pyplot as plt
from bbob_pproc import bestalg, bootstrap
from bbob_pproc.pptex import tableLaTeX, tableLaTeXStar, writeFEvals2, writeFEvalsMaxPrec, writeLabels
from bbob_pproc.pproc import significancetest
from bbob_pproc.bootstrap import ranksumtest

from pdb import set_trace

targetsOfInterest = (10., 1e-1, 1e-3, 1e-5, 1e-7) # Needs to be sorted
targetf = 1e-8 # value for determining the success ratio
samplesize = 1000 # TODO change samplesize

#Get benchmark short infos: put this part in a function?
funInfos = {}
isBenchmarkinfosFound = False

figure_legend = (
    r"\ERT\ in number of function evaluations divided by the best \ERT\ measured during " +
    r"BBOB-2009 given in the respective first row and the half inter-80\%ile in brackets for " +
    r"different $\Df$ values. \#succ is the number of trials that reached the final " +
    r"target $\fopt + 10^{-8}$. " +
    r"1:\:\algorithmAshort\ is \algorithmA\ and 2:\:\algorithmBshort\ is \algorithmB. " + 
    r"Bold entries are statistically significantly better compared to the other algorithm, " + 
    r"with $p=0.05$ or $p=10^{-k}$ where $k\in\{2,3,4,\dots\}$ is the number " +
    r"following the $\star$ symbol, with Bonferroni correction of #1. " +
    r"A $\downarrow$ indicates the same tested against the best BBOB-2009. ")

infofile = os.path.join(os.path.split(__file__)[0], '..',
                        'benchmarkshortinfos.txt')

try:
    f = open(infofile,'r')
    for line in f:
        if len(line) == 0 or line.startswith('%') or line.isspace() :
            continue
        funcId, funcInfo = line[0:-1].split(None,1)
        funInfos[int(funcId)] = funcId + ' ' + funcInfo
    f.close()
    isBenchmarkinfosFound = True
except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    print 'Could not find file', infofile, \
          'Titles in scaling figures will not be displayed.'

def main(dsList0, dsList1, dimsOfInterest, outputdir, info='', verbose=True):
    """One table per dimension, modified to fit in 1 page per table."""

    #TODO: method is long, split if possible

    dictDim0 = dsList0.dictByDim()
    dictDim1 = dsList1.dictByDim()

    alg0 = set(i[0] for i in dsList0.dictByAlg().keys()).pop()[0:3]
    alg1 = set(i[0] for i in dsList1.dictByAlg().keys()).pop()[0:3]

    open(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 'a'
         ).write(r'\providecommand{\algorithmAshort}{%s}' % writeLabels(alg0) + '\n' +
                 r'\providecommand{\algorithmBshort}{%s}' % writeLabels(alg1) + '\n')

    if info:
        info = '_' + info

    dims = set.intersection(set(dictDim0.keys()), set(dictDim1.keys()))
    if not bestalg.bestalgentries2009:
        bestalg.loadBBOB2009()

    header = [r'$\Delta f$']
    for i in targetsOfInterest:
        #header.append(r'\multicolumn{2}{@{}c@{}}{$10^{%d}$}' % (int(numpy.log10(i))))
        header.append(r'\multicolumn{2}{@{}c@{}}{1e%+d}' % (int(numpy.log10(i))))
    header.append(r'\multicolumn{2}{|@{}r@{}}{\#succ}')

    for d in dimsOfInterest: # TODO set as input arguments
        table = [header]
        extraeol = [r'\hline']
        try:
            dictFunc0 = dictDim0[d].dictByFunc()
            dictFunc1 = dictDim1[d].dictByFunc()
        except KeyError:
            continue
        funcs = set.union(set(dictFunc0.keys()), set(dictFunc1.keys()))

        nbtests = len(funcs) * 2. #len(dimsOfInterest)

        for f in sorted(funcs):
            bestalgentry = bestalg.bestalgentries2009[(d, f)]
            curline = [r'${\bf f_{%d}}$' % f]
            bestalgdata = bestalgentry.detERT(targetsOfInterest)
            bestalgevals, bestalgalgs = bestalgentry.detEvals(targetsOfInterest)

            for i in bestalgdata[:-1]:
                curline.append(r'\multicolumn{2}{@{}c@{}}{%s}' % writeFEvalsMaxPrec(i, 2))
            curline.append(r'\multicolumn{2}{@{}c@{}|}{%s}' % writeFEvalsMaxPrec(bestalgdata[-1], 2))

            tmp = bestalgentry.detEvals([targetf])[0][0]
            tmp2 = numpy.sum(numpy.isnan(tmp) == False)
            curline.append('%d' % (tmp2))
            if tmp2 > 0:
                curline.append('/%d' % len(tmp))

            table.append(curline[:])
            extraeol.append('')

            rankdata0 = []  # never used

            # generate all data from ranksum test
            entries = []
            ertdata = {}
            averageevalsdata = {}
            for nb, dsList in enumerate((dictFunc0, dictFunc1)):
                try:
                    entry = dsList[f][0] # take the first DataSet, there should be only one?
                except KeyError:
                    warnings.warn('data missing for data set ' + str(nb) + ' and function ' + str(f))
                    print('*** Warning: data missing for data set ' + str(nb) + ' and function ' + str(f) + '***')
                    continue # TODO: problem here!
                ertdata[nb] = entry.detERT(targetsOfInterest)
                averageevalsdata[nb] = entry.detAverageEvals(targetsOfInterest)  # must depend on target
                entries.append(entry)

            for _t in ertdata.values():
                for _tt in _t:
                    if _tt is None:
                        raise ValueError
                    
            if len(entries) < 2: # funcion not available for *both* algorithms
                continue  # TODO: check which one is missing and make sure that what is there is displayed properly in the following
            
            testres0vs1 = significancetest(entries[0], entries[1], targetsOfInterest)
            testresbestvs1 = significancetest(bestalgentry, entries[1], targetsOfInterest)
            # 1/0 qqq
            testresbestvs0 = significancetest(bestalgentry, entries[0], targetsOfInterest)

            for nb, entry in enumerate(entries):
                if nb == 0:
                    curline = [r'1:\:\algorithmAshort\hspace*{\fill}']
                else:
                    curline = [r'2:\:\algorithmBshort\hspace*{\fill}']

                #data = entry.detERT(targetsOfInterest)
                dispersion = []
                data = []
                evals = entry.detEvals(targetsOfInterest)
                for i in evals:
                    succ = (numpy.isnan(i) == False)
                    tmp = i.copy()
                    tmp[succ==False] = entry.maxevals[numpy.isnan(i)]
                    #set_trace()
                    data.append(bootstrap.sp(tmp, issuccessful=succ)[0])
                    #if not any(succ):
                        #set_trace()
                    if any(succ):
                        tmp2 = bootstrap.drawSP(tmp[succ], tmp[succ==False],
                                                (10, 50, 90), samplesize)[0]
                        dispersion.append((tmp2[-1]-tmp2[0])/2.)
                    else:
                        dispersion.append(None)

                if nb == 0:
                    assert not isinstance(data, numpy.ndarray)
                    data0 = data[:] # TODO: check if it is not an array

                for i, dati in enumerate(data):  

                    z, p = testres0vs1[i] #TODO: there is something with the sign that I don't get
                    # assign significance flag
                    significance0vs1 = 0
                    if nb == 0:
                        istat0 = 0
                        istat1 = 1
                    else:
                        z = -z
                        istat0 = 1
                        istat1 = 0
                    # the sign of significance0vs1 shall indicate which entry is better
                    if nbtests * p < 0.05:
                        if (z > 0 and ertdata[istat0][i] <= ertdata[istat1][i] and # TODO: the conditions below should be part of significancetest() ?
                            (ertdata[istat0][i] is numpy.inf or  # comparable data: only f-values are compared for significance (they are compared for the same #evals)
                             averageevalsdata[istat0][i] < averageevalsdata[istat1][i])):  # for incomparable data the effort of the better algorithm must be lower
                            significance0vs1 = -int(numpy.ceil(numpy.log10(nbtests * p)))
                        # elif nbtests * p < 0.05 and z < 0 and z * (ertdata[istat1][i] - ertdata[istat0][i]) > 0:
                        elif (z < 0 and ertdata[istat1][i] <= ertdata[istat0][i] and 
                            (ertdata[istat1][i] is numpy.inf or  # see above
                             averageevalsdata[istat1][i] < averageevalsdata[istat0][i])):  # see above
                            significance0vs1 = int(numpy.ceil(numpy.log10(nbtests * p)))

                    isBold = significance0vs1 > 0
                    alignment = 'c'
                    if i == len(data) - 1: # last element
                        alignment = 'c|'

                    if numpy.isinf(bestalgdata[i]): # if the 2009 best did not solve the problem

                        tmp = writeFEvalsMaxPrec(float(dati), 2)
                        if not numpy.isinf(dati):
                            tmp = r'\textit{%s}' % (tmp)
                            if isBold:
                                tmp = r'\textbf{%s}' % tmp

                        if dispersion[i] and numpy.isfinite(dispersion[i]):
                            tmp += r'${\scriptscriptstyle (%s)}$' % writeFEvalsMaxPrec(dispersion[i], 1)
                        tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                      % (alignment, tmp))
                    else:
                        # Formatting
                        tmp = float(dati)/bestalgdata[i]
                        assert not numpy.isnan(tmp)
                        isscientific = False
                        if tmp >= 1000:
                            isscientific = True
                        tableentry = writeFEvals2(tmp, 2, isscientific=isscientific)
                        tableentry = writeFEvalsMaxPrec(tmp, 2)

                        if numpy.isinf(tmp) and i == len(data)-1:
                            tableentry = (tableentry 
                                          + r'\textit{%s}' % writeFEvals2(numpy.median(entry.maxevals), 2))
                            if isBold:
                                tableentry = r'\textbf{%s}' % tableentry
                            elif 11 < 3 and significance0vs1 < 0:
                                tableentry = r'\textit{%s}' % tableentry
                            if dispersion[i] and numpy.isfinite(dispersion[i]/bestalgdata[i]):
                                tableentry += r'${\scriptscriptstyle (%s)}$' % writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)
                            tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                          % (alignment, tableentry))

                        elif tableentry.find('e') > -1 or (numpy.isinf(tmp) and i != len(data) - 1):
                            if isBold:
                                tableentry = r'\textbf{%s}' % tableentry
                            elif 11 < 3 and significance0vs1 < 0:
                                tableentry = r'\textit{%s}' % tableentry
                            if dispersion[i] and numpy.isfinite(dispersion[i]/bestalgdata[i]):
                                tableentry += r'${\scriptscriptstyle (%s)}$' % writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)
                            tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                          % (alignment, tableentry))
                        else:
                            tmp = tableentry.split('.', 1)
                            if isBold:
                                tmp = list(r'\textbf{%s}' % i for i in tmp)
                            elif 11 < 3 and significance0vs1 < 0:
                                tmp = list(r'\textit{%s}' % i for i in tmp)
                            tableentry = ' & .'.join(tmp)
                            if len(tmp) == 1:
                                tableentry += '&'
                            if dispersion[i] and numpy.isfinite(dispersion[i]/bestalgdata[i]):
                                tableentry += r'${\scriptscriptstyle (%s)}$' % writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 1)

                    superscript = ''

                    if nb == 0:
                        z, p = testresbestvs0[i]
                    else:
                        z, p = testresbestvs1[i]

                    #The conditions are now that ERT < ERT_best
                    if ((nbtests * p) < 0.05 and dati - bestalgdata[i] < 0.
                        and z < 0.):
                        nbstars = -numpy.ceil(numpy.log10(nbtests * p))
                        #tmp = '\hspace{-.5ex}'.join(nbstars * [r'\star'])
                        if z > 0:
                            superscript = r'\uparrow' #* nbstars
                        else:
                            superscript = r'\downarrow' #* nbstars
                            # print z, linebest[i], line1
                        if nbstars > 1:
                            superscript += str(int(nbstars))

                    if superscript or significance0vs1:
                        s = ''
                        if significance0vs1 > 0:
                            s = '\star'
                        if significance0vs1 > 1:
                            s += str(significance0vs1)
                        s = r'$^{' + s + superscript + r'}$'

                        if tableentry.endswith('}'):
                            tableentry = tableentry[:-1] + s + r'}'
                        else:
                            tableentry += s

                    curline.append(tableentry)

                    #curline.append(tableentry)
                    #if dispersion[i] is None or numpy.isinf(bestalgdata[i]):
                        #curline.append('')
                    #else:
                        #tmp = writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 2)
                        #curline.append('(%s)' % tmp)

                tmp = entry.evals[entry.evals[:, 0] <= targetf, 1:]
                try:
                    tmp = tmp[0]
                    curline.append('%d' % numpy.sum(numpy.isnan(tmp) == False))
                except IndexError:
                    curline.append('%d' % 0)
                curline.append('/%d' % entry.nbRuns())

                table.append(curline[:])
                extraeol.append('')

            extraeol[-1] = r'\hline'
        extraeol[-1] = ''

        outputfile = os.path.join(outputdir, 'pptable2_%02dD%s.tex' % (d, info))
        spec = r'@{}c@{}|' + '*{%d}{@{}r@{}@{}l@{}}' % len(targetsOfInterest) + '|@{}r@{}@{}l@{}'
        res = r'\providecommand{\algorithmAshort}{%s}' % writeLabels(alg0) + '\n'
        res += r'\providecommand{\algorithmBshort}{%s}' % writeLabels(alg1) + '\n'
        # open(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 'a').write(res)
        
        #res += tableLaTeXStar(table, width=r'0.45\textwidth', spec=spec,
                              #extraeol=extraeol)
        res += tableLaTeX(table, spec=spec, extraeol=extraeol)
        f = open(outputfile, 'w')
        f.write(res)
        f.close()
        if verbose:
            print "Table written in %s" % outputfile

