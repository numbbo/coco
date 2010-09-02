#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for generating tables used by rungeneric1.py."""

from __future__ import absolute_import

import os
import numpy
import matplotlib.pyplot as plt
from bbob_pproc import bestalg, bootstrap
from bbob_pproc.pptex import tableLaTeX, tableLaTeXStar, writeFEvals2, writeFEvalsMaxPrec
from bbob_pproc.pproc import significancetest
from bbob_pproc.bootstrap import ranksums

from pdb import set_trace

targetsOfInterest = (10., 1., 1e-1, 1e-3, 1e-5, 1e-7) # targets of the table
targetf = 1e-8 # value for determining the success ratio
samplesize = 3000 # TODO: change samplesize

def main(dsList, dimsOfInterest, outputdir, info='', verbose=True):
    """Generate a table of ERT/ERTbest vs Deltaf: 1 per function and dimension

    Rank-sum tests table on "Final Data Points" for only one algorithm.
    that is, for example, using 1/#fevals(ftarget) if ftarget was reached and
    -f_final otherwise as input for the rank-sum test, where obviously the
    larger the better.

    """

    #TODO: check that it works for any reference algorithm?
    #in the following the reference algorithm is the one given in
    #bestalg.bestalgentries which is the virtual best of BBOB
    #TODO: the method is long, split it if possible

    dictDim = dsList.dictByDim()

    if info:
        info = '_' + info
        # insert a separator between the default file name and the additional
        # information string.

    dims = set(dictDim.keys())
    if not bestalg.bestalgentries2009:
        bestalg.loadBBOB2009()

    header = [r'$\Delta f$']
    for i in targetsOfInterest:
        header.append(r'\multicolumn{2}{@{}c@{}}{1e%+d}' % (int(numpy.log10(i))))
    header.append(r'\multicolumn{2}{|@{}r@{}}{\#succ}')

    for d in dimsOfInterest:
        # the table variable will store all data (in the form of a list of
        # list of strings) for a result table.
        table = [header]
        extraeol = [r'\hline']
        try:
            dictFunc = dictDim[d].dictByFunc()
        except KeyError:
            continue
        funcs = set(dictFunc.keys())
        nbtests = float(len(funcs)) # #funcs tests times one algorithm

        for f in sorted(funcs):
            bestalgentry = bestalg.bestalgentries2009[(d, f)]
            curline = [r'${\bf f_{%d}}$' % f]
            bestalgdata = bestalgentry.detERT(targetsOfInterest)
            bestalgevals, bestalgalgs = bestalgentry.detEvals(targetsOfInterest)

            # write #fevals of the reference alg
            for i in bestalgdata[:-1]:
                curline.append(r'\multicolumn{2}{@{}c@{}}{%s}'
                               % writeFEvalsMaxPrec(i, 2))
            curline.append(r'\multicolumn{2}{@{}c@{}|}{%s}'
                           % writeFEvalsMaxPrec(bestalgdata[-1], 2))

            # write the success ratio for the reference alg
            tmp = bestalgentry.detEvals([targetf])[0][0]
            tmp2 = numpy.sum(numpy.isnan(tmp) == False) # count the nb of success
            curline.append('%d' % (tmp2))
            if tmp2 > 0:
                curline.append('/%d' % len(tmp))

            table.append(curline[:])
            extraeol.append('')

            # generate all data for ranksum test
            entry = dictFunc[f][0] # take the first element
            ertdata = entry.detERT(targetsOfInterest)

            testresbestvs1 = significancetest(bestalgentry, entry, targetsOfInterest)

            #for nb, entry in enumerate(entries):
            #curline = [r'\algshort\hspace*{\fill}']
            curline = ['']
            #data = entry.detERT(targetsOfInterest)
            evals = entry.detEvals(targetsOfInterest)
            dispersion = []
            data = []
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

            for i, ert in enumerate(data):
                alignment = 'c'
                if i == len(data) - 1: # last element
                    alignment = 'c|'

                nbstars = 0
                z, p = testresbestvs1[i]
                if ert - bestalgdata[i] < 0. and not numpy.isinf(bestalgdata[i]):
                    evals = entry.detEvals([targetsOfInterest[i]])[0]
                    evals[numpy.isnan(evals)] = entry.maxevals[numpy.isnan(evals)]
                    bestevals = bestalgentry.detEvals([targetsOfInterest[i]])
                    bestevals, bestalgalg = (bestevals[0][0], bestevals[1][0])
                    bestevals[numpy.isnan(bestevals)] = bestalgentry.maxevals[bestalgalg][numpy.isnan(bestevals)]
                    evals = numpy.array(sorted(evals))[0:min(len(evals), len(bestevals))]
                    bestevals = numpy.array(sorted(bestevals))[0:min(len(evals), len(bestevals))]

                #The conditions are now that ERT < ERT_best and
                # all(sorted(FEvals_best) > sorted(FEvals_current)).
                if ((nbtests * p) < 0.05 and ert - bestalgdata[i] < 0.
                    and z < 0.
                    and (numpy.isinf(bestalgdata[i])
                         or all(evals < bestevals))):
                    nbstars = -numpy.ceil(numpy.log10(nbtests * p))

                if numpy.isinf(bestalgdata[i]): # if the best did not solve the problem
                    isBold = False
                    if nbstars > 0:
                       isBold = True

                    tmp = writeFEvalsMaxPrec(float(ert), 2)
                    if not numpy.isinf(ert):
                        tmp = r'\textit{%s}' % (tmp)
                        if isBold:
                            tmp = r'\textbf{%s}' % tmp

                    tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                  % (alignment, tmp))
                else:
                    # Formatting
                    tmp = float(ert)/bestalgdata[i]
                    assert not numpy.isnan(tmp)
                    tableentry = writeFEvalsMaxPrec(tmp, 2)

                    isBold = False
                    if nbstars > 0:
                       isBold = True

                    if numpy.isinf(tmp) and i == len(data)-1:
                        tableentry = (tableentry
                                      + r'\textit{%s}' % writeFEvals2(numpy.median(entry.maxevals), 2))
                        if isBold:
                            tableentry = r'\textbf{%s}' % tableentry
                        elif 11 < 3 and significance0vs1 < 0:
                            tableentry = r'\textit{%s}' % tableentry
                        tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                      % (alignment, tableentry))
                    elif tableentry.find('e') > -1 or (numpy.isinf(tmp) and i != len(data) - 1):
                        if isBold:
                            tableentry = r'\textbf{%s}' % tableentry
                        elif 11 < 3 and significance0vs1 < 0:
                            tableentry = r'\textit{%s}' % tableentry
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

                superscript = ''

                if nbstars > 0:
                    #tmp = '\hspace{-.5ex}'.join(nbstars * [r'\star'])
                    if z > 0:
                        superscript = r'\uparrow' #* nbstars
                    else:
                        superscript = r'\downarrow' #* nbstars
                        # print z, linebest[i], line1
                    if nbstars > 1:
                        superscript += str(int(nbstars))

                #if superscript or significance0vs1:
                    #s = ''
                    #if significance0vs1 > 0:
                       #s = '\star'
                    #if significance0vs1 > 1:
                       #s += str(significance0vs1)
                    #s = r'$^{' + s + superscript + r'}$'

                    #if tableentry.endswith('}'):
                        #tableentry = tableentry[:-1] + s + r'}'
                    #else:
                        #tableentry += s

                if dispersion[i] and not numpy.isinf(bestalgdata[i]):
                    tmp = writeFEvalsMaxPrec(dispersion[i]/bestalgdata[i], 2)
                    tableentry += (r'${\scriptscriptstyle(%s)}$' % tmp)

                if superscript:
                    s = r'$^{' + superscript + r'}$'

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
            extraeol.append(r'\hline')
        extraeol[-1] = ''

        outputfile = os.path.join(outputdir, 'pptable_%02dD%s.tex' % (d, info))
        spec = r'@{}c@{}|' + '*{%d}{@{}r@{}@{}l@{}}' % len(targetsOfInterest) + '|@{}r@{}@{}l@{}'
        #res = r'\providecommand{\algshort}{%s}' % alg1 + '\n'
        #res += tableLaTeXStar(table, width=r'0.45\textwidth', spec=spec,
                              #extraeol=extraeol)
        res = tableLaTeX(table, spec=spec, extraeol=extraeol)
        f = open(outputfile, 'w')
        f.write(res)
        f.close()
        if verbose:
            print "Table written in %s" % outputfile

