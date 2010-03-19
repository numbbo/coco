#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Rank-sum tests table on "Final Data Points".
that is, for example, using 1/#fevals(ftarget) if ftarget was reached and
-f_final otherwise as input for the rank-sum test, where obviously the larger
the better.
One table per function and dimension."""

from __future__ import absolute_import

import os
import numpy
import matplotlib.pyplot as plt
from bbob_pproc import bestalg
from bbob_pproc.pptex import tableLaTeX, tableLaTeXStar
#from bbob_pproc import ranksumtest
from bbob_pproc.bootstrap import ranksums
#from bbob_pproc.pplogloss import detERT
from bbob_pproc.pptex import writeFEvals2
#try:
    #supersede this module own ranksums method
    #from scipy.stats import ranksums as ranksums
#except ImportError:
    #from bbob_pproc.bootstrap import ranksums
    #pass

from pdb import set_trace

targetsOfInterest = (10., 1., 1e-1, 1e-3, 1e-5, 1e-7) # Needs to be sorted
targetf = 1e-8

#Get benchmark short infos: put this part in a function?
funInfos = {}
isBenchmarkinfosFound = False
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

def generateData(dsList0, dsList1):
    """Will create a numpy.array of the rank sum test values."""
    table = []
    it0 = iter(dsList0.evals)
    it1 = iter(dsList1.evals)

    def setNextLine(it, target):
        """Use the iterator of the evals array from DataSet to set nline to
           the current target function value."""

        try:
           nline = it.next()
           while target > nline[0]:
               nline = it.next()
        except StopIteration:
           #The algorithm did not reach the target function value.
           nline = numpy.array([-numpy.inf] + (len(nline) - 1) * [numpy.nan])

        return nline

    for t in targetsOfInterest:
        nline0 = (numpy.power(setNextLine(it0, t)[1:], -1)).copy()
        idxNan = numpy.isnan(nline0)
        nline0[idxNan] = -dsList0.finalfunvals[idxNan]
        nline1 = (numpy.power(setNextLine(it0, t)[1:], -1)).copy()
        idxNan = numpy.isnan(nline1)
        nline1[idxNan] = -dsList0.finalfunvals[idxNan]
        #table.append(numpy.array([t, ranksumtest.ranksums(nline0, nline1)[0]]))
        table.append(numpy.array([t, ranksums(nline0, nline1)[0]]))
        #set_trace()

    table = numpy.vstack(table)
    header = ['\Delta f', 'U']
    format = ('spec', '%3.2g' )
    return table, header, format

def formatData(table, header, format, fun):
    """Will try to format the data, if possible just from the table."""

    if isBenchmarkinfosFound:
        funname = ' %s' % funInfos[fun]
    else:
        funname = '%d' % fun

    header = ['\multicolumn{%d}{@{}c@{}}{%s}' % (max(len(l) for l in table),
                                          funname)]
    tableStrings = [header]
    for line in table:
         curline = []
         for i, elem in enumerate(line):
             if format[i] == 'spec':
                 if elem >= 1 and elem <= 100:
                     tmpstring = str(int(round(elem)))
                 else:
                     tmpstring = '%2.0e' % elem
                     tmpstring = tmpstring.split('e')
                     tmpstring = (tmpstring[0]
                                  + '\\!\\mathrm{\\hspace{0.10em}e}'
                                  + '%d' % int(tmpstring[1]))
                 curline.append(tmpstring)
             else:
                 curline.append(format[i] % elem)
         tableStrings.append(curline)

    return tableStrings

def main2(dsList0, dsList1, dimsOfInterest, outputdir, info='', verbose=True):
    """Generate comparison tables.
    One table per dimension...
    """

    dictDim0 = dsList0.dictByDim()
    dictDim1 = dsList1.dictByDim()

    alg0 = set(i[0] for i in dsList0.dictByAlg().keys()).pop()[0:3]
    alg1 = set(i[0] for i in dsList1.dictByAlg().keys()).pop()[0:3]

    if info:
        info = '_' + info

    dims = set.intersection(set(dictDim0.keys()), set(dictDim1.keys()))
    if not bestalg.bestalgentries:
        bestalg.loadBBOB2009()

    header = [r'$\Delta f$']
    for i in targetsOfInterest:
        #header.append(r'\multicolumn{2}{@{}c@{}}{$10^{%d}$}' % (int(numpy.log10(i))))
        header.append(r'\multicolumn{2}{@{}c@{}}{1e%+d}' % (int(numpy.log10(i))))
    header.append(r'\multicolumn{2}{|@{}r@{}}{\#succ}')

    for d in dimsOfInterest: # TODO set as input arguments
        table = [header]
        extraeol = [r'\hline']
        dictFunc0 = dictDim0[d].dictByFunc()
        dictFunc1 = dictDim1[d].dictByFunc()
        funcs = set.union(set(dictFunc0.keys()), set(dictFunc1.keys()))

        nbtests = len(funcs) * 2. #len(dimsOfInterest)

        for f in sorted(funcs):
            bestalgentry = bestalg.bestalgentries[(d, f)]
            curline = [r'${\bf f_{%d}}$' % f]
            bestalgdata = bestalgentry.detERT(targetsOfInterest)
            bestalgevals, bestalgalgs = bestalgentry.detEvals(targetsOfInterest)

            for i in bestalgdata[:-1]:
                curline.append(r'\multicolumn{2}{@{}c@{}}{%s}' % writeFEvals2(i, 2))
            curline.append(r'\multicolumn{2}{@{}c@{}|}{%s}' % writeFEvals2(bestalgdata[-1], 2))
            rankdatabest = []
            for i, j in enumerate(bestalgevals):
                if bestalgalgs[i] is None:
                    tmp = -bestalgentry.finalfunvals[bestalgentry.algs[-1]]
                else:
                    tmp = numpy.power(j, -1.)
                    tmp[numpy.isnan(tmp)] = -bestalgentry.finalfunvals[bestalgalgs[i]][numpy.isnan(tmp)]
                rankdatabest.append(tmp)

            tmp = bestalgentry.detEvals([targetf])[0][0]
            tmp2 = numpy.sum(numpy.isnan(tmp) == False)
            curline.append('%d' % (tmp2))
            if tmp2 > 0:
                curline.append('/%d' % len(tmp))

            table.append(curline[:])
            extraeol.append('')

            rankdata0 = []

            # generate all data for ranksum test
            rankdata = {}  # rankdata[nb +/- 1,i] is of interest later
            ertdata = {}
            for nb, entries in enumerate((dictFunc0, dictFunc1)):  # copy paste of loop below
                try:
                    entry = entries[f][0] # take the first element
                except KeyError:
                    continue
                ertdata[nb] = entry.detERT(targetsOfInterest)
                evals = entry.detEvals(targetsOfInterest)
                for i, tmp in enumerate(ertdata[nb]): 
                    # print i
                    rankdata[nb,i] = numpy.power(evals[i], -1.)
                    rankdata[nb,i][numpy.isnan(rankdata[nb,i])] = -entry.finalfunvals[numpy.isnan(rankdata[nb,i])]

            for nb, entries in enumerate((dictFunc0, dictFunc1)):
                try:
                    entry = entries[f][0] # take the first element
                except KeyError:
                    continue
                if nb == 0:
                    curline = [r'0:\:\algzeroshort']#\hspace*{\fill}']
                else:
                    curline = [r'1:\:\algoneshort']#\hspace*{\fill}']

                #curline = [r'\alg%sshort' % tmp]
                #curline = [r'Alg%d' % nb]
                #curline = [r'%.3s%d' % (entry.algId, nb)]

                data = entry.detERT(targetsOfInterest)
                evals = entry.detEvals(targetsOfInterest)
                if nb == 0:
                    assert not isinstance(data, numpy.ndarray)
                    data0 = data[:] # check if it is not an array
                 
                for i, j in enumerate(data):  # is j an appropriate identifier here?
                    #if numpy.isnan(float(j)/bestalgdata[i]):
                    #    set_trace()

                    # assign significance flag
                    significance0vs1 = 0
                    if nb == 0:
                          istat0 = 0
                          istat1 = 1 
                    else: 
                          istat0 = 1
                          istat1 = 0 
                    z, p = ranksums(rankdata[istat0,i], rankdata[istat1,i])
                    if (nbtests * p < 0.05
                            and z > 0 and not numpy.isinf(ertdata[istat0][i]) and 
                            z * (ertdata[istat1][i] - ertdata[istat0][i]) > 0):  # z-value and ERT-ratio must agree
                          significance0vs1 = -int(numpy.ceil(numpy.log10(nbtests * p)))
                    if nbtests * p < 0.05 and z < 0 and z * (ertdata[istat1][i] - ertdata[istat0][i]) > 0:
                          significance0vs1 = int(numpy.ceil(numpy.log10(nbtests * p)))

                    alignment = 'c'
                    if i == len(data) - 1: # last element
                        alignment = 'c|'
                    if numpy.isinf(bestalgdata[i]):
                        tableentry = (r'\multicolumn{2}{@{}%s@{}}{\textbf{\textit{%s}}}'
                                      % (alignment, writeFEvals2(float(j), 2)))
                        # TODO: is this the correct behaviour?
                    else:
                        # Formatting
                        assert not numpy.isnan(float(j)/bestalgdata[i])
                        tmp = float(j)/bestalgdata[i]
                        tableentry = writeFEvals2(tmp, 2)
                        isBold = False
                        if 11 < 3 and tmp <= 3:
                            isBold = True

                        if significance0vs1 > 0: 
                           isBold = True

                        if tableentry.find('e') > -1:
                            if isBold:
                                tableentry = r'\textbf{%s}' % tableentry
                            elif significance0vs1 < 0:
                                tableentry = r'\textit{%s}' % tableentry
                            tableentry = (r'\multicolumn{2}{@{}%s@{}}{%s}'
                                          % (alignment, tableentry))
                        else:
                            tmp = tableentry.split('.', 1)
                            if isBold:
                                tmp = list(r'\textbf{%s}' % i for i in tmp)
                            elif significance0vs1 < 0:
                                tmp = list(r'\textit{%s}' % i for i in tmp)
                            tableentry = ' & .'.join(tmp)
                            if len(tmp) == 1:
                                tableentry += '&'

                    currankdata = numpy.power(evals[i], -1.)
                    currankdata[numpy.isnan(currankdata)] = -entry.finalfunvals[numpy.isnan(currankdata)]

                    if 11 < 3 and nb == 0:
                        rankdata0.append(currankdata.copy())
                        addition = ''
                    elif 11 < 3: #nb==1
                        z, p = ranksums(rankdata0[i], currankdata)
                        tmp = '.'
                        #addition = '' # Uncomment to gain space. Meanwhile it displays the worst case scenario
                        if ((nbtests * p) < 0.05
                            and ((numpy.isinf(data0[i]) and numpy.isinf(j))
                                 or z * (j - data0[i]) > 0)):  # z-value and ERT-ratio must agree
                            nbstars = -numpy.ceil(numpy.log10(nbtests * p))
                            if z > 0:
                                tmp = r'\uparrow' #* nbstars
                            else:
                                tmp = r'\downarrow' #* nbstars
                                # print z, linebest[i], line1
                            if nbstars > 1:
                                tmp += str(int(nbstars))
                            #addition = tmp # uncomment to gain space.
                        addition = r'/' + tmp # Comment and uncomment line above to gain space
                        #addition = tmp # Comment and uncomment line above to gain space

                    superscript = '\uparrow'
                    superscript = ''
                    #if addition:
                        #superscript = '.'

                    z, p = ranksums(rankdatabest[i], currankdata)
                    if ((nbtests * p) < 0.05
                        and ((numpy.isinf(bestalgdata[i]) and numpy.isinf(j))
                             or z * (j - bestalgdata[i]) > 0)):  # z-value and ERT-ratio must agree
                        nbstars = -numpy.ceil(numpy.log10(nbtests * p))
                        #tmp = '\hspace{-.5ex}'.join(nbstars * [r'\star'])
                        #set_trace()
                        if z > 0:
                            superscript = r'\uparrow' #* nbstars
                        else:
                            superscript = r'\downarrow' #* nbstars
                            # print z, linebest[i], line1
                        if nbstars > 1:
                            superscript += str(int(nbstars))

                    addition = '' 
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

                    if 11 < 3 and (superscript or addition):
                        isClosingBrace = False
                        if tableentry.endswith('}'):
                            isClosingBrace = True
                            tableentry = tableentry[:-1]
                        #tableentry += r'$^{' + superscript + '}_{' + addition + '}$'
                        tableentry += r'$^{' + superscript + addition + '}$'
                        if isClosingBrace:
                            tableentry += '}'

                    curline.append(tableentry)

                # Two cases: both tabular give an overfull hbox
                # AND generate a LaTeX Warning: Float too large for page by 16.9236pt on input line 421. (noisy)
                # OR  generate a LaTeX Warning: Float too large for page by 33.57658pt on input line 421. (noisy)

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

        outputfile = os.path.join(outputdir, 'cmptable_%02dD%s.tex' % (d, info))
        spec = r'@{}c@{}|' + '*{%d}{@{}r@{}@{}l@{}}' % len(targetsOfInterest) + '|@{}r@{}@{}l@{}'
        res = r'\providecommand{\algzeroshort}{%s}' % alg0 + '\n'
        res += r'\providecommand{\algoneshort}{%s}' % alg1 + '\n'
        res += tableLaTeXStar(table, width=r'0.45\textwidth', spec=spec,
                              extraeol=extraeol)
        #res += tableLaTeX(table, spec=spec, extraeol=extraeol)
        f = open(outputfile, 'w')
        f.write(res)
        f.close()
        if verbose:
            print "Table written in %s" % outputfile

def main(dsList0, dsList1, outputdir, info='', verbose=True):

    """Will loop over the functions, dimension and so on."""

    dictFunc0 = dsList0.dictByFunc()
    dictFunc1 = dsList1.dictByFunc()
    funcs = set.union(set(dictFunc0), set(dictFunc1))

    if info:
        info = '_' + info

    for f in funcs:
        #replace dictFunc0[func] (a DataSetList) with a dictionary of DataSetList 
        dictFunc0[f] = dictFunc0[f].dictByDim()
        dictFunc1[f] = dictFunc1[f].dictByDim()
        #TODO: what if all functions were not tested for alg0 et alg1?
        #TODO: what if all dimensions were not tested for alg0 et alg1?
        dims = set.union(set(dictFunc0[f]), set(dictFunc1[f]))
        for d in dims:
            outputfile = os.path.join(outputdir, 'cmptable_f%02d_%02dD%s'
                                                 % (f, d, info))
            table, header, format = generateData(dictFunc0[f][d][0],
                                                 dictFunc1[f][d][0])
            # Both dictFun[f][d] should be of length 1.
            tableofstrings = formatData(table, header, format, f)
            set_trace()
            res = tableLaTeX(tableofstrings)
            f = open(outputfile, 'w')
            f.write(res)
            f.close()
