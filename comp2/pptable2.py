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
from bbob_pproc.pptex import tableLaTeX
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

    header = ['\multicolumn{%d}{c}{%s}' % (max(len(l) for l in table),
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

    if info:
        info = '_' + info

    dims = set.intersection(set(dictDim0.keys()), set(dictDim1.keys()))
    if not bestalg.bestalgentries:
        bestalg.loadBBOB2009()

    header = [r'$\Delta f$']
    for i in targetsOfInterest:
        header.append(r'\multicolumn{2}{c}{$10^{%d}$}' % (int(numpy.log10(i))))
    header.append(r'\multicolumn{2}{|@{}l@{}}{\#succ}')

    for d in dimsOfInterest: # TODO set as input arguments
        table = [header]
        extraeol = [r'\hline']
        dictFunc0 = dictDim0[d].dictByFunc()
        dictFunc1 = dictDim1[d].dictByFunc()
        funcs = set.union(set(dictFunc0.keys()), set(dictFunc1.keys()))

        for f in sorted(funcs):
            bestalgentry = bestalg.bestalgentries[(d, f)]
            curline = [r'${\bf f_{%d}}$' % f]
            bestalgdata = bestalgentry.detERT(targetsOfInterest)
            bestalgevals, bestalgalgs = bestalgentry.detEvals(targetsOfInterest)

            for i in bestalgdata[:-1]:
                curline.append(r'\multicolumn{2}{c}{%s}' % writeFEvals2(i, 2))
            curline.append(r'\multicolumn{2}{c|}{%s}' % writeFEvals2(bestalgdata[-1], 2))
            line0 = []
            for i, j in enumerate(bestalgevals):
                if bestalgalgs[i] is None:
                    tmp = -bestalgentry.finalfunvals[bestalgentry.algs[-1]]
                else:
                    tmp = numpy.power(j, -1.)
                    tmp[numpy.isnan(tmp)] = -bestalgentry.finalfunvals[bestalgalgs[i]][numpy.isnan(tmp)]
                line0.append(tmp)

            tmp = bestalgentry.detEvals([targetf])[0][0]
            if not tmp is numpy.array([numpy.nan]):
                #set_trace()
                curline.append('%d' % (numpy.sum(numpy.isnan(tmp) == False)))
                curline.append('/%d' % len(tmp))
            else:
                curline.append('%d' % 0)
                curline.append('/%d' % 0)

            table.append(curline[:])
            extraeol.append('')

            for nb, entries in enumerate((dictFunc0, dictFunc1)):
                try:
                    entry = entries[f][0] # take the first element
                except KeyError:
                    continue
                #if nb == 0:
                    #tmp = 'zero'
                #else:
                    #tmp = 'one'
                #curline = [r'\alg%s' % tmp]
                #curline = [r'Alg%d' % nb]
                curline = [r'%.3s%d' % (entry.algId, nb)]


                data = entry.detERT(targetsOfInterest)
                evals = entry.detEvals(targetsOfInterest)
                for i, j in enumerate(data):
                    #if numpy.isnan(float(j)/bestalgdata[i]):
                    #    set_trace()
                    if numpy.isinf(bestalgdata[i]):
                        tableentry = r'\multicolumn{2}{c}{\textit{%s}}' % writeFEvals2(float(j), 2)
                    else:
                        # Formatting
                        tableentry = writeFEvals2(float(j)/bestalgdata[i], 2)
    
                        if tableentry.find('e') > -1:
                            tableentry = r'\multicolumn{2}{c}{%s}' % tableentry
                        else:
                            if tableentry.find('.') > -1:
                                tableentry = ' & .'.join(tableentry.split('.'))
                            else:
                                tableentry += '&'

                    line1 = numpy.power(evals[i], -1.)
                    line1[numpy.isnan(line1)] = -entry.finalfunvals[numpy.isnan(line1)]

                    z, p = ranksums(line0[i], line1)
                    nbtests = 1 # TODO?
                    if (nbtests * p) < 0.05:
                        nbstars = -numpy.ceil(numpy.log10(nbtests * p))
                        #tmp = '\hspace{-.5ex}'.join(nbstars * [r'\star'])
                        if nbstars > 0:
                            tmp = r'\star'
                            if nbstars > 1:
                                tmp += str(int(nbstars))
                            if tableentry.endswith('}'):
                                tableentry = tableentry[:-1]
                                tableentry += '$^{' + tmp + '}$}'
                            else:
                                tableentry += '$^{' + tmp + '}$'

                    curline.append(tableentry)

                tmp = entry.evals[entry.evals[:, 0] <= targetf, 1:] # set as global variable?
                try:
                    tmp = tmp[0]
                    curline.append('%d' % numpy.sum(numpy.isnan(tmp) == False))
                except IndexError:
                    curline.append('%d' % 0)
                curline.append('/%d' % entry.nbRuns())
                #if any(numpy.isinf(data)) and numpy.sum(numpy.isnan(tmp) == False) == 15:
                    #set_trace()

                table.append(curline[:])
                extraeol.append('')

            extraeol[-1] = r'\hline'
        extraeol[-1] = ''

        outputfile = os.path.join(outputdir, 'cmptable_%02dD%s.tex' % (d, info))
        spec = '@{}c@{}|' + '*{%d}{@{}r@{}@{}l@{}}' % len(targetsOfInterest) + '|@{}r@{}@{}l@{}'
        res = tableLaTeX(table, spec=spec, extraeol=extraeol)
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
