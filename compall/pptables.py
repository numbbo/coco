#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for the generation of TeX tables.
See Section Comparison Tables in
http://tao.lri.fr/tiki-index.php?page=BBOC+Data+presentation
"""

from __future__ import absolute_import

import os
from pdb import set_trace
import numpy
from bbob_pproc import bestalg, bootstrap
from bbob_pproc.pptex import writeFEvals, writeFEvals2, writeFEvalsMaxPrec, writeLabels, tableLaTeX
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

maxfloatrepr = 100000.
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

def writeFunVal(funval):
    """Returns string representation of a function value to use in a table."""
    res = ('%.1e' % funval).split('e')
    res[0] = res[0].replace('.', '')
    res[1] = '%+d' % (int(res[1]) - 1)
    return r'\textit{%s}' % 'e'.join(res)

def onealg(dsList, allmintarget, allertbest):
    """Helper routine for the generation of a table for one algorithm."""

    table = []
    unsolved = {}

    for t in sorted(allmintarget.keys()):
        erts = []
        soltrials = 0
        nbtrials = 0
        solinstances = 0
        nbinstances = 0
        solfcts = 0
        nbfcts = 0
        for i, entry in enumerate(dsList):
            try:
                if numpy.isnan(allmintarget[t][(entry.funcId, entry.dim)]):
                    continue
            except KeyError:
                continue
            nbtrials += numpy.shape(entry.evals)[1] - 1
            dictinstance = entry.createDictInstance()
            nbinstances += len(dictinstance)
            nbfcts += 1
            tmp = unsolved.setdefault(i, {})
            tmp['unsoltrials'] = numpy.shape(entry.evals)[1] - 1 # may be modified a posteriori
            tmp['nbtrials'] = numpy.shape(entry.evals)[1] - 1
            tmp['unsolinstances'] = len(dictinstance) # may be modified a posteriori
            tmp['nbinstances'] = len(dictinstance)
            tmp['unsolved'] = True
            tmp['runlengths'] = entry.maxevals

            for l in range(len(entry.evals)):
                tmpline = entry.evals[l]
                if tmpline[0] < allmintarget[t][(entry.funcId, entry.dim)]:
                    solfcts += 1
                    tmp['unsolved'] = False
                    soltrials += numpy.sum(numpy.isfinite(tmpline[1:]))
                    tmp['runlengths'] = entry.maxevals[numpy.isnan(tmpline[1:])]
                    tmp['unsoltrials'] = len(tmp['runlengths'])
                    #TODO: hard to read
                    tmpsolinstances = 0
                    for idx in dictinstance.values():
                        try:
                            if numpy.isfinite(list(tmpline[j+1] for j in idx)).any():
                                tmpsolinstances += 1
                        except IndexError:
                            pass
                            #set_trace() # TODO: problem with the instances... MCS!
                    solinstances += tmpsolinstances
                    tmp['unsolinstances'] = len(dictinstance) - tmpsolinstances
                    erts.append(float(entry.ert[l]) / allertbest[t][(entry.funcId, entry.dim)])
                    break

        if len(erts) > 0:
            erts.sort()
            line = [t]
            line.extend((float(soltrials)/nbtrials*100., float(solinstances)/nbinstances*100.,
                         solfcts, nbfcts))
            line.append(erts[0])
            line.extend(prctile(erts, [10, 25, 50, 75, 90]))
            table.append(line)

    unsolved = unsolved.values()
    unsolvedrl = []
    for i in unsolved:
        unsolvedrl.extend(i['runlengths'])

    if unsolvedrl:
        unsolvedrl.sort()
        if float(sum(i['unsolinstances'] for i in unsolved))/sum(i['nbinstances'] for i in unsolved) > 1:
            #set_trace() # TODO: problem
            pass
        line = [numpy.inf,
                float(sum(i['unsoltrials'] for i in unsolved))/sum(i['nbtrials'] for i in unsolved) * 100,
                float(sum(i['unsolinstances'] for i in unsolved))/sum(i['nbinstances'] for i in unsolved) * 100,
                sum(list(i['unsolved'] for i in unsolved)), len(dsList), unsolvedrl[0]]
        line.extend(prctile(unsolvedrl, [10, 25, 50, 75, 90]))
        table.append(line)

    return table

def tableonealg(dsList, allmintarget, allertbest, sortedAlgs=None,
                outputdir='.'):
    """Routine for the generation of a table for an algorithm."""

    header2 = ('evals/D', '\%trials', '\%inst', '\multicolumn{2}{c|}{fcts}', 'best', '10', '25', 'med', '75', '90')
    format2 = ('%.3g', '%d', '%d', '%d', '%d', '%1.1e', '%1.1e', '%1.1e', '%1.1e', '%1.1e', '%1.1e')
    ilines = [r'\begin{tabular}{cccc@{/}c|cccccc}',
              r'\multicolumn{5}{c|}{Solved} & \multicolumn{6}{|c}{ERT/ERT$_{\textrm{best}}$} \\',
              ' & '.join(header2)]

    dictDim = dsList.dictByDim()
    for d, dentries in dictDim.iteritems():
        dictAlg = dentries.dictByAlg()

        # one-alg table
        for alg in sortedAlgs:
            # Regroup entries by algorithm
            lines = ilines[:]
            algentries = DataSetList()
            for i in alg:
                if dictAlg.has_key(i):
                    algentries.extend(dictAlg[i])
            table = onealg(algentries, allmintarget, allertbest)
            for i in table:
                lines[-1] += r'\\'

                if numpy.isinf(i[0]):
                    tmpstr = r'$\infty$'
                else:
                    tmpstr = format2[0] % i[0]
                for j in range(1, 5):
                    tmpstr += (' & %s' % format2[j]) % i[j]
                for j in range(5, len(i)):
                    tmpstr += ' & %s' % writeFEvals(i[j])
                lines.append(tmpstr)

            lines.append(r'\end{tabular}')
            f = open(os.path.join(outputdir, 'pptable_%s_%02dD.tex' % (algShortInfos[alg[0]], d)), 'w')
            # any element of alg would convene.
            f.write('\n'.join(lines) + '\n')
            f.close()

def tablemanyalg(dsList, allmintarget, allertbest, sortedAlgs=None,
                 outputdir='.'):
    """Generate a table with the figures of multiple algorithms."""

    stargets = sorted(allmintarget.keys())
    dictDim = dsList.dictByDim()
    maxRank = 3

    for d, dentries in dictDim.iteritems():
        dictAlg = dentries.dictByAlg()
        # Multiple algorithms table.
        # Generate data
        table = []
        algnames = []

        for alg in sortedAlgs:
            # Regroup entries by algorithm
            algentries = DataSetList()
            for i in alg:
                if dictAlg.has_key(i):
                    algentries.extend(dictAlg[i])
            if not algentries:
                continue
            algnames.append(writeLabels(algPlotInfos[alg[0]]['label']))
            tmp = []
            for t in stargets:
                dictFunc = algentries.dictByFunc()
                erts = []
                for func, entry in dictFunc.iteritems():
                    try:
                        entry = entry[0]
                    except:
                        raise Usage('oops too many entries')

                    try:
                        if numpy.isnan(allmintarget[t][(func, d)]):
                            continue
                    except LookupError:
                        continue
                    # At this point the target exists.
                    try:
                        erts.append(entry.ert[entry.target<=allmintarget[t][(func, d)]][0]/allertbest[t][(func, d)])
                    except LookupError:
                        erts.append(numpy.inf)

                if numpy.isfinite(erts).any():
                    tmp += [numpy.median(erts), numpy.min(erts), numpy.sum(numpy.isfinite(erts))]
                else:
                    tmp += [numpy.inf, numpy.inf, 0]
            table.append(tmp)

        # Process over all data
        table = numpy.array(table)
        kept = [] # range(numpy.shape(table)[1])
        targetkept = []
        for i in range(1, (numpy.shape(table)[1])/3 + 1):
            if (table[:, 3*i - 1] != 0).any():
                kept.extend([3*i - 3, 3*i - 2 , 3*i - 1])
                targetkept.append(i-1)
        table = table[:, kept]
        #set_trace()
        dtype = []
        for i, t in enumerate(stargets):
            dtype.extend([('med%d' % i, 'f4'), ('min%d' % i, 'f4'),
                          ('nbsolved%d' % i, 'i1')])
        dtype = list(dtype[i] for i in kept)
        boldface = sortColumns(table, maxRank)

        idxsort = numpy.argsort(numpy.array(list(tuple(i) for i in table),
                                            dtype=dtype),
                                order=('med4', 'med2', 'med0', 'min0'))
        # Sorted successively by med(ERT) / ERTbest for fevals/D = 100, 10, 1
        # and then min(ERT) / ERTbest for fevals/D = 1

        # format the data
        lines = [r'\begin{tabular}{c' + 'c@{/}c@{(}c@{) }'*len(targetkept) + '}']
        tmpstr = 'evals/D'
        for t in list(stargets[i] for i in targetkept):
            nbsolved = sum(numpy.isfinite(list(allmintarget[t][i] for i in allmintarget[t] if i[1] == d)))
            #set_trace()
            tmpstr += (r' & \multicolumn{2}{c@{(}}{%s} & %d' % (writeFEvals(t), nbsolved))
        lines.append(tmpstr)

        for i in idxsort:
            line = table[i]

            lines[-1] += r'\\'
            curline = algnames[i]
            for j in range(len(table[i])):
                curline += ' & '
                if (j + 1) % 3 > 0: # the test may not be necessary
                    if numpy.isinf(line[j]):
                        tmpstr = '.'
                    else:
                        tmpstr = '%s' % (writeFEvals(line[j]))

                    if i in boldface[j]:
                        tmpstr = r'\textbf{' + tmpstr + '}'

                    curline += tmpstr
                else:
                    curline += '%d' % line[j] # nb solved.

            lines.append(curline)

        lines.append(r'\end{tabular}')

        f = open(os.path.join(outputdir, 'pptableall_%02dD.tex' % (d)), 'w')
        f.write('\n'.join(lines) + '\n')
        f.close()

def tablemanyalg2(dsListperAlg, dimsOfInterest=None, outputdir='.', info=''):
    """Generate tables with the results of many algorithms over many functions.

    One table per dimension.

    """

    # TODO: how for the significance test?
    # When aggregating over functions, choose to display data with a fixed budget

    maxRank = 3

    #dimensions = set.union(set(i.dim for i in dsList for dsList in dsListperAlg))
    dictDim = {}
    for dsList in dsListperAlg:
        tmp = dsList.dictByDim()
        for dim, curdsList in tmp.iteritems():
            dictDim.setdefault(dim, []).append(curdsList)

    if not dimsOfInterest:
        dimsOfInterest = dictDim.keys()

    for dim in dimsOfInterest:
        dentries = dictDim.get(dim, False)
        if not dentries:
            continue

        # dentries should be a list of DataSetList, each corresponding to one
        # algorithm in one dimension

        # Generate data
        table = []
        algnames = []

        funcs = set(i.funcId for i in dsList for dsList in dentries)
        maxevals = max(i.mMaxEvals() for i in dsList for dsList in dentries)
        budgets = [2.*dim]
        budgets.extend(numpy.power(10., numpy.arange(1, numpy.floor(numpy.log10(maxevals))))*dim)

        # determine targets and best algorithms, one per budget, function
        targets = {}
        bestalg = {}
        for b in budgets:
            for f in funcs:
                for dsList in dentries: # loop over algorithms
                    entry = dsList.dictByFunc()[f][0]
                    curtarget = detf(entry, [b])[0] # does this output a distribution?
                    set_trace()
                    if curtarget < targets.get((b, f), numpy.inf): # what in case of ties?
                        targets[(b, f)] = curtarget
                        bestalg[(b, f)] = (entry.funcId, entry.comment)
                        bestert[(b, f)] = entry.detERT([curtarget])[0]

        #algnames.append(writeLabels(algPlotInfos[tmp]['label']))

        # compute the ERT for all algorithms adn divide by ERT_best
        for dsList in dentries:
            dictFunc = dsList.dictByFunc()
            for b in budgets:
                erts = []
                ertratios = []
                for f in dictFunc:
                    entry = dictFunc[f]
                    tmp = entry.detERT([targets[(b, f)]])[0]
                    erts.append(tmp)
                    ertratios = tmp/bestert[(b, f)]
                    #TODO: statistics on ertratios...
                    set_trace()


        # TODO: write table

def tablemanyalgonefunc(dictAlg, allmintarget, allertbest, sortedAlgs=None,
                        outputdir='.', verbose=True):
    """Generate one table per function showing results of multiple algorithms.
    """

    dictDim = {}
    for alg, tmpdsList in dictAlg.iteritems():
        tmpdictDim = tmpdsList.dictByDim()
        for d, entries in tmpdictDim.iteritems():
            dictDim.setdefault(d, {})[alg] = entries

    #widthtable = 3 # Put in as global? 3 functions wide
    #TODO: what about the keys of allmintarget and allertbest: why are they negative?
    # TODO: split the generation of the tables from their formatting/output
    # ... if its possible
    for d, dictAlgbyDim in dictDim.iteritems():
        # Summary table: multiple algorithm for each function
        nbtarget = len(allmintarget)
        stargets = sorted(allmintarget.keys())
        #groups = [[1, 2], [3], [4], [5, 6], [7], [8, 9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21, 22], [23], [24]]
        #funcs = sorted(dictAlgbyDim.dictByFunc().keys())
        #groups = list(funcs[i*widthtable:(i+1)*widthtable] for i in range(len(funcs)/widthtable + 1))
        funcs = set()
        for i in dictAlgbyDim.values():
            funcs |= set(j.funcId for j in i)

        groups = list([i] for i in funcs)

        for numgroup, g in enumerate(groups):
            isFunNoisefree = False
            isFunNoisy = False
            if not g:
                continue

            algnames = []
            table = []
            replacement = []
            if sortedAlgs is None:
                sortedAlgs = dictAlgbyDim.keys()

            #if g == [4] and d == 40:
                #set_trace()

            for alg in sortedAlgs:
                curline = []
                replacementLine = []
                # Regroup entries by algorithm
                try:
                    algentries = dictAlgbyDim[alg]
                except KeyError:
                    continue

                if not algentries:
                    continue
                dictF = algentries.dictByFunc()
                for func in g:
                    isLastInfoWritten = False
                    try:
                        entries = dictF[func]
                        try:
                            entry = entries[0]
                        except:
                            raise Usage('Problem with the entries')

                    except KeyError:
                        if curline or replacementLine:
                            curline.extend([numpy.inf]*len(stargets))
                            replacementLine.extend(['.']*len(stargets))

                        continue # empty data

                    for t in stargets:
                        try:
                            if numpy.isnan(allmintarget[t][(func, d)]):
                                continue
                        except KeyError:
                            continue
                        try:
                            if not numpy.isfinite(allertbest[t][(func, d)]):
                                curline.append(entry.ert[entry.target<=allmintarget[t][(func, d)]][0])
                            else:
                                curline.append(entry.ert[entry.target<=allmintarget[t][(func, d)]][0]/allertbest[t][(func, d)])
                            replacementLine.append('')
                        except LookupError: #IndexError, KeyError:
                            curline.append(numpy.inf)
                            if not isLastInfoWritten:
                                replacementLine.append(r'%s\textit{/%s}' % (writeFunVal(numpy.median(entry.finalfunvals)), writeFEvals(numpy.median(entry.maxevals)/entry.dim, precision='.1')))
                                isLastInfoWritten = True
                            else:
                                replacementLine.append('.')

                tmp = set((i.algId, i.comment) for i in algentries)
                if replacementLine and curline:
                    try:
                        tmp = algPlotInfos[tmp.pop()]['label']
                    except KeyError:
                        tmp = algentries[0].algId # Take the first reasonable one.
                    algnames.append(writeLabels(tmp))
                    replacement.append(replacementLine)
                    table.append(curline)

            try:
                table = numpy.array(table)
            except ValueError:
                pass

            # Process data
            boldface = sortColumns(table, maxRank=3)
            # do not use bold face for infinite ert entries
            for i, line in enumerate(table):
                for j, entry in enumerate(line):
                    if numpy.isinf(entry) and i in boldface[j]:
                        boldface[j].remove(i)

            # Format data
            lines = [r'\begin{tabular}{c', '', r'$\Delta$ftarget', r'ERT$_{\textrm{best}}$/D']
            for func in g:
                if func in range(1, 25):
                    isFunNoisefree = True
                elif func in range(101, 131):
                    isFunNoisy = True
                curtargets = []
                for t in stargets:
                    try:
                        if numpy.isfinite(allmintarget[t][(func, d)]):
                            curtargets.append(t)
                    except KeyError:
                        continue
                lines[0] += len(curtargets) * 'c'
                lines[0] += 'c' # algname in the end.
                if isBenchmarkinfosFound:
                    lines[1] += (r' & \multicolumn{%d}{c}{{\normalsize \textbf{%s}}}' % (len(curtargets), funInfos[func]))
                else:
                    lines[1] += (r' & \multicolumn{%d}{c}{{\normalsize \textbf{f%d}}}' % (len(curtargets), func))

                for t in curtargets:
                    try:
                        lines[2] += (r'& %1.0e' % allmintarget[t][(func, d)])
                    except KeyError:
                        lines[2] += (r'& .')
                    try:
                        if numpy.isnan(allertbest[t][(func, d)]):
                            tmp = '$\infty$'
                        else:
                            tmp = (writeFEvalsMaxPrec(float(allertbest[t][(func, d)])/d, 2))
                        lines[3] += (r'& %s' % tmp)
                        #lines[3] += (r'& %s' % (writeFEvals(float(allertbest[t][(func, d)])/d, '.3')))
                    except KeyError:
                        lines[3] += (r'& .')

            lines[0] += '}'
            lines[1] += r'\\'
            lines[2] += r' & $\Delta$ftarget \\'
            lines[3] += r' & ERT$_{\textrm{best}}$/D \\'
            lines.append(r'\hline')

            for i, line in enumerate(table):
                if lines[-1] != r'\hline':
                    lines[-1] += r'\\'
                tmpstr = '%s' % algnames[i]
                # Regroup entries by algorithm
                #dictF = algentries.dictByFunc()
                for j in range(len(line)):
                    if replacement[i][j]:
                        tmp = '%s' % replacement[i][j]
                    else:
                        try:
                            tmp = '%s' % writeFEvalsMaxPrec(line[j], 2)
                        except AssertionError:
                            tmp = '.'
                            #set_trace()
                        #tmp = '%s' % writeFEvals(line[j])

                    if i in boldface[j] or line[j] < 3:
                        tmp = r'\textbf{' + tmp + '}'
                    tmpstr += r' & ' + tmp
                # Repeated algorithm name.
                tmpstr += r' & %s %s' % (algnames[i], cite(algnames[i],
                                                   isFunNoisefree, isFunNoisy))

                lines.append(tmpstr)

            lines.append(r'\end{tabular}')
            #f = open(os.path.join(outputdir, 'pptablef%d_%02dD.tex' % (numgroup + 1, d)), 'w')
            #Line below preferred because the numgroup corresponds to the
            #function number which is the case as long as each group has only
            #one function
            f = open(os.path.join(outputdir, 'pptablef%d_%02dD.tex' % (g[0], d)), 'w')
            f.write('\n'.join(lines) + '\n')
            f.close()

def tablemanyalgonefunc2(dictAlg, sortedAlgs, targets, outputdir='.',
                         verbose=True):
    """Generate one table per function showing results of multiple algorithms.

    Difference with the first version:
    * numbers aligned using the decimal separator
    * premices for dispersion measure
    * significance test against best algorithm
    * table width...
    """

    if not bestalg.bestalgentriesever:
        bestalg.loadBBOBever()

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

        # best ever
        refalgentry = bestalg.bestalgentriesever[df]
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
            assert len(entries) == 1 # TODO: could be 0?
            entry = entries[0]
            algentry.append(entry)

            algnames.append(sortedAlgs[n][0:5]) # TODO: check

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

        for i, alg in enumerate(algnames):
        #algname, entries, irs, line, line2, succ, runs, testres1alg in zip(algnames,
                   #data, dispersion, isBoldArray, isItalArray, nbsucc, nbruns, testres):
            curline = [writeLabels(alg) + r'\hspace*{\fill}']

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
                        curline.append(r'\multicolumn{2}{%s}{\textbf{%s}\,(%s)}'
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
                            tmp += '\,(%s)' % writeFEvalsMaxPrec(dispersion/refalgert[j], 2)
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
                            tmp2[-1] += ('\,(%s)' % writeFEvalsMaxPrec(dispersion/refalgert[j], 2))

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
        f = open(os.path.join(outputdir, 'pptables_f%03d_%02dD.tex' % (df[1], df[0])), 'w')
        f.write(res)
        f.close()
