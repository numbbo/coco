#!/usr/bin/env python

from __future__ import absolute_import

import os
from pdb import set_trace
import pickle
import numpy
from bbob_pproc import pptex
from bbob_pproc.bootstrap import prctile
from bbob_pproc.dataoutput import algShortInfos, algPlotInfos
from bbob_pproc.pproc import DataSetList

allmintarget = {}
allmedtarget = {}

"""
"""

#TODO: use structured arrays!

def sortColumns(table, maxRank=None):
    """For each column in table, returns a list of the maxRank-ranked
    elements. This list may have a length larger than maxRank in the case of
    ties.
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

def writeLabels(label):
    """Format text to be output by LaTeX."""
    return label.replace('_', r'\_')

def writeFunVal(funval):
    """Returns string representation of a function value to use in a table."""
    res = ('%.1e' % funval).split('e')
    res[0] = res[0].replace('.', '')
    res[1] = '%+d' % (int(res[1]) - 1)
    return r'\textit{%s}' % 'e'.join(res)

def writeFEvals(fevals, precision='.2'):
    """Returns string representation of a number of function evaluations to use
    in a table.
    """
    tmp = (('%' + precision + 'g') % fevals)
    res = tmp.split('e')
    if len(res) > 1:
        res[1] = '%d' % int(res[1])
        res = '%s' % 'e'.join(res)
        pr2 = str(float(precision) + .2)
        #res2 = (('%' + pr2 + 'g') % fevals)
        res2 = (('%' + pr2 + 'g') % float(tmp))
        # To have the same number of significant digits.
        if len(res) >= len(res2):
            res = res2
    else:
        res = res[0]
    return res

def onealg(dsList, allmintarget, allertbest):
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
                            set_trace() # TODO: problem with the instances... MCS!
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
            set_trace()
        line = [numpy.inf,
                float(sum(i['unsoltrials'] for i in unsolved))/sum(i['nbtrials'] for i in unsolved) * 100,
                float(sum(i['unsolinstances'] for i in unsolved))/sum(i['nbinstances'] for i in unsolved) * 100,
                sum(list(i['unsolved'] for i in unsolved)), len(dsList), unsolvedrl[0]]
        line.extend(prctile(unsolvedrl, [10, 25, 50, 75, 90]))
        table.append(line)

    return table

def tableonealg(dsList, allmintarget, allertbest, sortedAlgs=None,
                outputdir='.'):
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

def tablemanyalgonefunc(dsList, allmintarget, allertbest, sortedAlgs=None,
                        outputdir='.'):
    dictDim = dsList.dictByDim()
    widthtable = 3 # Put in as global? 3 functions wide
    # TODO: split the generation of the tables from their formatting/output
    # ... if its possible
    for d, dentries in dictDim.iteritems():
        dictFunc = dentries.dictByFunc()
        dictAlg = dentries.dictByAlg()
        # Summary table: multiple algorithm for each function
        funcs = sorted(dictFunc.keys())
        nbtarget = len(allmintarget)
        stargets = sorted(allmintarget.keys())
        #groups = [[1, 2], [3], [4], [5, 6], [7], [8, 9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21, 22], [23], [24]]
        #groups = list(funcs[i*widthtable:(i+1)*widthtable] for i in range(len(funcs)/widthtable + 1))
        groups = list([i] for i in funcs)
        #set_trace()
        for numgroup, g in enumerate(groups):
            if not g:
                continue

            algnames = []
            table = []
            replacement = []
            for alg in sortedAlgs:
                curline = []
                replacementLine = []
                # Regroup entries by algorithm
                algentries = DataSetList()
                for a in alg:
                    if dictAlg.has_key(a):
                        algentries.extend(dictAlg[a])
                if not algentries:
                    continue
                algnames.append(writeLabels(algPlotInfos[alg[0]]['label']))
                dictF = algentries.dictByFunc()
                for func in g:
                    isLastInfoWritten = False
                    try:
                        entry = dictF[func]
                    except KeyError:
                        continue
                    try:
                        entry = entry[0]
                    except:
                        raise Usage('oops too many entries')

                    for t in stargets:
                        try:
                            if numpy.isnan(allmintarget[t][(func, d)]):
                                continue
                        except KeyError:
                            continue
                        try:
                            curline.append(entry.ert[entry.target<=allmintarget[t][(func, d)]][0]/allertbest[t][(func, d)])
                            replacementLine.append('')
                        except LookupError: #IndexError, KeyError:
                            curline.append(numpy.inf)
                            if not isLastInfoWritten:
                                replacementLine.append(r'%s\textit{/%s}' % (writeFunVal(numpy.median(entry.finalfunvals)), writeFEvals(numpy.median(entry.maxevals)/entry.dim, precision='.1')))
                                isLastInfoWritten = True
                            else:
                                replacementLine.append('.')
                replacement.append(replacementLine)
                table.append(curline)

            try:
                table = numpy.array(table)
            except ValueError:
                set_trace()
            # Process data
            boldface = sortColumns(table, maxRank=3)

            # Format data
            lines = [r'\begin{tabular}{c', '', r'$\Delta$ftarget', r'ERT$_{\textrm{best}}$/D']
            for func in g:
                curtargets = []
                for t in stargets:
                    try:
                        if numpy.isfinite(allmintarget[t][(func, d)]):
                            curtargets.append(t)
                    except KeyError:
                        continue
                lines[0] += '|' + len(curtargets) * 'c'
                lines[1] += (r' & \multicolumn{%d}{|c|}{f%d}' % (len(curtargets), func))

                for t in curtargets:
                    try:
                        lines[2] += (r'& %1.0e' % allmintarget[t][(func, d)])
                    except KeyError:
                        lines[2] += (r'& .')
                    try:
                        lines[3] += (r'& %s' % (writeFEvals(float(allertbest[t][(func, d)])/d, '.3')))
                    except KeyError:
                        lines[3] += (r'& .')

            lines[0] += '|}'
            lines[1] += r'\\'
            lines[2] += r'\\'
            lines[3] += r'\\'
            lines.append(r'\hline')

            for i, line in enumerate(table):
                if lines[-1] != r'\hline':
                    lines[-1] += r'\\'
                tmpstr = '%s' % algnames[i]
                # Regroup entries by algorithm
                dictF = algentries.dictByFunc()
                for j in range(len(line)):
                    if replacement[i][j]:
                        tmp = '%s' % replacement[i][j]
                    else:
                        tmp = '%s' % writeFEvals(line[j])

                    if i in boldface[j] or line[j] < 3:
                        tmp = r'\textbf{' + tmp + '}'
                    tmpstr += ' & ' + tmp

                lines.append(tmpstr)

            lines.append(r'\end{tabular}')
            f = open(os.path.join(outputdir, 'pptablef%d_%02dD.tex' % (numgroup, d)), 'w')
            f.write('\n'.join(lines) + '\n')
            f.close()
