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

def writeFunVal(funval):
    """Returns string representation of a function value to use in a table."""
    str = ('%.1e' % funval).split('e')
    str[0] = str[0].replace('.', '')
    str[1] = '%+d' % (int(str[1]) - 1)
    return r'\textit{%s}' % 'e'.join(str)

def writeFEvals(fevals):
    """Returns string representation of a number of function evaluations to use
    in a table.
    """
    str = ('%.2g' % fevals).split('e')
    if len(str) > 1:
        str[1] = '%d' % int(str[1])
    return '%s' % 'e'.join(str)

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
                    for idx in dictinstance.values():
                        try:
                            if numpy.isfinite(list(tmpline[j+1] for j in idx)).any():
                                solinstances += 1
                        except IndexError:
                            set_trace() # TODO: problem with the instances... MCS!
                    tmp['unsolinstances'] = nbinstances - solinstances
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
                    tmpstr = r'$\infty$ & '
                else:
                    tmpstr = format2[0] % i[0] + ' & '

                tmpstr += ' & '.join(format2[1:]) % tuple(i[1:])
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
    limitRank = 3

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
            algnames.append(algPlotInfos[alg[0]]['label'].replace('_', '\\_')) # TODO: escape special latex characters
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
        boldface = []
        for i in range(numpy.shape(table)[1]):
            tmp = table[:, i].argsort()
            prevValue = 0
            tmp2 = []
            for j in tmp:
                if table[j, i] != prevValue:
                    tmp2.extend(numpy.where(table[:, i] == table[j, i])[0])
                    prevValue = table[j, i]
                if len(tmp2) >= limitRank:
                    break
            boldface.append(tmp2)
            #set_trace()

        # format the data
        lines = [r'\begin{tabular}{c' + 'c@{/}c@{(}c@{) }'*len(targetkept) + '}']
        tmpstr = 'evals/D'
        for t in list(stargets[i] for i in targetkept):
            nbsolved = sum(numpy.isfinite(list(allmintarget[t][i] for i in allmintarget[t] if i[1] == d)))
            #set_trace()
            tmpstr += (r' & \multicolumn{2}{c@{(}}{%.2g} & %d' % (t, nbsolved))
        lines.append(tmpstr)

        for i in range(len(table)):
            lines[-1] += r'\\'
            curline = algnames[i]
            for j in range(len(table[i])):
                curline += ' & '
                if (j + 1) % 3 > 0: # the test may not be necessary
                    if numpy.isinf(table[i, j]):
                        tmpstr = '.'
                    else:
                        tmpstr = '%s' % (writeFEvals(table[i, j]))

                    if any(list(k == i for k in boldface[j])):
                        tmpstr = r'\textbf{' + tmpstr + '}'

                    curline += tmpstr
                else:
                    curline += '%d' % table[i, j] # nb solved.

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
        groups = list(funcs[i*widthtable:(1+i)*widthtable] for i in range(len(funcs)/widthtable + 1))
        for i, g in enumerate(groups):
            if not g:
                continue
            nbcols = {}
            lines = [r'\begin{tabular}{c', '', r'$\Delta$ftarget', r'ERT\_best/D']
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
                        lines[3] += (r'& %.3g' % (float(allertbest[t][(func, d)])/d))
                    except KeyError:
                        lines[3] += (r'& .')

            lines[0] += '|}'
            lines[1] += r'\\'
            lines[2] += r'\\'
            lines[3] += r'\\'

            for alg in sortedAlgs:
                tmpstr = ''
                # Regroup entries by algorithm
                algentries = DataSetList()
                for a in alg:
                    if dictAlg.has_key(a):
                        algentries.extend(dictAlg[a])
                if not algentries:
                    continue
                tmpstr += algPlotInfos[alg[0]]['label'].replace('_', '\\_') # TODO: escape special latex characters
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
                            tmp = entry.ert[entry.target<=allmintarget[t][(func, d)]][0]/allertbest[t][(func, d)]
                            if tmp < 3:
                                tmpstr += (r' & \textbf{%s} ' % writeFEvals(tmp))
                            else:
                                tmpstr += (' & %s ' % writeFEvals(tmp))
                        except LookupError: #IndexError, KeyError:
                            if not isLastInfoWritten:
                                tmpstr += (r' & %s\textit{/%.1g}' % (writeFunVal(numpy.median(entry.finalfunvals)), numpy.median(entry.maxevals)))
                                isLastInfoWritten = True
                            else:
                                tmpstr += (' & .')
                tmpstr += r'\\'
                lines.append(tmpstr)

            f = open(os.path.join(outputdir, 'pptablef%d_%02dD.tex' % (i, d)), 'w')
            lines[-1] = lines[-1][0:-2] # Take away the last line jump character
            lines.append(r'\end{tabular}')
            f.write('\n'.join(lines) + '\n')
            f.close()
