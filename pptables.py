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
    '''Returns string representation of a function value to use in a table.'''
    str = ('%.1e' % funval).split('e')
    str[0] = str[0].replace('.', '')
    str[1] = '%+d' % (int(str[1]) - 1)
    return r'\textit{%s}' % 'e'.join(str)

def writeFEvals(fevals):
    '''Returns string representation of a number of function evaluations to use
    in a table.
    '''
    return ('%.2g' % fevals).replace('+', '')

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

def manyalg(dsList):
    pass

def tablemanyalg(dsList, allmintarget, allertbest, sortedAlgs=None, outputdir='.'):
    dictDim = dsList.dictByDim()
    for d, dentries in dictDim.iteritems():
        dictAlg = dentries.dictByAlg()
        # Multiple algorithms table.
        nbtarget = len(allmintarget)
        lines = [r'\begin{tabular}{c@{}' + 'c@{/}c@{(}c@{) }'*nbtarget + '}']
        stargets = sorted(allmintarget.keys())
        tmpstr = 'evals/D'
        for t in stargets:
            nbsolved = sum(numpy.isfinite(list(allmintarget[t][i] for i in allmintarget[t] if i[1] == d)))
            #set_trace()
            tmpstr += (r' & \multicolumn{2}{c@{(}}{%.2g} & %d' % (t, nbsolved))
        lines.append(tmpstr)

        for alg in sortedAlgs:
            # Regroup entries by algorithm
            algentries = DataSetList()
            for i in alg:
                if dictAlg.has_key(i):
                    algentries.extend(dictAlg[i])
            if not algentries:
                continue
            lines[-1] += r'\\'
            tmpstr = algPlotInfos[alg[0]]['label'].replace('_', '\\_') # TODO: escape special latex characters
            for t in stargets:
                dictFunc = algentries.dictByFunc()
                erts = []
                for func, entry in dictFunc.iteritems():
                    try:
                        entry = entry[0]
                    except:
                        raise Usage('oops too many entries')
                    try:
                        erts.append(entry.ert[entry.target<=allmintarget[t][(func, d)]][0]/allertbest[t][(func, d)])
                    except LookupError:
                        erts.append(numpy.inf)
                if numpy.isfinite(erts).any():
                    med = numpy.median(erts)
                    if numpy.isinf(med):
                        tmpstr += r' & . '
                    else:
                        tmpstr += r' & %s ' % (writeFEvals(numpy.median(erts)))
                    tmpstr += '& %s & %d' % (writeFEvals(numpy.min(erts)), numpy.sum(numpy.isfinite(erts)))
                else:
                    tmpstr += (r' & . & . & 0')
            lines.append(tmpstr)
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
                                tmpstr += (r' & \textbf{%.3g} ' % tmp)
                            else:
                                tmpstr += (' & %.3g ' % tmp)
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

def summaryfunction(dsList):
    pass

def createTables(dsList):
    dictFunc = dsList.dictByFunc()

    if dictFunc.keys()[0] <= 100:
        targetsfile = '/users/dsa/ros/Desktop/coco/BBOB/final-submissions/ppdata/noisefreetarget.pickle'
    else:
        targetsfile = '/users/dsa/ros/Desktop/coco/BBOB/final-submissions/ppdata/noisytarget.pickle'

    f = open(targetsfile, 'r')
    algSet = pickle.load(f)
    allmintarget = pickle.load(f)
    allmedtarget = pickle.load(f)
    allertbest = pickle.load(f)
    f.close()

    dictAlg = dsList.dictByAlg()
    dictAlg2 = {}
    for i, j in dictAlg.iteritems():
        dictAlg2.setdefault(algShortInfos[i], []).extend(j)
    #set_trace()
    if len(dictAlg2) > 1:
        manyalg(dsList)
    else:
        table = onealg(dsList, allmintarget, allertbest)

    #header2 = ('evals/D', '\%trials', '\%inst', 'fcts', 'best', '10', '25', 'med', '75', '90')

    f = open('test', 'w')
    f.write(r'\begin{tabular}{cccccccccc}'+'\n')
    f.write(r'\multicolumn{4}{c}{Solved} & \multicolumn{6}{c}{ERT/ERT\_best} \\'+'\n')
    #set_trace()
    for i in table:
        pptex.writeArray(f, i, ['%d', '%d', '%d', '%s', '%1.1e', '%1.1e', '%1.1e', '%1.1e', '%1.1e', '%1.1e'],
        'scriptstyle')
    f.write(r'\end{tabular}'+'\n')
    f.close()
    return table
