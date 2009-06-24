#!/usr/bin/env python

from __future__ import absolute_import

from pdb import set_trace
import pickle
import numpy
from bbob_pproc import pptex
from bbob_pproc.bootstrap import prctile
from bbob_pproc.dataoutput import algShortInfos


allmintarget = {}
allmedtarget = {}


def onealg(dsList, allmintarget, allertbest):
    header1 = ('solved', 'ERT/ERT_best')
    header2 = ('evals/D', '\%trials', '\%inst', 'fcts', 'best', '10', '25', 'med', '75', '90')
    table = []
    for t in sorted(allmintarget.keys()):
        erts = []
        soltrials = 0
        nbtrials = 0
        solinstances = 0
        nbinstances = 0
        solfcts = 0
        nbfcts = 0
        for entry in dsList:
            try:
                if numpy.isnan(allmintarget[t][(entry.funcId, entry.dim)]):
                    continue
            except KeyError:
                continue
            nbfcts += 1
            for l in range(len(entry.evals)):
                tmpline = entry.evals[l]
                if tmpline[0] < allmintarget[t][(entry.funcId, entry.dim)]:
                    solfcts += 1
                    soltrials += numpy.sum(numpy.isfinite(tmpline[1:]))
                    nbtrials += len(tmpline[1:])
                    #TODO: hard to read
                    dictinstance = entry.createDictInstance()
                    for idx in dictinstance.values():
                        if numpy.isfinite(list(tmpline[j+1] for j in idx)).any():
                            solinstances += 1
                    nbinstances += len(dictinstance)
                    erts.append(float(entry.ert[l]) / allertbest[t][(entry.funcId, entry.dim)])
                    break

        if len(erts) > 0:
            erts.sort()
            line = [t]
            line.extend((float(soltrials)/nbtrials*100., float(solinstances)/nbinstances*100.,
                         '%d/%d' % (solfcts, nbfcts)))
            line.append(erts[0])
            line.extend(prctile(erts, [10, 25, 50, 75, 90]))
            table.append(line)
    return table

def manyalg(dsList):
    pass

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
