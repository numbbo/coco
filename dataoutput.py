#!/usr/bin/env python

"""Generates output either pickle files or elaborate data files. 1 file per
solver per function and per dimension (unit experiment).

"""

from __future__ import absolute_import

import os
import sys
import pickle
from bbob_pproc import pproc2

from pdb import set_trace

# Will read in this file where to put the pickle files.
infofilename = 'algorithmshortinfos.txt'
infofile = os.path.join(os.path.split(__file__)[0], infofilename)
algShortInfos = {}
algLongInfos = {}
isAlgorithminfosFound = True
try:
    f = open(infofile,'r')
    for line in f:
        if len(line) == 0 or line.startswith('%') or line.isspace() :
            continue
        algShortInfo, algId, comment = line.strip().split(':', 2)
        algShortInfos[(algId, comment)] = algShortInfo
        algLongInfos[algShortInfo] = (algId, comment)
    f.close()
except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    isAlgorithminfosFound = False
    print 'Could not find file', infofile, \
          'Will not generate any output.'

def updateAlgorithmInfo(alg):
    try:
        f = open(infofile, 'a')
        if not alg[0] in algLongInfos:
            algShortInfos[alg] = alg[0]
            algLongInfos[alg[0]] = alg
        else:
            raise Usage('Problem here')
    except:
        print 'There was a problem here'
    f.close()


def outputPickle(dsList, verbose=True):
    """Generates pickle files from a DataSetList."""
    dictAlg = dsList.dictByAlg()
    for alg, entries in dictAlg.iteritems():
        try:
            algShortInfos[alg]
        except KeyError:
            print '%s is not an entry in file: %s.' % (alg, infofilename)
            updateAlgorithmInfo(alg)

        if not os.path.exists(algShortInfos[alg]):
            os.mkdir(algShortInfos[alg])

        entries.pickle(outputdir=algShortInfos[alg], verbose=verbose)


def outputDataFiles(dsList, verbose=True):
    """Generates data files from a DataSetList."""
    dictAlg = dsList.dictByAlg()
    for alg, entries in dictAlg.iteritems():
        # TODO: entries.dataf(outputdiralgInfos[alg], verbose=verbose)
        pass
