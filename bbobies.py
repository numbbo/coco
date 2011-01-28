#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module based on runcompall.py used by BBOBies.

This module has not been updated for a while. Process data and generates some
comparison results.

   Synopsis:
      python path_to_folder/bbob_pproc/runcompall.py [OPTIONS] FOLDER_NAME...
    Help:
      python path_to_folder/bbob_pproc/runcompall.py -h

"""

from __future__ import absolute_import

import os
import sys
import glob
import getopt
import pickle
import tarfile
from pdb import set_trace
import warnings
import numpy

# Add the path to bbob_pproc
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc import dataoutput, pproc, pprldistr
from bbob_pproc.dataoutput import algPlotInfos
from bbob_pproc.pproc import DataSetList, processInputArgs
from bbob_pproc.compall import ppperfprof, pptables
from bbob_pproc.compall import organizeRTDpictures

# GLOBAL VARIABLES used in the routines defining desired output for BBOB 2009.
#single_target_function_values = (1e1, 1e0, 1e-1, 1e-3, 1e-5, 1e-7, 1e-8)  # one figure for each
#single_target_function_values = (1e-8,)  # one figure for each
##summarized_target_function_values = (1e0, 1e-1, 1e-3, 1e-5, 1e-7)   # all in one figure
#summarized_target_function_values = ()   # all in one figure
#tableconstant_target_function_values = [1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7]

# Deterministic instance of interest: only one trial is required.
instancesOfInterest = {1:3, 2:3, 3:3, 4:3, 5:3}
instancesOfInterestDet = {1:1, 2:1, 3:1, 4:1, 5:1}
instancesOfInterest2010 = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,
                           11:1, 12:1, 13:1, 14:1, 15:1}

algs2009 = ("ALPS", "AMALGAM", "BAYEDA", "BFGS", "Cauchy-EDA",
"BIPOP-CMA-ES", "CMA-ESPLUSSEL", "DASA", "DE-PSO", "DIRECT", "EDA-PSO",
"FULLNEWUOA", "G3PCX", "GA", "GLOBAL", "iAMALGAM", "IPOPSEPCMA", "LSfminbnd",
"LSstep", "MA-LS-CHAIN", "MCS", "NELDER", "NELDERDOERR", "NEWUOA", "ONEFIFTH",
"POEMS", "PSO", "PSO_Bounds", "RANDOMSEARCH", "Rosenbrock", "SNOBFIT",
"VNS")

algs2010 = ("1komma2", "1komma2mir", "1komma2mirser", "1komma2ser", "1komma4",
"1komma4mir", "1komma4mirser", "1komma4ser", "1plus1", "1plus2mirser", "ABC",
"AVGNEWUOA", "CMAEGS", "DE-F-AUC", "DEuniform", "IPOP-ACTCMA-ES",
"IPOP-CMA-ES", "MOS", "NBC-CMA", "NEWUOA", "PM-AdapSS-DE", "RCGA", "SPSA",
"oPOEMS", "pPOEMS")

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#FUNCTION DEFINITIONS

def detertbest(dsList, minFtarget):
    """Determines the best ert for a given traget function value."""
    erts = []
    ertbest = []
    for alg in dsList:
        idx = 0  # index of ert or target.
        for i, val in enumerate(minFtarget):
            try:
                erts[i]
            except IndexError:
                erts.append([])
            if numpy.isfinite(val):
                while (idx < len(alg.target) and alg.target[idx] > val):
                    idx += 1
                try:
                    erts[i].append(alg.ert[idx])
                except IndexError:
                    pass
                    #TODO: what value to put?
                    #erts[i].append(numpy.nan)

    for elem in erts:
        if not elem:
            ertbest.append(numpy.nan) # TODO: what value to put?
        else:
            ertbest.append(min(elem))
    return numpy.array(ertbest)

def detTarget(dsList):
    """Creates the data structure of the target function values.
    """

    allmintarget = {}
    allertbest = {}
    targets = tableconstant_target_function_values

    dictDim = {}
    for i in dsList:
        dictDim.setdefault(i.dim, []).append(i)

    for d, dimentries in dictDim.iteritems():
        dictFunc = {}
        for i in dimentries:
            dictFunc.setdefault(i.funcId, []).append(i)

        for f, funentries in dictFunc.iteritems():
            tmpertbest = detertbest(funentries, targets)
            for i in range(len(targets)):
               tmp = allmintarget.setdefault(-targets[i], {}) # Why the minus?
               tmp.setdefault((f, d), targets[i])

               tmp = allertbest.setdefault(-targets[i], {}) # Why the minus?
               tmp.setdefault((f, d), tmpertbest[i])

    return allmintarget, allertbest

def usage():
    print main.__doc__

def main(argv=None):

    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, "hvo:",
                                       ["help", "output-dir=", "noisy",
                                        "noise-free", "perfprof-only",
                                        "tab-only", "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        args = set(algs2009) | set(algs2010)

        if not (args):
            usage()
            sys.exit()

        verbose = False
        outputdir = 'bbobies'
        isNoisy = False
        isNoiseFree = False

        isPer = False
        isTab = False

        #Process options
        for o, a in opts:
            if o in ("-v","--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-o", "--output-dir"):
                outputdir = a
            elif o == "--noisy":
                isNoisy = True
            elif o == "--noise-free":
                isNoiseFree = True
            elif o == "--tab-only":
                isPer = False
                isEff = False
            elif o == "--perfprof-only":
                isEff = False
                isTab = False
            else:
                assert False, "unhandled option"

        if (not verbose):
            warnings.simplefilter('ignore')

        print ("BBOB Post-processing: will generate comparison " +
               "data in folder %s" % outputdir)
        print "  this might take several minutes."

        dsList, sortedAlgs, dictAlg = processInputArgs(args,
                                                       verbose=verbose)

        if not dsList:
            sys.exit()

        for i in dsList:
            if not i.dim in (2, 3, 5, 10, 20):
                continue
            # Deterministic algorithms
            if i.algId in ('Original DIRECT', ):
                tmpInstancesOfInterest = instancesOfInterestDet
            else:
                tmpInstancesOfInterest = instancesOfInterest

            if ((dict((j, i.itrials.count(j)) for j in set(i.itrials)) <
                tmpInstancesOfInterest) and
                (dict((j, i.itrials.count(j)) for j in set(i.itrials)) <
                instancesOfInterest2010)):
                warnings.warn('The data of %s do not list ' %(i) +
                              'the correct instances ' +
                              'of function F%d or the ' %(i.funcId) +
                              'correct number of trials for each.')

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
            if verbose:
                print 'Folder %s was created.' % (outputdir)

        # Generate RLD data
        dictres = {}
        for alg, tmpdictAlg in dictAlg.iteritems():
            tmpdict = dictres.setdefault(alg, {})
            for f, tmpdictFunc in tmpdictAlg.dictByFunc().iteritems():
                tmpdict1 = tmpdict.setdefault(f, {})
                for d, entries in tmpdictFunc.dictByDim().iteritems():
                    if len(entries) != 1:
                        raise Usage('Problem for alg %s, f%d, %d-D' % alg, f, d)
                    entry = entries[0]
                    tmp = entry.generateRLData([1e-08])[1e-08]
                    #set_trace()
                    tmpdict1.setdefault(d, (tmp, entry.maxevals))
        picklefilename = os.path.join(outputdir, 'testdata.pickle')
        f = open(picklefilename, 'w', 2)
        pickle.dump(dictres, f)
        f.close()

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())

