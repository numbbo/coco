#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process data and generates some comparison results from pickle data files
   only.

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

from bbob_pproc.compall import ppperfprof, pptables
from bbob_pproc.compall import organizeRTDpictures, dataoutput
from bbob_pproc.compall.dataoutput import algLongInfos, algPlotInfos
from bbob_pproc.pproc import DataSetList
import matplotlib.pyplot as plt

# GLOBAL VARIABLES used in the routines defining desired output for BBOB 2009.
constant_target_function_values = (1e1, 1e0, 1e-1, 1e-3, 1e-5, 1e-7)
tableconstant_target_function_values = [1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7]
instancesOfInterest = {1:3, 2:3, 3:3, 4:3, 5:3}
#Deterministic instance of interest: only one trial is required.
instancesOfInterestDet = {1:1, 2:1, 3:1, 4:1, 5:1}
figformat = ('png', 'pdf')

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
    dictDim = dsList.dictByDim()
    targets = tableconstant_target_function_values

    for d, dimentries in dictDim.iteritems():
        dictFunc = dimentries.dictByFunc()
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
    """
    Keyword arguments:
    argv -- list of strings containing options and arguments. If not provided,
    sys.argv is accessed.

    argv must list folders containing pickle files.
    Furthermore, argv can begin with, in any order, facultative option
    flags listed below.

        -h, --help

            display this message

        -v, --verbose
 
            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

        -o, --output-dir OUTPUTDIR

            change the default output directory ('defaultoutputdirectory') to
            OUTPUTDIR

        --noise-free, --noisy

            restrain the post-processing to part of the data set only. Actually
            quicken the post-processing since it loads only part of the pickle
            files.

        --tab-only, --perfprof-only

            these options can be used to output respectively the comparison
            tex tables or the performance profiles only.
            A combination of any two of these options results in
            no output.

    Exceptions raised:
    Usage -- Gives back a usage message.

    Examples:

    * Calling the runcompall.py interface from the command line:

        $ python bbob_pproc/runcompall.py -v


    * Loading this package and calling the main from the command line
      (requires that the path to this package is in python search path):

        $ python -m bbob_pproc -h


    This will print out this help message.

    * From the python interactive shell (requires that the path to this
      package is in python search path):

        >>> import bbob_pproc
        >>> bbob_pproc.runcompall.main('-o outputfolder folder1'.split())

    This will execute the post-processing on the pickle files found in folder1.
    The -o option changes the output folder from the default ppdata to
    outputfolder.

    If you need to process new data, you must add a line in the file
    algorithmshortinfos.txt
    The line in question must have 4 fields separated by colon (:) character.
    The 1st must be the name of the folder which will contain the
    post-processed pickle data file, the 2nd is the exact string used as algId
    in the info files, the 3rd is the exact string for the comment. The 4th
    will be a python dictionary which will be use for the plotting.
    If different comment lines (3rd field) have been used for a single
    algorithm, there should be a line in algorithmshortinfos.txt corresponding
    to each of these.

    * Generate post-processing data for some algorithms:

        $ python runcompall.py AMALGAM BFGS CMA-ES

    """

    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, "hvpo:",
                                       ["help", "output-dir=", "noisy",
                                        "noise-free", "write-pickles",
                                        "perfprof-only", "tab-only", 
                                        "targets=", "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        verbose = False
        outputdir = 'defaultoutputdirectory'
        isWritePickle = False
        isNoisy = False
        isNoiseFree = False
        targets = False

        isPer = True
        isTab = True

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

        #Get only pickles!
        tmpargs = []
        sortedAlgs = []
        for i in args:
            #TODO: Check that i is a valid directory
            if not os.path.exists(i):
                warntxt = ('The folder %s does not exist.' % i)
                warnings.warn(warntxt)
                continue

            if isNoisy and isNoiseFree:
                ext = "*.pickle"
            elif isNoisy:
                ext = "*f1*.pickle"
            elif isNoiseFree:
                ext = "*f0*.pickle"
            else:
                ext = "*.pickle"

            tmpargs.extend(glob.glob(os.path.join(i, ext)))
            # remove trailing slashes and keep only the folder name which is
            # supposed to be the algorithm name.
            tmpalg = os.path.split(i.rstrip(os.path.sep))[1]
            if not tmpalg in algLongInfos:
                warntxt = ('The algorithm %s is not an entry in' %(tmpalg)
                           + '%s.' %(os.path.join([os.path.split(__file__)[0],
                                                    'algorithmshortinfos.txt']))
                           + 'An entry will be created.' )
                warnings.warn(warntxt)
                tmpdsList = DataSetList(glob.glob(os.path.join(i, ext)),
                                        verbose=False)
                tmpdsList = tmpdsList.dictByAlg()
                for alg in tmpdsList:
                    dataoutput.updateAlgorithmInfo(alg)
            sortedAlgs.append(algLongInfos[tmpalg])

        dsList = DataSetList(tmpargs, verbose=verbose)

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

            if (dict((j, i.itrials.count(j)) for j in set(i.itrials)) <
                tmpInstancesOfInterest):
                warnings.warn('The data of %s do not list ' %(i) +
                              'the correct instances ' +
                              'of function F%d or the ' %(i.funcId) +
                              'correct number of trials for each.')

        # group targets:
        dictTarget = {}
        for t in constant_target_function_values:
            tmpdict = dict.fromkeys(((f, d) for f in range(0, 25) + range(101, 131) for d in (2, 3, 5, 10, 20, 40)), t)
            stmp = 'E'
            if t == 1:
                stmp = 'E-'
            # dictTarget['_f' + stmp + '%2.1f' % numpy.log10(t)] = (tmpdict, )
            dictTarget['_f' + stmp + '%02d' % numpy.log10(t)] = (tmpdict, )
            dictTarget.setdefault('_allfs', []).append(tmpdict)

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
            if verbose:
                print 'Folder %s was created.' % (outputdir)

        # Performance profiles
        if isPer:
            dictDim = dsList.dictByDim()
            for d, entries in dictDim.iteritems():
                for k, t in dictTarget.iteritems():
                    ppperfprof.main(entries, target=t, order=sortedAlgs,
                                    plotArgs=algPlotInfos,
                                    outputdir=outputdir,
                                    info=('%02d%s' % (d, k)),
                                    fileFormat=figformat,
                                    verbose=verbose)
            organizeRTDpictures.do(outputdir)

        allmintarget, allertbest = detTarget(dsList)

        if isTab:
            pptables.tablemanyalgonefunc(dsList, allmintarget, allertbest,
                                         sortedAlgs, outputdir)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
   sys.exit(main())
