#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for the comparison of 2 algorithms
Synopsis:
    python path_to_folder/bbob_pproc/runcomp2.py [OPTIONS] FOLDER_NAME1 FOLDER_NAME2...
Help:
    python path_to_folder/bbob_pproc/runcomp2.py -h

"""

from __future__ import absolute_import

import os
import sys
import glob
import warnings
import getopt
from pdb import set_trace
import matplotlib
matplotlib.use('Agg') # To avoid window popup and use without X forwarding

# Add the path to bbob_pproc
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc.pproc import DataSetList
from bbob_pproc.comp2 import ppfig2, pprldistr2
from bbob_pproc import dataoutput, pprldistr
from bbob_pproc.dataoutput import algLongInfos, algPlotInfos

# GLOBAL VARIABLES used in the routines defining desired output  for BBOB 2009.
instancesOfInterest = {1:3, 2:3, 3:3, 4:3, 5:3}
instancesOfInterestDet = {1:1, 2:1, 3:1, 4:1, 5:1}
instancesOfInterest2010 = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,
                           11:1, 12:1, 13:1, 14:1, 15:1}

#figValsOfInterest = (10, 1e-1, 1e-4, 1e-8)
figValsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8)
figDimsOfInterest = (2, 3, 5, 10, 20, 40)

rldDimsOfInterest = (2, 3, 5, 10, 20, 40)
#rldValsOfInterest = (1e-4, 1e-8)
#rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8)
rldValsOfInterest = ({1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10,
                      9: 10, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10,
                      16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10,
                      23: 10, 24: 10, 101: 10, 102: 10, 103: 10, 104: 10,
                      105: 10, 106: 10, 107: 10, 108: 10, 109: 10, 110: 10,
                      111: 10, 112: 10, 113: 10, 114: 10, 115: 10, 116: 10,
                      117: 10, 118: 10, 119: 10, 120: 10, 121: 10, 122: 10,
                      123: 10, 124: 10, 125: 10, 126: 10, 127: 10, 128: 10,
                      129: 10, 130: 10},
                      {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1,
                       8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1, 12: 0.1, 13: 0.1,
                       14: 0.1, 15: 0.1, 16: 0.1, 17: 0.1, 18: 0.1, 19: 0.1,
                       20: 0.1, 21: 0.1, 22: 0.1, 23: 0.1, 24: 0.1, 101: 0.1,
                       102: 0.1, 103: 0.1, 104: 0.1, 105: 0.1, 106: 0.1,
                       107: 0.1, 108: 0.1, 109: 0.1, 110: 0.1, 111: 0.1,
                       112: 0.1, 113: 0.1, 114: 0.1, 115: 0.1, 116: 0.1,
                       117: 0.1, 118: 0.1, 119: 0.1, 120: 0.1, 121: 0.1,
                       122: 0.1, 123: 0.1, 124: 0.1, 125: 0.1, 126: 0.1,
                       127: 0.1, 128: 0.1, 129: 0.1, 130: 0.1},
                      {1: 1e-04, 2: 1e-04, 3: 1e-04, 4: 1e-04, 5: 1e-04,
                       6: 1e-04, 7: 1e-04, 8: 1e-04, 9: 1e-04, 10: 1e-04,
                       11: 1e-04, 12: 1e-04, 13: 1e-04, 14: 1e-04, 15: 1e-04,
                       16: 1e-04, 17: 1e-04, 18: 1e-04, 19: 1e-04, 20: 1e-04,
                       21: 1e-04, 22: 1e-04, 23: 1e-04, 24: 1e-04, 101: 1e-04,
                       102: 1e-04, 103: 1e-04, 104: 1e-04, 105: 1e-04,
                       106: 1e-04, 107: 1e-04, 108: 1e-04, 109: 1e-04,
                       110: 1e-04, 111: 1e-04, 112: 1e-04, 113: 1e-04,
                       114: 1e-04, 115: 1e-04, 116: 1e-04, 117: 1e-04,
                       118: 1e-04, 119: 1e-04, 120: 1e-04, 121: 1e-04,
                       122: 1e-04, 123: 1e-04, 124: 1e-04, 125: 1e-04,
                       126: 1e-04, 127: 1e-04, 128: 1e-04, 129: 1e-04,
                       130: 1e-04},
                      {1: 1e-08, 2: 1e-08, 3: 1e-08, 4: 1e-08, 5: 1e-08,
                       6: 1e-08, 7: 1e-08, 8: 1e-08, 9: 1e-08, 10: 1e-08,
                       11: 1e-08, 12: 1e-08, 13: 1e-08, 14: 1e-08, 15: 1e-08,
                       16: 1e-08, 17: 1e-08, 18: 1e-08, 19: 1e-08, 20: 1e-08,
                       21: 1e-08, 22: 1e-08, 23: 1e-08, 24: 1e-08, 101: 1e-08,
                       102: 1e-08, 103: 1e-08, 104: 1e-08, 105: 1e-08,
                       106: 1e-08, 107: 1e-08, 108: 1e-08, 109: 1e-08,
                       110: 1e-08, 111: 1e-08, 112: 1e-08, 113: 1e-08,
                       114: 1e-08, 115: 1e-08, 116: 1e-08, 117: 1e-08,
                       118: 1e-08, 119: 1e-08, 120: 1e-08, 121: 1e-08,
                       122: 1e-08, 123: 1e-08, 124: 1e-08, 125: 1e-08,
                       126: 1e-08, 127: 1e-08, 128: 1e-08, 129: 1e-08,
                       130: 1e-08})
#Put backward to have the legend in the same order as the lines.

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


#FUNCTION DEFINITIONS

def usage():
    print main.__doc__


def main(argv=None):
    """Generates from BBOB experiment data sets of two algorithms some outputs.

    Keyword arguments:
    argv -- list of strings containing options and arguments. If not given,
    sys.argv is accessed.

    argv must list folders containing pickle files. Each of these folders
    should correspond to the data of one algorithm and should be listed in
    algorithmshortinfos.txt, a file from the bbob_pproc package listing the
    information of various algorithms treated using bbob_pproc.dataoutput

    Furthermore, argv can begin with, in any order, facultative option flags
    listed below.

        -h, --help

            display this message

        -v, --verbose

            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

        -o, --output-dir OUTPUTDIR

            change the default output directory ('cmp2data') to OUTPUTDIR

        --noise-free, --noisy

            restrain the post-processing to part of the data set only. Actually
            quicken the post-processing since it loads only part of the pickle
            files.

        --fig-only, --rld-only

            these options can be used to output respectively the ERT graphs
            figures, run length distribution figures only. A combination of
            these options results in no output.

    Exceptions raised:
    Usage -- Gives back a usage message.

    Examples:

    * Calling the runcomp2.py interface from the command line:

        $ python bbob_pproc/runcomp2.py -v CMA-ES RANDOMSEARCH

    will post-process the pickle files of folder CMA-ES and RANDOMSEARCH. The
    result will appear in folder cmp2data. The -v option adds verbosity.

    * From the python interactive shell (requires that the path to this
      package is in python search path):

        >>> from bbob_pproc import runcomp2
        >>> runcomp2.main('-o outputfolder PSO DEPSO'.split())

    This will execute the post-processing on the pickle files found in folder
    PSO and DEPSO. The -o option changes the output folder from the default
    cmp2data to outputfolder.

    """

    if argv is None:
        argv = sys.argv[1:]
        # The zero-th input argument which is the name of the calling script is
        # disregarded.

    try:

        try:
            opts, args = getopt.getopt(argv, "hvo:",
                                       ["help", "output-dir", "noisy",
                                        "noise-free", "fig-only", "rld-only",
                                        "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        isfigure = True
        isrldistr = True
        isNoisy = False
        isNoiseFree = False # Discern noisy and noisefree data?
        verbose = False
        outputdir = 'cmp2data'

        #Process options
        for o, a in opts:
            if o in ("-v","--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-o", "--output-dir"):
                outputdir = a
            elif o == "--fig-only":
                isrldistr = False
            elif o == "--rld-only":
                isfigure = False
            elif o == "--noisy":
                isNoisy = True
            elif o == "--noise-free":
                isNoiseFree = True
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

            if not (isNoisy ^ isNoiseFree):
                ext = "*.pickle"
            elif isNoisy:
                ext = "*f1*.pickle"
            elif isNoiseFree:
                ext = "*f0*.pickle"

            tmpargs.extend(glob.glob(os.path.join(i, ext)))
            # remove trailing slashes and keep only the folder name which is
            # supposed to be the algorithm name.
            tmpalg = os.path.split(i.rstrip(os.path.sep))[1]
            if not dataoutput.isListed(tmpalg):
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

            if ((dict((j, i.itrials.count(j)) for j in set(i.itrials)) <
                tmpInstancesOfInterest) and
                (dict((j, i.itrials.count(j)) for j in set(i.itrials)) <
                instancesOfInterest2010)):
                warnings.warn('The data of %s do not list ' %(i) +
                              'the correct instances ' +
                              'of function F%d or the ' %(i.funcId) +
                              'correct number of trials for each.')

        if len(sortedAlgs) < 2:
            Usage('Expect data from two different algorithms, could only find '
                  'one.')
            sys.exit()
        elif len(sortedAlgs) > 2:
            warnings.warn('Data with multiple algId %s ' % (sortedAlgs) +
                          'were found, the first two among will be processed.')

        # Group by algorithm
        dictAlg = dsList.dictByAlg()
        dsList0 = DataSetList()
        for elem in sortedAlgs[0]:
            try:
                dsList0.extend(dictAlg[elem])
            except KeyError:
                pass
        #set_trace()

        dsList1 = DataSetList()
        for elem in sortedAlgs[1]:
            try:
                dsList1.extend(dictAlg[elem])
            except KeyError:
                pass

        #for i, entry in enumerate(sortedAlgs): #Nota: key is sortedAlgs
            #print "Alg%d is: %s" % (i, entry)

        if isfigure or isrldistr:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if verbose:
                    print 'Folder %s was created.' % (outputdir)

        if isfigure:
            ppfig2.main(dsList0, dsList1, outputdir, 1e-8, verbose)
            if verbose:
                print "log ERT1/ERT0 vs target function values done."

        if isrldistr:
            dictFN0 = dsList0.dictByNoise()
            dictFN1 = dsList1.dictByNoise()
            if len(dictFN0) > 1 or len(dictFN1) > 1:
                warnings.warn('Data for functions from both the noisy and '
                              'non-noisy testbeds have been found. Their '
                              'results will be mixed in the "all functions"'
                              'ECDF figures.')

            dictDim0 = dsList0.dictByDim()
            dictDim1 = dsList1.dictByDim()

            for dim in set(dictDim0.keys()) | set(dictDim1.keys()):
                if dim in rldDimsOfInterest:
                    try:
                        pprldistr2.main(dictDim0[dim], dictDim1[dim],
                                        rldValsOfInterest, False,
                                        outputdir, 'dim%02dall' % dim, verbose)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.'
                                      % (dim))
                        continue

                    dictFG0 = dictDim0[dim].dictByFuncGroup()
                    dictFG1 = dictDim1[dim].dictByFuncGroup()

                    for fGroup in set(dictFG0.keys()) | set(dictFG1.keys()):
                        pprldistr2.main(dictFG0[fGroup], dictFG1[fGroup],
                                        rldValsOfInterest, False,
                                        outputdir, 'dim%02d%s' % (dim, fGroup),
                                        verbose)

                    dictFN0 = dictDim0[dim].dictByNoise()
                    dictFN1 = dictDim1[dim].dictByNoise()

                    for fGroup in set(dictFN0.keys()) | set(dictFN1.keys()):
                        pprldistr2.main(dictFN0[fGroup], dictFN1[fGroup],
                                        rldValsOfInterest, False, outputdir,
                                        'dim%02d%s' % (dim, fGroup),
                                        verbose)
            if verbose:
                print "ECDF absolute target graphs done."

            for dim in set(dictDim0.keys()) | set(dictDim1.keys()):
                if dim in rldDimsOfInterest:
                    try:

                        pprldistr2.main(dictDim0[dim], dictDim1[dim], None,
                                        True, outputdir, 'dim%02dall' % dim,
                                        verbose)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.'
                                      % (dim))
                        continue

                    dictFG0 = dictDim0[dim].dictByFuncGroup()
                    dictFG1 = dictDim1[dim].dictByFuncGroup()

                    for fGroup in set(dictFG0.keys()) | set(dictFG1.keys()):
                        pprldistr2.main(dictFG0[fGroup], dictFG1[fGroup], None,
                                        True, outputdir,
                                        'dim%02d%s' % (dim, fGroup), verbose)

                    dictFN0 = dictDim0[dim].dictByNoise()
                    dictFN1 = dictDim1[dim].dictByNoise()
                    for fGroup in set(dictFN0.keys()) | set(dictFN1.keys()):
                        pprldistr2.main(dictFN0[fGroup], dictFN1[fGroup],
                                        None, True, outputdir,
                                        'dim%02d%s' % (dim, fGroup), verbose)

            if verbose:
                print "ECDF relative target graphs done."

            for dim in set(dictDim0.keys()) | set(dictDim1.keys()):
                pprldistr.fmax = None #Resetting the max final value
                pprldistr.evalfmax = None #Resetting the max #fevalsfactor
                if dim in rldDimsOfInterest:
                    try:
                        pprldistr.comp(dictDim0[dim], dictDim1[dim],
                                       rldValsOfInterest, True,
                                       outputdir, 'dim%02dall' % dim, verbose)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.'
                                      % (dim))
                        continue

                    dictFG0 = dictDim0[dim].dictByFuncGroup()
                    dictFG1 = dictDim1[dim].dictByFuncGroup()

                    for fGroup in set(dictFG0.keys()) | set(dictFG1.keys()):
                        pprldistr.comp(dictFG0[fGroup], dictFG1[fGroup],
                                       rldValsOfInterest, True, outputdir,
                                       'dim%02d%s' % (dim, fGroup), verbose)

                    dictFN0 = dictDim0[dim].dictByNoise()
                    dictFN1 = dictDim1[dim].dictByNoise()
                    for fGroup in set(dictFN0.keys()) | set(dictFN1.keys()):
                        pprldistr.comp(dictFN0[fGroup], dictFN1[fGroup],
                                       rldValsOfInterest, True, outputdir,
                                       'dim%02d%s' % (dim, fGroup), verbose)

            if verbose:
                print "ECDF dashed-solid graphs done."

        if verbose:
            if isfigure or isrldistr:
                print "Output data written to folder %s." % outputdir

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2


if __name__ == "__main__":
   sys.exit(main())
