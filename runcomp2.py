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
from bbob_pproc.compall import dataoutput
from bbob_pproc.compall.dataoutput import algLongInfos, algPlotInfos

# GLOBAL VARIABLES used in the routines defining desired output  for BBOB 2009.
instancesOfInterest = {1:3, 2:3, 3:3, 4:3, 5:3}
instancesOfInterestDet = {1:1, 2:1, 3:1, 4:1, 5:1}
figformat = ('png', 'pdf')

#figValsOfInterest = (10, 1e-1, 1e-4, 1e-8)
figValsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8)
figDimsOfInterest = (2, 3, 5, 10, 20, 40)

rldDimsOfInterest = (2, 3, 5, 10, 20, 40)
#rldValsOfInterest = (1e-4, 1e-8)
rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8)
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

    argv should list names of folders containing pickle files. Furthermore,
    argv can begin with, in any order, facultative option flags listed below.

        -h, --help

            display this message

        -v, --verbose

            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

        -o, --output-dir OUTPUTDIR

            change the default output directory ('cmpdata') to OUTPUTDIR

        --fig-only, --rld-only

            these options can be used to output respectively the ERT graphs
            figures, run length distribution figures only. A combination of
            these options results in no output.

    Exceptions raised:
    Usage -- Gives back a usage message.

    Examples:

    * Calling the run.py interface from the command line:

        $ python bbob_pproc/run.py -v CMA-ES RANDOMSEARCH

    will post-process the pickle files of folder CMA-ES and RANDOMSEARCH. The
    result will appear in folder cmpdata. The -v option adds verbosity.

    * From the python interactive shell (requires that the path to this
      package is in python search path):

        >>> import bbob_pproc
        >>> bbob_pproc.main('-o outputfolder PSO DEPSO'.split())

    This will execute the post-processing on the pickle files found in folder
    PSO and DEPSO. The -o option changes the output folder from the default
    cmpdata to outputfolder.

    """

    if argv is None:
        argv = sys.argv[1:]
        # The zero-th input argument which is the name of the calling script is
        # disregarded.

    try:

        try:
            opts, args = getopt.getopt(argv, "hvo:",
                                       ["help", "output-dir",
                                        "fig-only", "rld-only",
                                        "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        isfigure = True
        isrldistr = True
        isNoisy = False
        isNoiseFree = True # Discern noisy and noisefree data?
        verbose = False
        outputdir = 'cmpdata'

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
                break
            except KeyError:
                pass

        dsList1 = DataSetList()
        for elem in sortedAlgs[1]:
            try:
                dsList1.extend(dictAlg[elem])
                break
            except KeyError:
                pass

        #set_trace()

        #for i, entry in enumerate(sortedAlgs): #Nota: key is sortedAlgs
            #print "Alg%d is: %s" % (i, entry)

        if isfigure or isrldistr:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if verbose:
                    print 'Folder %s was created.' % (outputdir)

        if isfigure:
            ppfig2.main(dsList0, dsList1,
                        figDimsOfInterest, outputdir, 1e-8, verbose)

        if isrldistr:
            dictDim0 = dsList0.dictByDim()
            dictDim1 = dsList1.dictByDim()

            print "ECDF absolute target graphs",
            #set_trace()
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
            print "done."

            print "ECDF relative target graphs",
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
            print "done."


        #if verbose:
            #tmp = []
            #tmp.extend(tabValsOfInterest)
            #tmp.extend(figValsOfInterest)
            #tmp.extend(rldValsOfInterest)
            #if indexEntries:
                #print ('Overall ps = %g\n'
                       #% indexEntries.successProbability(min(tmp)))

        if isfigure or isrldistr:
            print "Output data written to folder %s." % outputdir

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2


if __name__ == "__main__":
   sys.exit(main())
