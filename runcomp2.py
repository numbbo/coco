#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for the comparison of 2 algorithms.
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

# Add the path to bbob_pproc
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc.pproc import DataSetList, processInputArgs
from bbob_pproc.bbob2010 import pprldistr
from bbob_pproc.bbob2010.comp2 import ppfig2, pprldistr2, pptable2, ppscatter

import matplotlib.pyplot as plt

# GLOBAL VARIABLES used in the routines defining desired output for BBOB.
instancesOfInterest = {1:3, 2:3, 3:3, 4:3, 5:3}
instancesOfInterestDet = {1:1, 2:1, 3:1, 4:1, 5:1}
instancesOfInterest2010 = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,
                           11:1, 12:1, 13:1, 14:1, 15:1}

#figValsOfInterest = (10, 1e-1, 1e-4, 1e-8)
figValsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8)
figDimsOfInterest = (2, 3, 5, 10, 20, 40)

#rldDimsOfInterest = (2, 3, 5, 10, 20, 40)
rldDimsOfInterest = (5, 20)
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

tabDimsOfInterest = (5, 20)

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


#FUNCTION DEFINITIONS

def usage():
    print main.__doc__

def main(argv=None):
    """Generates some outputs from BBOB experiment data sets of two algorithms.

    Provided with some data, this routine outputs figure and TeX files in the
    folder 'cmp2data' needed for the compilation of the latex document
    templateBBOBcmparticle.tex. These output files will contain performance
    tables, performance scaling figures, scatter plot figures and empirical
    cumulative distribution figures. On subsequent executions, new files will
    be added to the output directory, overwriting existing files in the
    process.

    Keyword arguments:
    argv -- list of strings containing options and arguments. If not given,
    sys.argv is accessed.

    argv must list folders containing BBOB data files. Each of these folders
    should correspond to the data of ONE algorithm.

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

        --fig-only, --rld-only, --tab-only, --sca-only

            these options can be used to output respectively the ERT graphs
            figures, run length distribution figures or the comparison tables
            scatter plot figures only. Any combination of these options results
            in no output.

    Exceptions raised:
    Usage -- Gives back a usage message.

    Examples:

    * Calling the runcomp2.py interface from the command line:

        $ python bbob_pproc/runcomp2.py -v Alg0-baseline Alg1-of-interest

    will post-process the data from folders Alg0-baseline and Alg1-of-interest,
    the former containing data for the reference algorithm (zero-th) and the
    latter data for the algorithm of concern (first). The results will be
    output in folder cmp2data. The -v option adds verbosity.

    * From the python interactive shell (requires that the path to this
      package is in python search path):

        >>> from bbob_pproc import runcomp2
        >>> runcomp2.main('-o outputfolder PSO DEPSO'.split())

    This will execute the post-processing on the data found in folder
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
                                        "tab-only", "sca-only", "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        isfigure = True
        isrldistr = True
        istable = True
        isscatter = True
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
                istable = False
                isscatter = False
            elif o == "--rld-only":
                isfigure = False
                istable = False
                isscatter = False
            elif o == "--tab-only":
                isfigure = False
                isrldistr = False
                isscatter = False
            elif o == "--sca-only":
                isfigure = False
                isrldistr = False
                istable = False
            elif o == "--noisy":
                isNoisy = True
            elif o == "--noise-free":
                isNoiseFree = True
            else:
                assert False, "unhandled option"

        if (not verbose):
            warnings.simplefilter('ignore')

        print ("BBOB Post-processing: will generate comparison " +
               "data in folder %s" % outputdir)
        print "  this might take several minutes."

        dsList, sortedAlgs, dictAlg = processInputArgs(args, verbose=verbose)

        if not dsList:
            sys.exit()

        for i in dictAlg:
            if isNoisy and not isNoiseFree:
                dictAlg[i] = dictAlg[i].dictByNoise().get('nzall', DataSetList())
            if isNoiseFree and not isNoisy:
                dictAlg[i] = dictAlg[i].dictByNoise().get('noiselessall', DataSetList())

        for i in dsList:
            if not i.dim in (2, 3, 5, 10, 20):
                continue

            #### The following lines are BBOB 2009 checking.###################
            # Deterministic algorithms
            #if i.algId in ('Original DIRECT', ):
                #tmpInstancesOfInterest = instancesOfInterestDet
            #else:
                #tmpInstancesOfInterest = instancesOfInterest
            #if ((dict((j, i.itrials.count(j)) for j in set(i.itrials)) <
                #tmpInstancesOfInterest) and
                #(dict((j, i.itrials.count(j)) for j in set(i.itrials)) <
                #instancesOfInterest2010)):
                #warnings.warn('The data of %s do not list ' %(i) +
                              #'the correct instances ' +
                              #'of function F%d or the ' %(i.funcId) +
                              #'correct number of trials for each.')
            ###################################################################

            if (dict((j, i.itrials.count(j)) for j in set(i.itrials)) <
                instancesOfInterest2010):
                warnings.warn('The data of %s do not list ' %(i) +
                              'the correct instances ' +
                              'of function F%d.' %(i.funcId))

        if len(sortedAlgs) < 2:
            raise Usage('Expect data from two different algorithms, could ' +
                        'only find one.')
        elif len(sortedAlgs) > 2:
            #raise Usage('Expect data from two different algorithms, found ' +
            #            'more than two.')
            warnings.warn('Data from folders: %s ' % (sortedAlgs) +
                          'were found, the first two will be processed.')

        # Group by algorithm
        dsList0 = dictAlg[sortedAlgs[0]]
        if not dsList0:
            raise Usage('Could not find data for algorithm %s.' % (sortedAlgs[0]))
        #set_trace()

        dsList1 = dictAlg[sortedAlgs[1]]
        if not dsList1:
            raise Usage('Could not find data for algorithm %s.' % (sortedAlgs[0]))

        tmppath0, alg0name = os.path.split(sortedAlgs[0].rstrip(os.sep))
        tmppath1, alg1name = os.path.split(sortedAlgs[1].rstrip(os.sep))
        #Trick for having different algorithm names in the tables...
        #Does not really work.
        #while alg0name == alg1name:
        #    tmppath0, alg0name = os.path.split(tmppath0)
        #    tmppath1, alg1name = os.path.split(tmppath1)
        #
        #    if not tmppath0 and not tmppath1:
        #        break
        #    else:
        #        if not tmppath0:
        #            tmppath0 = alg0name
        #        if not tmppath1:
        #            tmppath1 = alg1name
        #assert alg0name != alg1name
        # should not be a problem, these are only used in the tables.
        for i in dsList0:
            i.algId = alg0name
        for i in dsList1:
            i.algId = alg1name

        #for i, entry in enumerate(sortedAlgs): #Nota: key is sortedAlgs
            #print "Alg%d is: %s" % (i, entry)

        if isfigure or isrldistr or istable or isscatter:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if verbose:
                    print 'Folder %s was created.' % (outputdir)

        plt.rc("axes", labelsize=24, titlesize=24)
        plt.rc("xtick", labelsize=20)
        plt.rc("ytick", labelsize=20)
        plt.rc("font", size=20)
        plt.rc("legend", fontsize=20)

        dictFN0 = dsList0.dictByNoise()
        dictFN1 = dsList1.dictByNoise()
        k0 = set(dictFN0.keys())
        k1 = set(dictFN1.keys())
        symdiff = k1 ^ k0
        if symdiff: # symmetric difference
            tmpdict = {}
            for i, noisegrp in enumerate(symdiff):
                if noisegrp == 'nzall':
                    tmp = 'noisy'
                elif noisegrp == 'noiselessall':
                    tmp = 'noiseless'

                if dictFN0.has_key(noisegrp):
                    tmp2 = sortedAlgs[0]
                elif dictFN1.has_key(noisegrp):
                    tmp2 = sortedAlgs[1]

                tmpdict.setdefault(tmp2, []).append(tmp)

            txt = []
            for i, j in tmpdict.iteritems():
                txt.append('Only input folder %s lists %s data.'
                            % (i, ' and '.join(j)))
            raise Usage('Data Mismatch: \n  ' + ' '.join(txt)
                        + '\nTry using --noise-free or --noisy flags.')

        if isfigure:
            ppfig2.main(dsList0, dsList1, 1e-8, outputdir, verbose)
            print "log ERT1/ERT0 vs target function values done."

        if isrldistr:
            if len(dictFN0) > 1 or len(dictFN1) > 1:
                warnings.warn('Data for functions from both the noisy and ' +
                              'non-noisy testbeds have been found. Their ' +
                              'results will be mixed in the "all functions" ' +
                              'ECDF figures.')

            dictDim0 = dsList0.dictByDim()
            dictDim1 = dsList1.dictByDim()

            for dim in set(dictDim0.keys()) | set(dictDim1.keys()):
                if dim in rldDimsOfInterest:
                    try:
                        pprldistr2.main2(dictDim0[dim], dictDim1[dim],
                                         rldValsOfInterest,
                                         outputdir, '%02dD_all' % dim, verbose)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.'
                                      % (dim))
                        continue

                    dictFG0 = dictDim0[dim].dictByFuncGroup()
                    dictFG1 = dictDim1[dim].dictByFuncGroup()

                    for fGroup in set(dictFG0.keys()) | set(dictFG1.keys()):
                        pprldistr2.main2(dictFG0[fGroup], dictFG1[fGroup],
                                         rldValsOfInterest, 
                                         outputdir, '%02dD_%s' % (dim, fGroup),
                                         verbose)

                    dictFN0 = dictDim0[dim].dictByNoise()
                    dictFN1 = dictDim1[dim].dictByNoise()

                    for fGroup in set(dictFN0.keys()) | set(dictFN1.keys()):
                        pprldistr2.main2(dictFN0[fGroup], dictFN1[fGroup],
                                         rldValsOfInterest, outputdir,
                                         '%02dD_%s' % (dim, fGroup),
                                         verbose)
            print "ECDF absolute target graphs done."

            #for dim in set(dictDim0.keys()) | set(dictDim1.keys()):
                #if dim in rldDimsOfInterest:
                    #try:

                        #pprldistr2.main(dictDim0[dim], dictDim1[dim], None,
                                        #True, outputdir, 'dim%02dall' % dim,
                                        #verbose)
                    #except KeyError:
                        #warnings.warn('Could not find some data in %d-D.'
                                      #% (dim))
                        #continue

                    #dictFG0 = dictDim0[dim].dictByFuncGroup()
                    #dictFG1 = dictDim1[dim].dictByFuncGroup()

                    #for fGroup in set(dictFG0.keys()) | set(dictFG1.keys()):
                        #pprldistr2.main(dictFG0[fGroup], dictFG1[fGroup], None,
                                        #True, outputdir,
                                        #'dim%02d%s' % (dim, fGroup), verbose)

                    #dictFN0 = dictDim0[dim].dictByNoise()
                    #dictFN1 = dictDim1[dim].dictByNoise()
                    #for fGroup in set(dictFN0.keys()) | set(dictFN1.keys()):
                        #pprldistr2.main(dictFN0[fGroup], dictFN1[fGroup],
                                        #None, True, outputdir,
                                        #'dim%02d%s' % (dim, fGroup), verbose)

            #print "ECDF relative target graphs done."

            for dim in set(dictDim0.keys()) | set(dictDim1.keys()):
                pprldistr.fmax = None #Resetting the max final value
                pprldistr.evalfmax = None #Resetting the max #fevalsfactor
                if dim in rldDimsOfInterest:
                    try:
                        pprldistr.comp(dictDim0[dim], dictDim1[dim],
                                       rldValsOfInterest, True,
                                       outputdir, '%02dD_all' % dim, verbose)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.'
                                      % (dim))
                        continue

                    dictFG0 = dictDim0[dim].dictByFuncGroup()
                    dictFG1 = dictDim1[dim].dictByFuncGroup()

                    for fGroup in set(dictFG0.keys()) | set(dictFG1.keys()):
                        pprldistr.comp(dictFG0[fGroup], dictFG1[fGroup],
                                       rldValsOfInterest, True, outputdir,
                                       '%02dD_%s' % (dim, fGroup), verbose)

                    dictFN0 = dictDim0[dim].dictByNoise()
                    dictFN1 = dictDim1[dim].dictByNoise()
                    for fGroup in set(dictFN0.keys()) | set(dictFN1.keys()):
                        pprldistr.comp(dictFN0[fGroup], dictFN1[fGroup],
                                       rldValsOfInterest, True, outputdir,
                                       '%02dD_%s' % (dim, fGroup), verbose)

            print "ECDF dashed-solid graphs done."

        if istable:
            dictNG0 = dsList0.dictByNoise()
            dictNG1 = dsList1.dictByNoise()

            for nGroup in set(dictNG0.keys()) & set(dictNG1.keys()):
                pptable2.main(dictNG0[nGroup], dictNG1[nGroup],
                              tabDimsOfInterest, outputdir,
                              '%s' % (nGroup), verbose)
            print "Tables done."

        if isscatter:
            ppscatter.main(dsList0, dsList1, outputdir, verbose=verbose)
            print "Scatter plots done."

        if isfigure or isrldistr or istable or isscatter:
            print "Output data written to folder %s." % outputdir

        plt.rcdefaults()

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "For help use -h or --help"
        return 2

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding
    sys.exit(main())

