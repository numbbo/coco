#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process data and generates some comparison results.

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
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import dataoutput, pproc
from bbob_pproc.dataoutput import algPlotInfos
from bbob_pproc.pproc import DataSetList, processInputArgs
from bbob_pproc.bbob2010.compall import organizeRTDpictures
from bbob_pproc.bbob2010.compall import ppperfprof, pptables, ppfigs
    
import matplotlib.pyplot as plt

# GLOBAL VARIABLES used in the routines defining desired output for BBOB.
single_target_function_values = (1e1, 1e0, 1e-1, 1e-2, 1e-4, 1e-6, 1e-8)  # one figure for each
summarized_target_function_values = (1e0, 1e-1, 1e-3, 1e-5, 1e-7)   # all in one figure
summarized_target_function_values = (10, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8) 
summarized_target_function_values = tuple(1.001*10**numpy.r_[-8:2:0.2]) # 50 values in [1e-8, 1e2[    
#summarized_target_function_values = (100, 10, 1e0, 1e-1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)  # 10 values, 1e-3 inbetween is missing 
#summarized_target_function_values = tuple(10**numpy.r_[-7:-1:0.2]) # 1e2 and 1e-8  
#summarized_target_function_values = tuple(10**numpy.r_[-1:2:0.2]) # easy easy 
# summarized_target_function_values = (10, 1e0, 1e-1)   # all in one figure
tableconstant_target_function_values = [1e3, 1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-7]
instancesOfInterest = {1:3, 2:3, 3:3, 4:3, 5:3}

# Deterministic instance of interest: only one trial is required.
instancesOfInterestDet = {1:1, 2:1, 3:1, 4:1, 5:1}
instancesOfInterest2010 = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,
                           11:1, 12:1, 13:1, 14:1, 15:1}
#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#FUNCTION DEFINITIONS

def detertbest(dsList, minFtarget):
    """Determines the best ert for a given target function value."""
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
    """Creates the data structure of the target function values."""

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

#    TODO: Next step: archive files as input arguments
#    (iii) tar archive files (eventually compressed
#    using gzip or bzip2). Each of these white-space separated arguments should
#    correspond to the data of one algorithm.
def main(argv=None):
    """Main routine for post-processing the data of multiple algorithms.

    Keyword arguments:
    argv -- list of strings containing options and arguments. If not provided,
    sys.argv is accessed.

    argv must list folders containing BBOB data files. Each of these folders
    should correspond to the data of ONE algorithm and should be listed in
    algorithmshortinfos.txt, a file from the bbob_pproc.compall package listing
    the information of various algorithms treated using bbob_pproc.dataoutput

    Furthermore, argv can begin with, in any order, facultative option flags
    listed below.

        -h, --help

            display this message

        -v, --verbose

            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

        -o, --output-dir OUTPUTDIR

            change the default output directory ('cmpalldata') to
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

        $ python -m bbob_pproc.runcompall -h

    This will print out this help message.

    * From the python interactive shell (requires that the path to this
      package is in python search path):

        >>> from bbob_pproc import runcompall
        >>> runcompall.main('-o outputfolder folder1 folder2'.split())

    This will execute the post-processing on the data found in folder1
    and folder2.
    The -o option changes the output folder from the default cmpalldata to
    outputfolder.

    * Generate post-processing data for some algorithms:

        $ python runcompall.py AMALGAM BFGS BIPOP-CMA-ES

    """

    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, "hvo:",
                                       ["help", "output-dir=", "noisy",
                                        "noise-free", "perfprof-only",
                                        "tab-only", "fig-only", "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        verbose = False
        outputdir = 'cmpalldata'
        isNoisy = False
        isNoiseFree = False

        isPer = True
        isTab = True
        isFig = True

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
                isFig = False
            elif o == "--perfprof-only":
                isTab = False
                isFig = False
            elif o == "--fig-only":
                isTab = False
                isPer = False
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
            elif isNoiseFree and not isNoisy:
                dictAlg[i] = dictAlg[i].dictByNoise().get('noiselessall', DataSetList())

            tmp = set((j.algId, j.comment) for j in dictAlg[i])
            for j in tmp:
                if not dataoutput.isListed(j):
                    dataoutput.updateAlgorithmInfo(j, verbose=verbose)

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

        # group targets:
        dictTarget = {}
        for t in sorted(set(single_target_function_values + summarized_target_function_values)):
            tmpdict = dict.fromkeys(((f, d) for f in range(0, 25) + range(101, 131) for d in (2, 3, 5, 10, 20, 40)), t)
            stmp = 'E'
            if t == 1:
                stmp = 'E-'
            # dictTarget['_f' + stmp + '%2.1f' % numpy.log10(t)] = (tmpdict, )
            if t in single_target_function_values: 
                dictTarget['_f' + stmp + '%02d' % numpy.log10(t)] = (tmpdict, )
            if t in summarized_target_function_values: 
                dictTarget.setdefault('_allfs', []).append(tmpdict)

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
            if verbose:
                print 'Folder %s was created.' % (outputdir)

        plt.rc("axes", labelsize='medium', titlesize='large')
        plt.rc("xtick", labelsize='medium')
        plt.rc("ytick", labelsize='medium')
        plt.rc("font", size=12.0)
        plt.rc("legend", fontsize='large')

        # Performance profiles
        if isPer:
            dictNoi = pproc.dictAlgByNoi(dictAlg)
            for ng, tmpdictAlg in dictNoi.iteritems():
                dictDim = pproc.dictAlgByDim(tmpdictAlg)
                for d, entries in dictDim.iteritems():
                    for k, t in dictTarget.iteritems():
                        #set_trace()
                        ppperfprof.main(entries, target=t, order=sortedAlgs,
                                        plotArgs=algPlotInfos,
                                        outputdir=outputdir,
                                        info=('%02d%s_%s' % (d, k, ng)),
                                        verbose=verbose)

            #dictFun = pproc.dictAlgByFun(dictAlg)
            #for fun, tmpdictAlg in dictFun.iteritems():
                #dictDim = pproc.dictAlgByDim(tmpdictAlg)
                #for d, entries in dictDim.iteritems():
                    #for k, t in dictTarget.iteritems():
                        ##set_trace()
                        #ppperfprof.main(entries, target=t, order=sortedAlgs,
                                        #plotArgs=algPlotInfos,
                                        #outputdir=outputdir,
                                        #info=('%03d_%02d%s' % (fun, d, k)),
                                        #verbose=verbose)

            organizeRTDpictures.do(outputdir)
            print "ECDFs of ERT figures done."

        if isTab:
            allmintarget, allertbest = detTarget(dsList)
            # load targets
            #f=open('2009best.pickle')
            #allmintarget = pickle.load(f)
            #allertbest = pickle.load(f)
            #f.close()
            # save targets, etc.
            #f=open('2009best.pickle', 'w')
            #pickle.dump(allmintarget, f)
            #pickle.dump(allertbest, f)
            #f.close()
            #set_trace()
            pptables.tablemanyalgonefunc(dictAlg, allmintarget, allertbest,
                                         sortedAlgs, outputdir, verbose)
            print "Comparison tables done."

        plt.rcdefaults()

        if isFig:
            plt.rc("axes", labelsize=20, titlesize=24)
            plt.rc("xtick", labelsize=20)
            plt.rc("ytick", labelsize=20)
            plt.rc("font", size=20)
            plt.rc("legend", fontsize=20)
            ppfigs.main(dictAlg, sortedAlgs, 1e-8, outputdir, verbose)
            plt.rcdefaults()
            print "Scaling figures done."

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())

