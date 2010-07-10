#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Deprecated
   Process data and generates some comparison results from either
   raw data files or pickle data files.
   Synopsis:
      python path_to_folder/bbob_pproc/minirun.py [OPTIONS] FILE_NAME FOLDER_NAME...
    Help:
      python path_to_folder/bbob_pproc/minirun.py -h

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

from bbob_pproc.compall import ppperfprof, pptables, determineFtarget2
from bbob_pproc.compall import organizeRTDpictures
from bbob_pproc import  pprldistr, dataoutput
from bbob_pproc.dataoutput import algLongInfos, algShortInfos, algPlotInfos
from bbob_pproc.pproc import DataSetList
import matplotlib.pyplot as plt

# GLOBAL VARIABLES used in the routines defining desired output for BBOB.
constant_target_function_values = (1e1, 1e0, 1e-1, 1e-2, 1e-4, 1e-6)
# constant_target_function_values = (1e-6, )  # single 
# constant_target_function_values = (1e1, 1e-0, 1e-1)  # light&easy 
# constant_target_function_values = (1e-3, 1e-5, 1e-7) # tight&heavy
# constant_target_function_values = 10**numpy.array(list(numpy.r_[2:0:-0.5]) + range(0,-9,-1))  # movie
# constant_target_function_values = 10**(numpy.array(list(numpy.r_[0:-7:-0.1]) + list(numpy.r_[-7:-8:-0.05])))  # dense 
# constant_target_function_values = 10**numpy.array([1, 0] + sorted(2*(-1,-2,-3,-4,-5,-6,-7,-7.98,-7.99), reverse=True))  # try for multimodal single functions
# constant_target_function_values = 10**numpy.array((0,-1,-2,-3,-4,-5,-6,-7,-7.98,-7.99))  # single function

instancesOfInterest = {1:3, 2:3, 3:3, 4:3, 5:3}
instancesOfInterestDet = {1:1, 2:1, 3:1, 4:1, 5:1}
figformat = ('png', 'pdf')

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#FUNCTION DEFINITIONS

def detTarget(dsList):
    allmintarget = {}
    allmedtarget = {}
    allertbest = {}
    dictDim = dsList.dictByDim()
    for d, dimentries in dictDim.iteritems():
        dictFunc = dimentries.dictByFunc()
        for f, funentries in dictFunc.iteritems():
            #tmp = allmintarget.setdefault(1, {})
            #tmp.setdefault((f, d), 1)
            tmptarget = determineFtarget2.FunTarget(funentries, d)
            #, use_uniform_fake_values=True) # for the tables to show uniform values
            for i in range(len(tmptarget.ert)):
               tmp = allmintarget.setdefault(tmptarget.ert[i], {})
               if (tmptarget.minFtarget[i] < 1e-8): # BBOB-dependent
                   if i == 0 or tmptarget.minFtarget[i-1] > 1e-8:
                       tmptarget.minFtarget[i] = 1e-8
                   else:
                       tmptarget.minFtarget[i] = numpy.NaN
               tmp.setdefault((f, d), tmptarget.minFtarget[i])

               tmp = allmedtarget.setdefault(tmptarget.ert[i], {})
               if (tmptarget.medianFtarget[i] < 1e-8): # BBOB-dependent
                   tmptarget.medianFtarget[i] = 1e-8
               tmp.setdefault((f, d), tmptarget.medianFtarget[i])

               tmp = allertbest.setdefault(tmptarget.ert[i], {})
               tmp.setdefault((f, d), tmptarget.ertbest[i])

    return allmintarget, allmedtarget, allertbest

def usage():
    print main.__doc__


def main(argv=None):
    """
    Keyword arguments:
    argv -- list of strings containing options and arguments. If not provided,
    sys.argv is accessed.

    argv should list either names of info files or folders containing info
    files or folders containing pickle files (preferred).
    Furthermore, argv can begin with, in any order, facultative option
    flags listed below.

        -h, --help

            display this message

        -v, --verbose
 
            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

        -p, --write-pickles

            generates pickle post processed data files and then stops!

        -o, --output-dir OUTPUTDIR

            change the default output directory ('defaultoutputdirectory') to OUTPUTDIR

        --noise-free, --noisy

            restrain the post-processing to part of the data set only. Actually fasten the
            post-processing since it loads only part of the pickle files.

        --targets TARGETFILE

            uses TARGETFILE instead of the targets defined by the data given as
            input arguments.

        --tab-only, --perfprof-only

            these options can be used to output respectively the comparison
            tex tables or the performance profiles only.
            A combination of any two of these options results in
            no output.

    Exceptions raised:
    Usage -- Gives back a usage message.

    Examples:

    * Calling the minirun.py interface from the command line:

        $ python bbob_pproc/minirun.py -v

        $ python bbob_pproc/minirun.py -o otherppdata experiment2/*.info


    * Loading this package and calling the main from the command line
      (requires that the path to this package is in python search path):

        $ python -m bbob_pproc -h


    This will print out this help message.

    * From the python interactive shell (requires that the path to this
      package is in python search path):

        >>> import bbob_pproc
        >>> bbob_pproc.minirun.main('-o outputfolder folder1'.split())

    This will execute the post-processing on the index files found in folder1.
    The -o option changes the output folder from the default ppdata to
    outputfolder.


    * Generate post-processed pickle data files:

        $ python minirun.py -p RAWDATAFOLDER

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

        $ python minirun.py AMALGAM BFGS CMA-ES

    * Generate post-processing data using a custom target pickle file:

        $ python minirun.py --targets customtargetfile.pickle OTHER_ALGORITHM

    Using the --targets option, the custom target file is not overwritten and
    the default target file is not generated.

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
        targetsfile = 'targetsfile.pickle'

        isPer = True
        isTab = False #True
        isTab2 = False #True
        isTab3 = True
        isEff = True
        isERT = True
        isECDF = True

        #Process options
        for o, a in opts:
            if o in ("-v","--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-o", "--output-dir"):
                outputdir = a
            elif o in ("-p", "--write-pickles"):
                isWritePickle = True
            elif o == "--targets":
                targets = True
                targetsfile = a
            elif o == "--noisy":
                isNoisy = True
            elif o == "--noise-free":
                isNoiseFree = True
            elif o == "--tab-only":
                isPer = False
                isEff = False
                isERT = False
                isECDF = False
            elif o == "--perfprof-only":
                isEff = False
                isTab = False
                isTab2 = False
                isTab3 = False
                isERT = False
                isECDF = False
            else:
                assert False, "unhandled option"

        # Write the pickle files if needed!
        if isWritePickle:
            dsList = DataSetList(args)
            dataoutput.outputPickle(dsList, verbose=verbose)
            sys.exit()

        #Get only pickles!
        tmpargs = []
        sortedAlgs = []
        for i in args:
            if i.endswith(".pickle"):
                tmpargs.append(i)
                tmpalg = os.path.split(os.path.split(i)[0])[1]
                if algLongInfos[tmpalg] in sortedAlgs:
                    continue
            else:
                if isNoisy and isNoiseFree:
                    ext = "*.pickle"
                elif isNoisy:
                    ext = "*f1*.pickle"
                elif isNoiseFree:
                    ext = "*f0*.pickle"
                else:
                    ext = "*.pickle"
                tmpargs.extend(glob.glob(os.path.join(i, ext)))
                tmpalg = os.path.split(i)[1]
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

        allmintarget = {}
        allmedtarget = {}
        allertbest = {}
        if targets:
            f = open(targetsfile, 'r')
            algSet = pickle.load(f)
            if not set(dsList.dictByAlg().keys()).issubset(algSet):
                #set_trace()
                raise Usage('some algorithm are not regarded in the used targets (if this is deliberate, uncomment the raise error line and rerun)')
            allmintarget = pickle.load(f)
            allmedtarget = pickle.load(f)
            allertbest = pickle.load(f)
            f.close()

        if not allmintarget or not allmedtarget or not allertbest:
            allmintarget, allmedtarget, allertbest = detTarget(dsList)
            f = open(targetsfile, 'w')
            pickle.dump(set(dsList.dictByAlg().keys()), f)
            pickle.dump(allmintarget, f)
            pickle.dump(allmedtarget, f)
            pickle.dump(allertbest, f)
            f.close()

        # Restrain the allmintarget to the functions considered
        if isNoisy:
            for t in allmintarget:
                allmintarget[t] = dict((i, allmintarget[t][i]) for i in allmintarget[t] if i[0] > 100)
                allmedtarget[t] = dict((i, allmedtarget[t][i]) for i in allmedtarget[t] if i[0] > 100)
                allertbest[t] = dict((i, allertbest[t][i]) for i in allertbest[t] if i[0] > 100)
        elif isNoiseFree:
            for t in allmintarget:
                allmintarget[t] = dict((i, allmintarget[t][i]) for i in allmintarget[t] if i[0] <= 100)
                allmedtarget[t] = dict((i, allmedtarget[t][i]) for i in allmedtarget[t] if i[0] <= 100)
                allertbest[t] = dict((i, allertbest[t][i]) for i in allertbest[t] if i[0] <= 100)

        # group targets:
        dictTarget = {}
        for i in sorted(allmintarget):
            if i < 10000:
                # dictTarget['_ertE%2.1fD' % numpy.log10(i)] = (allmintarget[i],)
                dictTarget['_ertE%02dD' % numpy.log10(i)] = (allmintarget[i],)
            if i >= 10000:
                # dictTarget.setdefault('_ertE4.0Dandmore', []).append(allmintarget[i])
                dictTarget.setdefault('_ertE04Dandmore', []).append(allmintarget[i])
            dictTarget.setdefault('_allerts', []).append(allmintarget[i])
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
                    #if k == '_fE1.0':
                    #    set_trace()
                    ppperfprof.main(entries, target=t, order=sortedAlgs,
                                    plotArgs=algPlotInfos,
                                    outputdir=outputdir,
                                    info=('%02d%s' % (d, k)),
                                    fileFormat=figformat,
                                    verbose=verbose)
            #try:
            organizeRTDpictures.do(outputdir)
            #os.system('./organizeRTDpictures.py ' + outputdir)

        if isTab:
            pptables.tableonealg(dsList, allmintarget, allertbest, sortedAlgs,
                                 outputdir)

        if isTab2:
            pptables.tablemanyalg(dsList, allmintarget, allertbest, sortedAlgs,
                                  outputdir)

        if isTab3:
            pptables.tablemanyalgonefunc(dsList, allmintarget, allertbest,
                                         sortedAlgs, outputdir)

        if isERT or isEff or isECDF:
            # ECDF: 1 per function and dimension
            dictDim = dsList.dictByDim()
            for d, dimentries in dictDim.iteritems():
                dictFunc = dimentries.dictByFunc()
                if isERT:
                    for f, funentries in dictFunc.iteritems():
                        dictAlg = funentries.dictByAlg()
                        # Plot the VTR vs ERT...
                        plt.figure()
                        for alg in sortedAlgs:
                            for elem in alg:
                                try:
                                    entry = dictAlg[elem][0]
                                    break
                                except KeyError:
                                    pass

                            plt.plot(entry.target[entry.target>=1e-8],
                                     entry.ert[entry.target>=1e-8],
                                     **algPlotInfos[elem])
                        #try log x-axis if possible. and labels !
                        plt.xscale("log")
                        plt.yscale("log")
                        plt.gca().invert_xaxis()
                        #set_trace()
                        #plt.xlim(plt.xlim()[0], max(plt.xlim()[1], 1e-8))
                        plt.legend(loc="best")
                        plt.xlabel("Target Function Value")
                        plt.ylabel("Expected Running Time")
                        plt.grid(True)
                        figname = os.path.join(outputdir, "ppfig_f%03d_%02d_ert" %(f, d))
                        for i in figformat:
                            plt.savefig(figname+"."+i, dpi=300, format=i)
                            if verbose:
                                print "Saved figure %s.%s" % (figname, i)
                        plt.close()

                for k, t in allmintarget.iteritems():
                    target = dict((f[0], t[f]) for f in t if f[1] == d)
                    if len(target) == 0:
                        continue
                    for f, funentries in dictFunc.iteritems():
                        target.setdefault(f, 0.)

                        dictAlg = funentries.dictByAlg()
                        if isEff:
                            plt.figure()
                            for alg in sortedAlgs:
                                #set_trace()
                                for elem in alg:
                                    try:
                                        entry = dictAlg[elem]
                                        break
                                    except KeyError:
                                        pass
                                pprldistr.plotERTDistr(entry,
                                                       target,
                                                       plotArgs=algPlotInfos[elem],
                                                       verbose=True)
                            #try log x-axis if possible. and labels !
                            plt.xscale("log")
                            plt.legend(loc="best")
                            plt.xlabel("Expected Running Time")
                            #plt.ylabel("Proportion Bootstrap")
                            plt.grid(True)
                            figname = os.path.join(outputdir, "ppertdistr_f%03d_%02d_ert%2.1eD" %(f, d, k))
                            for i in figformat:
                                plt.savefig(figname+"."+i, dpi=300, format=i)
                                if verbose:
                                    print "Saved figure %s.%s" % (figname, i)
                            plt.close()

                        if isECDF:
                            plt.figure()
                            maxEvalsF = 0
                            for alg in sortedAlgs:
                                for elem in alg:
                                    try:
                                        entries = dictAlg[elem]
                                        break
                                    except KeyError:
                                        pass
                                maxEvalsF = max((maxEvalsF, max(entries[0].maxevals/entries[0].dim)))

                            for alg in sortedAlgs:
                                for elem in alg:
                                    try:
                                        entries = dictAlg[elem]
                                        break
                                    except KeyError:
                                        pass
                                pprldistr.plotRLDistr2(entries, fvalueToReach=target,
                                                       maxEvalsF=maxEvalsF,
                                                       plotArgs=algPlotInfos[elem],
                                                       verbose=verbose)

                            #try log x-axis if possible. and labels !
                            plt.xscale("log")
                            #plt.gca().invert_xaxis()
                            #set_trace()
                            #plt.xlim(plt.xlim()[0], max(plt.xlim()[1], 1e-8))
                            plt.xlim(max(1./40, plt.xlim()[0]), maxEvalsF**1.05)
                            plt.ylim(0., 1.)
                            plt.legend(loc="best")
                            plt.xlabel("FEvals/DIM")
                            plt.ylabel("Proportion of trials")
                            plt.grid(True)
                            figname = os.path.join(outputdir, "pprldistr_f%03d_%02d_ert%2.1eD" %(f, d, k))
                            for i in figformat:
                                plt.savefig(figname+"."+i, dpi=300, format=i)
                                if verbose:
                                    print "Saved figure %s.%s" % (figname, i)
                            plt.close()

                            plt.figure()
                            for alg in sortedAlgs:
                                #set_trace()
                                for elem in alg:
                                    try:
                                        entries = dictAlg[elem]
                                        break
                                    except KeyError:
                                        pass
                                pprldistr.plotFVDistr2(entries, fvalueToReach=target,
                                                       maxEvalsF=max(entries[0].maxevals/entries[0].dim),
                                                       plotArgs=algPlotInfos[elem],
                                                       verbose=verbose)
                                #set_trace()

                            #try log x-axis if possible
                            try:
                                plt.xscale("log")
                            except OverflowError:
                                pass
                            #plt.gca().invert_xaxis()
                            #set_trace()
                            plt.xlim(1., max(1., plt.xlim()[1]))
                            plt.legend(loc="best")
                            plt.xlabel("Df/Dftarget")
                            plt.ylabel("Proportion of trials")
                            plt.grid(True)
                            #set_trace()
                            figname = os.path.join(outputdir, "ppfvdistr_f%03d_%02d_ert%2.1eD" %(f, d, k))
                            for i in figformat:
                                plt.savefig(figname+"."+i, dpi=300, format=i)
                                if verbose:
                                    print "Saved figure %s.%s" % (figname, i)
                            plt.close()

            #plt.rcdefaults()

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
   sys.exit(main())
