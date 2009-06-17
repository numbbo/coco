#!/usr/bin/env python

"""Mini run to display the performance profiles of algorithms.

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
import matplotlib
matplotlib.use("Agg") # To avoid window popup and use without X forwarding
import matplotlib.pyplot as plt

# Add the path to bbob_pproc
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc import ppperfprof, pprldistr
from bbob_pproc import dataoutput, determineFtarget
from bbob_pproc.pproc2 import DataSetList

figformat = ('png', )
# GLOBAL VARIABLES used in the routines defining desired output  for BBOB 2009.
instancesOfInterest = {1:3, 2:3, 3:3, 4:3, 5:3}
instancesOfInterestDet = {1:1, 2:1, 3:1, 4:1, 5:1}

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#FUNCTION DEFINITIONS

def usage():
    print main.__doc__


def main(argv=None):
    """Process info files and output performance profiles."""

    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, "hvpo:",
                                       ["help", "output-dir=", "noisy",
                                        "noise-free", "write-pickles",
                                        "perfprof-only", "targets=",
                                        "verbose"])
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
        isEff = True
        isERT = True
        isECDF = True
        #isPer = False
        #isEff = False
        #isERT = False
        #isECDF = False

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
            elif o == "--perfprof-only":
                isEff = False
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
            sortedAlgs.append(dataoutput.algLongInfos[tmpalg])

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
                #set_trace()

        # Get the target function values depending on the function
        # target = dict(...)
        #target = {1: 1e-8, 2: 1e-8, 3: 1e-8, 4: 1e-8, 5: 1e-8, 6: 1e-8, 7: 1e-8, 8: 1e-8, 9: 1e-8,
                  #10: 1e-8, 11: 1e-8, 12: 1e-8, 13: 1e-8, 14: 1e-8, 15: 1e-8, 16: 1e-8, 17: 1e-8,
                  #18: 1e-8, 19: 1e-8, 20: 1e-8, 21: 1e-8, 22: 1e-8, 23: 1e-8, 24: 1e-8}

        allmintarget = {}
        allmedtarget = {}
        if targets:
            f = open(targetsfile, 'r')
            algSet = pickle.load(f)
            if not set(dsList.dictByAlg().keys()).issubset(algSet):
                raise Usage('Problem here')
            allmintarget = pickle.load(f)
            allmedtarget = pickle.load(f)
            f.close()

        if not allmintarget or not allmedtarget:
            dictDim = dsList.dictByDim()
            for d, dimentries in dictDim.iteritems():
                dictFunc = dimentries.dictByFunc()
                for f, funentries in dictFunc.iteritems():
                    if 11 < 3:  # constant targets
                        for t in (10, 0.1, 1e-3, 1e-5, 1e-8):
                            tmp = allmintarget.setdefault(t, {})
                            tmp.setdefault((f, d), t)
                            tmp = allmedtarget.setdefault(t, {})
                            tmp.setdefault((f, d), t)
                    else:
                        tmptarget = determineFtarget.FunTarget(funentries, d)
                        for i in range(len(tmptarget.ert)):
                            tmp = allmintarget.setdefault(tmptarget.ert[i], {})
                            if (tmptarget.minFtarget[i] < 1e-8): # BBOB-dependent
                                tmptarget.minFtarget[i] = 1e-8
                            tmp.setdefault((f, d), tmptarget.minFtarget[i])

                            tmp = allmedtarget.setdefault(tmptarget.ert[i], {})
                            if (tmptarget.medianFtarget[i] < 1e-8): # BBOB-dependent
                                tmptarget.medianFtarget[i] = 1e-8
                            tmp.setdefault((f, d), tmptarget.medianFtarget[i])
            f = open(targetsfile, 'w')
            pickle.dump(set(dsList.dictByAlg().keys()), f)
            pickle.dump(allmintarget, f)
            pickle.dump(allmedtarget, f)
            f.close()

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
            if verbose:
                print 'Folder %s was created.' % (outputdir)

        # Performance profiles
        if isPer:
            dictDim = dsList.dictByDim()
            for d, entries in dictDim.iteritems():
                for k, t in allmintarget.iteritems():
                    if k >= 10000:
                        continue
                    target = {k: t}
                    ppperfprof.main2(entries, target=target, order=sortedAlgs,
                                    plotArgs=dataoutput.algPlotInfos,
                                    outputdir=outputdir,
                                    info=('%02d_ert%2.1eD' % (d, k)),
                                    verbose=verbose)

                ppperfprof.main2(entries, target=allmintarget,
                                 order=sortedAlgs,
                                 plotArgs=dataoutput.algPlotInfos,
                                 outputdir=outputdir,
                                 info=('%02d' % (d)),
                                 verbose=verbose)
                tmpmintarget = dict((k, t) for k, t in allmintarget.iteritems() if k >= 10000)
                ppperfprof.main2(entries, target=tmpmintarget,
                                 order=sortedAlgs,
                                 plotArgs=dataoutput.algPlotInfos,
                                 outputdir=outputdir,
                                 info=('%02d_ert1.0e+04Dandmore' % (d)),
                                 verbose=verbose)


        if isERT or isEff or isECDF:
            #plt.rc("axes", labelsize=20, titlesize=24)
            #plt.rc("xtick", labelsize=20)
            #plt.rc("ytick", labelsize=20)
            #plt.rc("font", size=20)
            #plt.rc("legend", fontsize=20)

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
                                     **dataoutput.algPlotInfos[elem])
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
                                                       plotArgs=dataoutput.algPlotInfos[elem],
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
                                                       plotArgs=dataoutput.algPlotInfos[elem],
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
                                plt.savefig(figname+"."+i, dpi=300, format="png")
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
                                                       plotArgs=dataoutput.algPlotInfos[elem],
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
