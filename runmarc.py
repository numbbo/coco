#!/usr/bin/env python

"""Mini run to display the performance profiles of algorithms.

"""

from __future__ import absolute_import

import os
import sys
import glob
import getopt
from pdb import set_trace
#import matplotlib
#matplotlib.use("Agg") # To avoid window popup and use without X forwarding
import matplotlib.pyplot as plt

# Add the path to bbob_pproc
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import ppperfprof, pprldistr
from bbob_pproc import dataoutput
from bbob_pproc.pproc import DataSetList

figformat = ('png', )

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
                                       ["help", "output-dir",
                                        "write-pickles", "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        verbose = False
        outputdir = 'defaultoutputdirectory'
        isWritePickle = False

        isEff = True
        isERT = True
        isPer = True
        isECDF = True
        #isEff = False
        #isERT = False
        #isPer = False
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
            else:
                assert False, "unhandled option"

        # Write the pickle files if needed!
        if isWritePickle:
            dsList = DataSetList(args)
            dataoutput.outputPickle(dsList, verbose=verbose)
            sys.exit()
        else:
            #Get only pickles!
            tmpargs = []
            sortedAlgs = []
            for i in args:
                if i.endswith(".pickle"):
                    tmpargs.append(i)
                    tmpalg = os.path.split(os.path.split(i)[0])[1]
                else:
                    tmpargs.extend(glob.glob(os.path.join(i, "*.pickle")))
                    tmpalg = os.path.split(i)[1]

                if not tmpalg in sortedAlgs:
                    sortedAlgs.append(dataoutput.algLongInfos[tmpalg])

            if not sortedAlgs:
                sortedAlgs = list(dataoutput.algLongInfos[i] for i in args)

            #set_trace()
            dsList = DataSetList(tmpargs)

            if not dsList:
                sys.exit()


        # Get the target function values depending on the function
        # target = dict(...)
        target = {1: 1e-8, 2: 1e-8, 3: 1e-8, 4: 1e-8, 5: 1e-8, 6: 1e-8, 7: 1e-8, 8: 1e-8, 9: 1e-8,
                  10: 1e-8, 11: 1e-8, 12: 1e-8, 13: 1e-8, 14: 1e-8, 15: 1e-8, 16: 1e-8, 17: 1e-8,
                  18: 1e-8, 19: 1e-8, 20: 1e-8, 21: 1e-8, 22: 1e-8, 23: 1e-8, 24: 1e-8}

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
            if verbose:
                print 'Folder %s was created.' % (outputdir)

        # Sort algorithms...
        # dsList.dictByAlg()

        # Performance profiles
        if isPer:
            dictDim = dsList.dictByDim()
            for d, entries in dictDim.iteritems():
                ppperfprof.main(entries, target=target,
                                order=sortedAlgs, plotArgs=dataoutput.algPlotInfos,
                                outputdir=outputdir,
                                info=('%02d' % d), verbose=verbose)

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
                for f, funentries in dictFunc.iteritems():
                    dictAlg = funentries.dictByAlg()
                    if isEff:
                        plt.figure()
                        for alg in sortedAlgs:
                            #set_trace()
                            pprldistr.plotERTDistr(dictAlg[alg],
                                                   target,
                                                   plotArgs=dataoutput.algPlotInfos[alg],
                                                   verbose=True)
                        #try log x-axis if possible. and labels !
                        plt.xscale("log")
                        plt.legend(loc="best")
                        plt.xlabel("Expected Running Time")
                        #plt.ylabel("Proportion Bootstrap")
                        plt.grid(True)
                        figname = os.path.join(outputdir, "ppertdistr_f%03d_%02d" %(f, d))
                        for i in figformat:
                            plt.savefig(figname+"."+i, dpi=300, format=i)
                            if verbose:
                                print "Saved figure %s.%s" % (figname, i)
                        plt.close()

                    # Plot the VTR vs ERT...
                    if isERT:
                        plt.figure()
                        for alg in sortedAlgs:
                            #set_trace()
                            entries = dictAlg[alg]
                            plt.plot(entries[0].target[entries[0].target>=target[f]],
                                     entries[0].ert[entries[0].target>=target[f]],
                                     **dataoutput.algPlotInfos[alg])
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
                        figname = os.path.join(outputdir, "ppfig_f%03d_%02d" %(f, d))
                        for i in figformat:
                            plt.savefig(figname+"."+i, dpi=300, format=i)
                            if verbose:
                                print "Saved figure %s.%s" % (figname, i)
                        plt.close()

                    if isECDF:
                        plt.figure()
                        maxEvalsF = 0
                        for alg in sortedAlgs:
                            entries = dictAlg[alg]
                            maxEvalsF = max((maxEvalsF, max(entries[0].maxevals/entries[0].dim)))

                        for alg in sortedAlgs:
                            #set_trace()
                            entries = dictAlg[alg]
                            pprldistr.plotRLDistr2(entries, fvalueToReach=target,
                                                   maxEvalsF=maxEvalsF,
                                                   plotArgs=dataoutput.algPlotInfos[alg],
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
                        figname = os.path.join(outputdir, "pprldistr_f%03d_%02d" %(f, d))
                        for i in figformat:
                            plt.savefig(figname+"."+i, dpi=300, format="png")
                            if verbose:
                                print "Saved figure %s.%s" % (figname, i)
                        plt.close()

                        plt.figure()
                        for alg in sortedAlgs:
                            #set_trace()
                            entries = dictAlg[alg]
                            pprldistr.plotFVDistr2(entries, fvalueToReach=target,
                                                   maxEvalsF=max(entries[0].maxevals/entries[0].dim),
                                                   plotArgs=dataoutput.algPlotInfos[alg],
                                                   verbose=verbose)
                            #set_trace()

                        #try log x-axis if possible. and labels !
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
                        figname = os.path.join(outputdir, "ppfvdistr_f%03d_%02d" %(f, d))
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
