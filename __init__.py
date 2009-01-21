#!/usr/bin/env python
# coding: utf-8

"""Black-Box Optimization Benchmarking (BBOB) post processing tool:

The BBOB post-processing tool takes as input data from BBOB experiments and
generates output that will be used in the generation of the LateX-formatted
article summarizing the experiments.

"""

#credits to G. Van Rossum: http://www.artima.com/weblogs/viewpost.jsp?thread=4829

from __future__ import absolute_import

import sys
import os
import getopt
import pickle

#import scipy
#import matplotlib.pyplot as plt

from pdb import set_trace
from bbob_pproc import readindexfiles, findindexfiles
from bbob_pproc import pproc, ppfig, pptex, pprldistr, ppfigdim

__all__  = ['readindexfiles', 'findindexfiles', 'ppfig', 'pptex', 'pprldistr',
            'main', 'ppfigdim', 'pproc']

#colors = {2:'b', 3:'g', 5:'r', 10:'c', 20:'m', 40:'y'} #TODO colormaps!
tabDimsOfInterest = [5, 20]    # dimension which are displayed in the tables
# tabValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
tabValsOfInterest = (10, 1.0, 1e-1, 1.0e-3, 1.0e-5, 1.0e-8)

figValsOfInterest = (10, 1e-1, 1e-4, 1e-8)  # should 

rldDimsOfInterest = (5, 20)
#rldValsOfInterest = (1e-8, 1e-5, 1e-2, 10)
rldValsOfInterest = (10, 1e-2, 1e-5, 1e-8)
#Put backward to have the legend in the same order as the lines.

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


###############################################################################


#FUNCTION DEFINITIONS

def createIndexEntries(args, outputdir, isPickled, verbose=True):
    """Returns a list of post-processed ÏndexEntries from a list of inputs.
    Keyword arguments:
    args -- list of strings being either info file names, folder containing 
            info files or pickled data files.
    verbose
    Outputs:
    indexEntries -- list of IndexEntry instances.
    """
    indexFiles = []
    args = set(args) #for unicity
    pickles = [] # list of pickled
    for i in args:
        if i.endswith('.info'):
            (filepath,filename) = os.path.split(i)
            indexFiles.append(i)
        elif os.path.isdir(i):
            indexFiles.extend(findindexfiles.main(i,verbose))
        elif i.endswith('.pickle'):
            pickles.append(i)
        else:
            raise Usage('Expect as input argument either info files or '+
                        'a folder containing info files.')
            #TODO: how do we deal with this?
    indexFiles = set(indexFiles) #for unicity

    indexEntries = readindexfiles.IndexEntries(verbose=verbose)
    for i in pickles:
        try:
            f = open(i,'r')
            entry = pickle.load(f)
            f.close()
            if verbose:
                print 'Unpickled %s.' % (i)
            indexEntries.append(entry)
        except IOError, (errno, strerror):
            print "I/O error(%s): %s" % (errno, strerror)
        except UnpicklingError:
            f.close()
            print '%s could not be unpickled.' %(i)
            if not i.endswith('.pickle'):
                print '%s might not be a pickle data file.' %(i)

    if indexFiles:
        #If indexFiles is not empty, some new indexEntries are added and
        #then some old pickled indexEntries may need to be updated.
        indexEntries.extend(readindexfiles.IndexEntries(indexFiles, verbose))

        if isPickled:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if verbose:
                    print '%s was created.' % (outputdir)

            for i in indexEntries:
                filename = os.path.join(outputdir, 'ppdata_f%d_%d'
                                                    %(i.funcId, i.dim))
                try:
                    f = open(filename + '.pickle','w')
                    pickle.dump(i, f)
                    f.close()
                    if verbose:
                        print 'Saved pickle in %s.' %(filename+'.pickle')
                except IOError, (errno, strerror):
                    print "I/O error(%s): %s" % (errno, strerror)
                except PicklingError:
                    print "Could not pickle %s" %(i)

    return indexEntries


def usage():
    print main.__doc__


def main(argv=None):
    """Generates from BBOB experiment data some outputs for a tex document.

    If provided with some index entries (from info files), this should return
    many output files in the folder 'ppdata' needed for the compilation of
    latex document ExampleDataPresentation.tex. These output files will contain
    performance tables, performance scaling figures and empirical cumulative
    distribution functions figures.

    Keyword arguments:
    argv -- list of strings containing options and arguments.

    argv should list either names of info files or folders containing info
    files. argv can also list post-processed pickle files generated from this
    method. Furthermore, argv can begin with, in any order, facultative option
    flags listed below.

        -h, --help

            display this message

        -v, --verbose

            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

        -n, --no-pickle

            prevents pickled post processed data files from being generated.

        -o, --output-dir OUTPUTDIR

            change the default output directory ('ppdata') to OUTPUTDIR

        --tab-only, --fig-only, --rld-only

            these options can be used to output respectively the tex tables,
            convergence and ENFEs graphs figures, run length distribution
            figures only. A combination of any two of these options results in
            no output.

    Exceptions raised:
    UsageError --
    """

    if argv is None:
        argv = sys.argv
    try:

        try:
            opts, args = getopt.getopt(argv[1:], "hvno:",
                                       ["help", "output-dir",
                                        "tab-only", "fig-only", "rld-only",
                                        "no-pickle","verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        isfigure = True
        istab = True
        isrldistr = True
        isPostProcessed = False
        isPickled = True
        verbose = True
        outputdir = 'ppdata'

        #Process options
        for o, a in opts:
            if o in ("-v","--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-n", "--no-pickle"):
                isPickled = False
            elif o in ("-o", "--output-dir"):
                outputdir = a
            #The next 3 are for testing purpose
            elif o == "--tab-only":
                isfigure = False
                isrldistr = False
            elif o == "--fig-only":
                istab = False
                isrldistr = False
            elif o == "--rld-only":
                istab = False
                isfigure = False
            else:
                assert False, "unhandled option"

        indexEntries = createIndexEntries(args, outputdir, isPickled, verbose)
        sortedByDim = indexEntries.sortByDim()
        sortedByFunc = indexEntries.sortByFunc()

        if isfigure or istab or isrldistr:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if verbose:
                    print '%s was created.' % (outputdir)

        if isfigure:
            ppfigdim.main(indexEntries, figValsOfInterest, outputdir,
                          verbose)

        if istab:
            for fun, sliceFun in sortedByFunc.items():
                tmp = []
                for i in sliceFun:
                    if i.dim in tabDimsOfInterest:
                        tmp.append(i)
                if tmp:
                    pptex.main(tmp, tabValsOfInterest, outputdir, 'f%d' % fun,
                               verbose)

        if isrldistr:
            for dim, sliceDim in sortedByDim.items():
                if dim in rldDimsOfInterest:
                    pprldistr.main(sliceDim, rldValsOfInterest, 
                                   'dim%02dall' % dim, outputdir, verbose)
                    sortedByFG = sliceDim.sortByFuncGroup()
                    #set_trace()
                    for funcGroup, sliceFuncGroup in sortedByFG.items():
                        pprldistr.main(sliceFuncGroup, rldValsOfInterest,
                                       'dim%02d%s' % (dim, funcGroup),
                                       outputdir, verbose)

        #if verbose:
            #print 'total ps = %g\n' % (float(scipy.sum(ps))/scipy.sum(nbRuns))

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2


if __name__ == "__main__":
   sys.exit(main()) #TODO change this to deal with args?
