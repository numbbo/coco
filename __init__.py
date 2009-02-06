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

#import numpy
#import matplotlib.pyplot as plt

from pdb import set_trace
from bbob_pproc import readindexfiles, findindexfiles
from bbob_pproc import pproc, ppfig, pptex, pprldistr, ppfigdim

__all__  = ['readindexfiles', 'findindexfiles', 'pptex', 'pprldistr',
            'main', 'ppfigdim', 'pproc']

#colors = {2:'b', 3:'g', 5:'r', 10:'c', 20:'m', 40:'y'} #TODO colormaps!
tabDimsOfInterest = [5, 20]    # dimension which are displayed in the tables
# tabValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
tabValsOfInterest = (10, 1.0, 1e-1, 1e-3, 1e-5, 1.0e-8)  # 
# tabValsOfInterest = (10, 1.0, 1e-1, 1.0e-4, 1.0e-8)  # 1e-3 1e-5

figValsOfInterest = (10, 1e-1, 1e-4, 1e-8)  # 
figValsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8)  # 

rldDimsOfInterest = (5, 20)
rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8) 
#Put backward to have the legend in the same order as the lines.

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


###############################################################################


#FUNCTION DEFINITIONS

def createIndexEntries(args, verbose=True):
    """Returns an instance of IndexEntries from a list of inputs.
    Keyword arguments:
    args -- list of strings being either info file names, folder containing 
            info files or pickled data files.
    verbose -- controls verbosity.
    Outputs:
    indexEntries -- list of IndexEntry instances.

    Exception:
    Usage --

    """

    #TODO split pickling process and reading.
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
            raise Usage('File or folder ' + i + ' not found. ' +
                        'Expecting as input argument either .info file(s) ' +
                        'or a folder containing .info file(s).')
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

    return indexEntries


def usage():
    print main.__doc__


def main(argv=None):
    """Generates from BBOB experiment data some outputs for a tex document.

    If provided with some index entries (from info files), this should return
    many output files in the folder 'ppdata' needed for the compilation of
    latex document templateBBOBarticle.tex. These output files will contain
    performance tables, performance scaling figures and empirical cumulative
    distribution figures.

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
    Usage --


    Examples:

    * Calling the bbob_pproc.py interface from the command line:

        $ python bbob_pproc.py OPTIONS DATA_TO_PROCESS1 DATA_TO_PROCESS2...

    * Loading this package and calling the main from the command line
      (requires that the path to this package is in python search path):

        $ python -m bbob_pproc main OPTIONS DATA_TO_PROCESS1...

    * From the python interactive shell (requires that the path to this 
      package is in python search path):

        >>> import bbob_pproc
        >>> python bbob_pproc.main(['', 'OPT1', 'OPT2', 'data_to_process_1',
                                    'data_to_process_2', ...])

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

        if isPickled or isfigure or istab or isrldistr:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if verbose:
                    print '%s was created.' % (outputdir)

        indexEntries = createIndexEntries(args, verbose)

        if isPickled:
            #Should get in there only if some data were not pickled.

            for i in indexEntries:
                filename = os.path.join(outputdir, 'ppdata_f%d_%d'
                                                    %(i.funcId, i.dim))
                try:
                    f = open(filename + '.pickle','w')
                    pickle.dump(i, f)
                    f.close()
                    #if verbose:
                        #print 'Saved pickle in %s.' %(filename+'.pickle')
                except IOError, (errno, strerror):
                    print "I/O error(%s): %s" % (errno, strerror)
                except PicklingError:
                    print "Could not pickle %s" %(i)


        if isfigure:
            ppfigdim.main(indexEntries, figValsOfInterest, outputdir,
                          verbose)

        if istab:
            sortedByFunc = indexEntries.sortByFunc()
            for fun, sliceFun in sortedByFunc.items():
                sortedByDim = sliceFun.sortByDim()
                tmp = []
                for dim in tabDimsOfInterest:
                    try:
                        if len(sortedByDim[dim]) > 1:
                            raise Usage('Do not expect to have multiple ' + 
                                        'IndexEntry with the same dimension ' +
                                        'and function.')
                        else:
                            tmp.extend(sortedByDim[dim])
                    except KeyError:
                        pass
                if tmp:
                    filename = os.path.join(outputdir,'ppdata_f%d' % fun)
                    pptex.main(tmp, tabValsOfInterest, filename, verbose)

        if isrldistr:
            sortedByDim = indexEntries.sortByDim()
            for dim, sliceDim in sortedByDim.items():
                if dim in rldDimsOfInterest:
                    pprldistr.main(sliceDim, rldValsOfInterest,
                                   outputdir, 'dim%02dall' % dim, verbose)
                    sortedByFG = sliceDim.sortByFuncGroup()
                    #set_trace()
                    for fGroup, sliceFuncGroup in sortedByFG.items():
                        pprldistr.main(sliceFuncGroup, rldValsOfInterest,
                                       outputdir, 'dim%02d%s' % (dim, fGroup),
                                       verbose)

        #if verbose:
            #print 'total ps = %g\n' % (float(numpy.sum(ps))/numpy.sum(nbRuns))

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2


if __name__ == "__main__":
   sys.exit(main()) #TODO change this to deal with args?
