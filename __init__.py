#!/usr/bin/env python
# coding: utf-8

"""Black-Box Optimization Benchmarking (BBOB) post processing tool:

The BBOB post-processing tool takes as input data from BBOB experiments and
generates output that will be used in the generation of the LateX-formatted
article summarizing the experiments.

Keyword arguments:
argv -- list of strings containing options and arguments to the main function.
Synopsis: python bbob_pproc.py [OPTIONS] input-files

    Running bbob_pproc.py or importing the bbob_pproc package and running the
    main method will take as input the input files and generate post-processed
    data that will be output as convergence and ENFEs graphs, tex tables and
    running length distribution graphs according to the experimentation
    process described in the documentation of the BBOB. All the outputs will
    be saved in an output folder as files that will be included in a TeX file.

    -h, --help

        display this message

    -v, --verbose

        verbose mode, prints out operations. When not in verbose mode, no
        output is to be expected, except for errors.

            opts, args = getopt.getopt(argv[1:], "hvpno:",
                                       ["help", "pproc-files", "output-dir",
                                        "tab-only", "fig-only", "rld-only",
                                        "no-pickle","verbose"])
    -p, --pproc-files

        input files are expected to be files post processed by this tool.

    -n, --no-pickle

        prevents pickled post processed data files from being generated.

    -o, --output-dir output-dir

        change the default output directory ('ppdata') to output-dir

    --tab-only, --fig-only, --rld-only

        these options can be used to output respectively the tex tables,
        convergence and ENFEs graphs figures, run length distribution figures
        only. A combination of any two of these options results in no output.

Exceptions raised:
UsageError --
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

"""Given a list of index files, returns convergence and ENFEs graphs, tables,
Run length distribution graphs.
ENFEs graphs: any indexEntries (usually for a given dim and a given func)
conv graphs: any indexEntries (usually for a given dim and a given func)
tables: should be any indexEntries and any precision
rlDistr: any indexEntries
"""

__all__  = ['readindexfiles', 'findindexfiles', 'ppfig', 'pptex', 'pprldistr',
            'main', 'ppfigdim']

#colors = {2:'b', 3:'g', 5:'r', 10:'c', 20:'m', 40:'y'} #TODO colormaps!
tabDimsOfInterest = [5, 20]    # dimension which are displayed in the tables
tabValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
tabRanksOfInterest = (1, 2)

figValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)

rldDimsOfInterest = (5, 20)
rldValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


###############################################################################


#FUNCTION DEFINITIONS

def createIndexEntries(args, outputdir, isPickled, verbose=True):
    """Returns a list of post-processed ÏndexEntries from a list of file names.
    Keyword arguments:
    args -- list of strings being file names.
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
            indexFiles.append(findindexfiles.IndexFile(filepath,
                                                       filename))
        elif os.path.isdir(i):
            indexFiles.extend(findindexfiles.main(i,verbose))
        elif i.endswith('.pickle'):
            pickles.append(i)
        else:
            raise Usage('Expect as input argument either info files or '+
                        'a folder containing info files.')
            #TODO: how do we deal with this?
    indexFiles = set(indexFiles) #for unicity

    indexEntries = []
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

    #indexEntries = readindexfiles.main(indexFiles, indexEntries, verbose)
    if indexFiles:
        #If new indexEntries are added then some old indexEntries may need to
        #be updated.
        indexEntries = readindexfiles.main(indexFiles, indexEntries, verbose)
        if isPickled:
            for i in indexEntries:
                filename = os.path.join(outputdir, 'ppdata_f%d_%d'
                                                    %(i.funcId, i.dim))
                try:
                    f = open(filename + '.pickle','w')
                    pickle.dump(i, f)
                    f.close()
                    if verbose:
                        print 'Pickle in %s.' %(filename+'.pickle')
                except IOError, (errno, strerror):
                    print "I/O error(%s): %s" % (errno, strerror)
                except PicklingError:
                    print "Could not pickle %s" %(i)

    return indexEntries

#def generateFigures(sortByFunc, outputdir, verbose, isBenchmarkinfosFound):
    #"""Creates image files of the convergence graphs."""

    #for func in sortByFunc:
        #filename = os.path.join(outputdir,'ppdata_f%d' % (func))
        ## initialize matrix containing the table entries
        #fig = plt.figure()

        #for dim in sorted(sortByFunc[func]):
            #tmp = ppfig.generateData(sortByFunc[func][dim])

            #h = ppfig.createFigure(tmp[:,[0,1]], fig)
            #for i in h:
                #plt.setp(i,'color',colors[dim])
            #h = ppfig.createFigure(tmp[:,[0,-1]], fig)
            #for i in h:
                #plt.setp(h,'color',colors[dim],'linestyle','--')
            ##Do all this in createFigure?
        #if isBenchmarkinfosFound:
            #title = funInfos[func]
        #else:
            #title = ''

        #ppfig.customizeFigure(fig, filename, title=title,
                              #fileFormat=('eps','png'), labels=['', ''],
                              #scale=['log','log'], locLegend='best',
                              #verbose=verbose)
        #plt.close(fig)


#def generateTables(sortByFunc, outputdir, verbose):
    #"""Generate tex files containing tabular grouping post processed data."""

    #for func in sortByFunc:
        #filename = os.path.join(outputdir,'ppdata_f%d' % (func))
        ## initialize matrix containing the table entries
        #tabData = scipy.zeros(0)
        #entryList = list()     # list of entries which are displayed in the table

        #for dim in sorted(sortByFunc[func]):
            #entry = sortByFunc[func][dim]

            #if dimOfInterest.count(dim) != 0 and tabData.shape[0] == 0:
                ## Array tabData has no previous values.
                #tabData = entry.arrayTab
                #entryList.append(entry)
            #elif dimOfInterest.count(dim) != 0 and tabData.shape[0] != 0:
                ## Array tabData already contains values for the same function
                #tabData = scipy.append(tabData,entry.arrayTab[:,1:],1)
                #entryList.append(entry)

        #[header,format] = pproc.computevalues(None,None,header=True)
        #set_trace()
        #pptex.writeTable2(tabData, filename, entryList, fontSize='tiny',
                          #header=header, format=format, verbose=verbose)


#def generateRLDistributions(indexEntries, outputdir, verbose):
    #"""Generate image files of run length distribution figures."""
    #sortedIndexEntries = pprldistr.sortIndexEntries(indexEntries)
    ##set_trace()
    #for key, indexEntries in sortedIndexEntries.iteritems():
        #figureName = os.path.join(outputdir,'pprldistr_%s' %(key))
        #figureName = os.path.join(outputdir,'ppfvdistr_%s' %(key))

        #fig = plt.figure()
        #for j in range(len(valuesOfInterest)):
        ##for j in [0]:
            #maxEvalsFactor = 1e3 #TODO: Global?
            #tmp = pprldistr.main(indexEntries, valuesOfInterest[j],
                                 #maxEvalsFactor, verbose)
            #if not tmp is None:
                #plt.setp(tmp, 'color', colorsDataProf[j])
        #pprldistr.beautify(fig, figureName, maxEvalsFactor, verbose=verbose)
        #plt.close(fig)


def usage():
    print __doc__


def main(argv=None):
    if argv is None:
        argv = sys.argv
    try:

        try:
            opts, args = getopt.getopt(argv[1:], "hvpno:",
                                       ["help", "pproc-files", "output-dir",
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
            elif o in ("-p", "--pproc-files"):
                #Post processed data files
                isPostProcessed = True
                isPickled = False
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

        if isfigure or istab or isrldistr:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if verbose:
                    print '%s was created.' % (outputdir)

        if isfigure or istab:
            if isfigure:
                ppfigdim.main(indexEntries, figValsOfInterest, outputdir,
                              verbose)
            if istab:
                pptex.main(indexEntries, tabDimsOfInterest, tabValsOfInterest,
                           tabRanksOfInterest, outputdir, verbose)

        if isrldistr:
            pprldistr.main(indexEntries, rldValsOfInterest, outputdir, verbose)

        #if verbose:
            #print 'total ps = %g\n' % (float(scipy.sum(ps))/scipy.sum(nbRuns))

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
   sys.exit(main()) #TODO change this to deal with args?
