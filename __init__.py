#!/usr/bin/env python
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

import scipy
import matplotlib.pyplot as plt

from pdb import set_trace
from bbob_pproc import readindexfiles, findindexfiles
from bbob_pproc import pproc, ppfig, pptex, pprldistr

"""Given a list of index files, returns convergence and ENFEs graphs, tables,
Run length distribution graphs.
ENFEs graphs: any indexEntries (usually for a given dim and a given func)
conv graphs: any indexEntries (usually for a given dim and a given func)
tables: should be any indexEntries and any precision
rlDistr: any indexEntries
"""

__all__  = ['readindexfiles','findindexfiles','ppfig','pptex','pprldistr',
            'main']

plt.rc("axes", labelsize=16, titlesize=24)
plt.rc("xtick", labelsize=16)
plt.rc("ytick", labelsize=16)
plt.rc("font", size=16)
plt.rc("legend", fontsize=16)

sp1index = 1
medianindex = 5
colors = {2:'b', 3:'g', 5:'r', 10:'c', 20:'m', 40:'y'} #TODO colormaps!
dimOfInterest = [5,20]    # dimension which are displayed in the tables
valuesOfInterest = [1.0,1.0e-2,1.0e-4,1.0e-6,1.0e-8]
colorsDataProf = ['b', 'g', 'r', 'c', 'm']

#Either we read it in a file (flexibility) or we hard code it here.
funInfos = {}
isBenchmarkinfosFound = True
try:
    infofile = os.path.join(os.path.split(__file__)[0], '..', '..', 
                            'benchmarkinfos')
    f = open(infofile,'r')
    for line in f:
        if len(line) == 0 or line.startswith('%') or line.isspace() :
            continue
        funcId, funcInfo = line[0:-1].split(None,1)
        funInfos[int(funcId)] = funcId + ' ' + funcInfo
    f.close()
except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    isBenchmarkinfosFound = False
    print 'Could not find benchmarkinfos file. '\
          'Titles in ENFEs and convergence figures will not be displayed.'


#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


###############################################################################


#FUNCTION DEFINITIONS

def processInputArgs(args, isPostProcessed, verbose=True):
    ps = []
    nbRuns = []

    if isPostProcessed:
        indexEntries = []
        tmp = set(args) #for unicity
        for i in tmp:
            try:
                f = open(i,'r')
                entry = pickle.load(f)
                f.close()
                if verbose:
                    print 'Unpickled %s.' % (i)
                indexEntries.append(entry)
                ps.append(entry.arrayTab[-1,4])
                nbRuns.append(entry.nbRuns)
            except IOError, (errno, strerror):
                print "I/O error(%s): %s" % (errno, strerror)
            except UnpicklingError:
                f.close()
                print '%s could not be unpickled.' %(i)
                if not i.endswith('.pickle'):
                    print '%s might not be a pickle data file.' %(i)

    else:
        indexFiles = []
        for i in args:
            if i.endswith('.info'):
                (filepath,filename) = os.path.split(i)
                indexFiles.append(findindexfiles.IndexFile(filepath,
                                                           filename))
            elif os.path.isdir(i):
                indexFiles.extend(findindexfiles.main(i,verbose))
            else:
                raise Usage('Expect as input argument either info files or '+
                            'a folder containing info files.')
                #TODO: how do we deal with this?
        indexFiles = set(indexFiles) #for unicity

        indexEntries = readindexfiles.main(indexFiles,verbose)

        for entry in indexEntries:
            pproc.main(entry,verbose)
            ps.append(entry.arrayTab[-1,4])
            nbRuns.append(entry.nbRuns)

    return indexEntries, ps, nbRuns


def sortIndexEntries(indexEntries, outputdir, isPickled=True, verbose=True):
    """From a list of IndexEntry, returs a post-processed sorted dictionary."""
    sortByFunc = {}
    for elem in indexEntries:
        sortByFunc.setdefault(elem.funcId,{})
        sortByFunc[elem.funcId][elem.dim] = elem
        if isPickled:
            filename = os.path.join(outputdir, 'ppdata_f%d_%d' 
                                                %(elem.funcId,elem.dim))
            try:
                f = open(filename + '.pickle','w')
                pickle.dump(elem,f)
                f.close()
                if verbose:
                    print 'Pickle in %s.' %(filename+'.pickle')
            except IOError, (errno, strerror):
                print "I/O error(%s): %s" % (errno, strerror)
            except PicklingError:
                print "Could not pickle %s" %(elem)
    return sortByFunc


def genFig(sortByFunc, outputdir, verbose, isBenchmarkinfosFound):
    for func in sortByFunc:
        filename = os.path.join(outputdir,'ppdata_f%d' % (func))
        # initialize matrix containing the table entries
        fig = plt.figure()

        for dim in sorted(sortByFunc[func]):
            entry = sortByFunc[func][dim]

            h = ppfig.createFigure(entry.arrayFullTab[:,[sp1index,0]], fig)
            for i in h:
                plt.setp(i,'color',colors[dim])
            h = ppfig.createFigure(entry.arrayFullTab[:,[medianindex,0]], fig)
            for i in h:
                plt.setp(h,'color',colors[dim],'linestyle','--')
            #Do all this in createFigure?    
        if isBenchmarkinfosFound:
            title = funInfos[entry.funcId]
        else:
            title = ''

        ppfig.customizeFigure(fig, filename, title=title,
                              fileFormat=('eps','png'), labels=['', ''],
                              scale=['log','log'], locLegend='best',
                              verbose=verbose)
        plt.close(fig)


def genTab(sortByFunc, outputdir, verbose):
    for func in sortByFunc:
        filename = os.path.join(outputdir,'ppdata_f%d' % (func))
        # initialize matrix containing the table entries
        tabData = scipy.zeros(0)
        entryList = list()     # list of entries which are displayed in the table

        for dim in sorted(sortByFunc[func]):
            entry = sortByFunc[func][dim]

            if dimOfInterest.count(dim) != 0 and tabData.shape[0] == 0:
                # Array tabData has no previous values.
                tabData = entry.arrayTab
                entryList.append(entry)
            elif dimOfInterest.count(dim) != 0 and tabData.shape[0] != 0:
                # Array tabData already contains values for the same function
                tabData = scipy.append(tabData,entry.arrayTab[:,1:],1)
                entryList.append(entry)

        [header,format] = pproc.computevalues(None,None,header=True)
        pptex.writeTable2(tabData, filename, entryList, fontSize='tiny',
                          header=header, format=format, verbose=verbose)


def genDataProf(indexEntries, outputdir, verbose):
    sortedIndexEntries = pprldistr.sortIndexEntries(indexEntries)

    for key, indexEntries in sortedIndexEntries.iteritems():
        figureName = os.path.join(outputdir,'pprldistr_%s' %(key))

        fig = plt.figure()
        for j in range(len(valuesOfInterest)):
        #for j in [0]:
            maxEvalsFactor = 1e4
            tmp = pprldistr.main(indexEntries, valuesOfInterest[j],
                                maxEvalsFactor, verbose)
            if not tmp is None:
                plt.setp(tmp, 'color', colorsDataProf[j])
        pprldistr.beautify(fig, figureName, maxEvalsFactor, verbose=verbose)
        plt.close(fig)


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

        indexEntries, ps, nbRuns = processInputArgs(args, isPostProcessed,
                                                    verbose)

        if isfigure or istab or isrldistr:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if verbose:
                    print '%s was created.' % (outputdir)

        if isfigure or istab:
            sortByFunc = sortIndexEntries(indexEntries, outputdir, isPickled,
                                          verbose)
            if isfigure:
                genFig(sortByFunc, outputdir, verbose, isBenchmarkinfosFound)
            if istab:
                genTab(sortByFunc, outputdir, verbose)

        if isrldistr:
            genDataProf(indexEntries, outputdir, verbose)

        if verbose:
            print 'total ps = %g\n' % (float(scipy.sum(ps))/scipy.sum(nbRuns))

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2


if __name__ == "__main__":
   sys.exit(main()) #TODO change this to deal with args?
