#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process data to be included in a generic template.

   Synopsis:
      python path_to_folder/bbob_pproc/rungeneric.py [OPTIONS] FOLDER_NAME...
    Help:
      python path_to_folder/bbob_pproc/rungeneric.py -h

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
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import dataoutput, pproc, rungeneric1, rungeneric2, rungenericmany
from bbob_pproc.dataoutput import algPlotInfos
from bbob_pproc.pproc import DataSetList, processInputArgs
from bbob_pproc.compall import ppperfprof, pptables
from bbob_pproc.compall import organizeRTDpictures

# Combine optlist for getopt:
# Make a set of the short option list, has one-letter elements that could be
# followed by colon
shortoptlist = set()
for i in (rungeneric1.shortoptlist, rungeneric2.shortoptlist,
          rungenericmany.shortoptlist):
    tmp = i[:]
    # split into logical elements: one-letter that could be followed by colon
    while tmp:
        if len(tmp) > 1 and tmp[1] is ':':
            shortoptlist.add(tmp[0:2])
            tmp = tmp[2:]
        else:
            shortoptlist.add(tmp[0])
            tmp = tmp[1:]
shortoptlist = ''.join(shortoptlist)

longoptlist = list(set.union(set(rungeneric1.longoptlist),
                             set(rungeneric2.longoptlist),
                             set(rungenericmany.longoptlist)))

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#FUNCTION DEFINITIONS

def splitshortoptlist(shortoptlist):
    """Split short options list used by getopt.
    Returns a set of the options.
    """

    res = set()
    tmp = shortoptlist[:]
    # split into logical elements: one-letter that could be followed by colon
    while tmp:
        if len(tmp) > 1 and tmp[1] is ':':
            res.add(tmp[0:2])
            tmp = tmp[2:]
        else:
            res.add(tmp[0])
            tmp = tmp[1:]

    return res

def usage():
    print main.__doc__

def main(argv=None):
    """Main routine for post-processing data from BBOB.
    This routine will call sub-routine rungeneric1 for each input arguments
    and either sub-routines rungeneric2 (2 input arguments) or rungenericmany
    (more than 2) for the input arguments altogether.

    The output figures and tables are included in
    * template1generic.tex, template1ecj.tex, noisytemplate1generic.tex,
    noisytemplate1ecj.tex for single algorithm results on the noise-free and
    noisy testbeds respectively
    * template2generic.tex, template2ecj.tex, noisytemplate2generic.tex,
    noisytemplate2ecj.tex for showing the comparison of 2 algorithms
    * template3generic.tex, template3ecj.tex, noisytemplate3generic.tex,
    noisytemplate3ecj.tex for showing the comparison of more than 2 algorithms.
    Given that the folder with all output figures and tables and the template
    files are in the current working directory, compiling the template file
    with LaTeX should produce a document.

    Keyword arguments:
    argv -- list of strings containing options and arguments. If not provided,
    sys.argv is accessed.

    argv must list folders containing BBOB data files. Each of these folders
    should correspond to the data of ONE algorithm.

    Furthermore, argv can begin with, in any order, facultative option flags.

        -h, --help

            display this message

        -v, --verbose

            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

        -o, --output-dir OUTPUTDIR

            change the default output directory ('ppdata') to
            OUTPUTDIR

    Exceptions raised:
    Usage -- Gives back a usage message.

    Examples:

    * Calling the rungenericmany.py interface from the command line:

        $ python bbob_pproc/rungeneric.py -v


    * Loading this package and calling the main from the command line
      (requires that the path to this package is in python search path):

        $ python -m bbob_pproc.rungeneric -h

    This will print out this help message.

    * From the python interactive shell (requires that the path to this
      package is in python search path):

        >>> from bbob_pproc import rungeneric
        >>> rungeneric.main('-o outputfolder folder1 folder2'.split())

    This will execute the post-processing on the data found in folder1
    and folder2.
    The -o option changes the output folder from the default ppdata to
    outputfolder.

    * Generate post-processing data for some algorithms:

        $ python rungeneric.py AMALGAM BFGS BIPOP-CMA-ES

    """

    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, shortoptlist, longoptlist)
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        verbose = False
        outputdir = 'ppdata'

        #Process options
        shortoptlist1 = list("-" + i.rstrip(":")
                             for i in splitshortoptlist(rungeneric1.shortoptlist))
        shortoptlist2 = list("-" + i.rstrip(":")
                             for i in splitshortoptlist(rungeneric2.shortoptlist))
        shortoptlistmany = list("-" + i.rstrip(":")
                                for i in splitshortoptlist(rungenericmany.shortoptlist))
        shortoptlist1.remove("-o")
        shortoptlist2.remove("-o")
        shortoptlistmany.remove("-o")
        longoptlist1 = list( "--" + i.rstrip("=") for i in rungeneric1.longoptlist)
        longoptlist2 = list( "--" + i.rstrip("=") for i in rungeneric2.longoptlist)
        longoptlistmany = list( "--" + i.rstrip("=") for i in rungenericmany.longoptlist)

        genopts1 = []
        genopts2 = []
        genoptsmany = []
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-o", "--output-dir"):
                outputdir = a
            else:
                isAssigned = False
                if o in longoptlist1 or o in shortoptlist1:
                    genopts1.append(o)
                    # Append o and then a separately otherwise the list of
                    # command line arguments might be incorrect
                    if a:
                        genopts1.append(a)
                    isAssigned = True
                if o in longoptlist2 or o in shortoptlist2:
                    genopts2.append(o)
                    if a:
                        genopts2.append(a)
                    isAssigned = True
                if o in longoptlistmany or o in shortoptlistmany:
                    genoptsmany.append(o)
                    if a:
                        genoptsmany.append(a)
                    isAssigned = True
                if o in ("-v","--verbose"):
                    verbose = True
                    isAssigned = True
                if not isAssigned:
                    assert False, "unhandled option"

        if (not verbose):
            warnings.simplefilter('ignore')

        print ("BBOB Post-processing: will generate output " +
               "data in folder %s" % outputdir)
        print "  this might take several minutes."

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            if verbose:
                print 'Folder %s was created.' % (outputdir)

        for alg in args:
            tmpoutputdir = os.path.join(outputdir, os.path.split(alg)[-1])
            # TODO: if there is a problem, skip: try except...
            rungeneric1.main(genopts1
                             + ["-o", tmpoutputdir, alg])
        if len(args) == 2:
            rungeneric2.main(genopts2 + ["-o", outputdir] + args)
        elif len(args) > 2:
            rungenericmany.main(genoptsmany + ["-o", outputdir] + args)
    #TODO prevent loading the data every time...

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())

