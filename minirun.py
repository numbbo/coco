#!/usr/bin/env python

"""Mini run to display the performance profiles of algorithms.

"""

from __future__ import absolute_import

import os
import sys
import getopt
#import matplotlib
#matplotlib.use('Agg') # To avoid window popup and use without X forwarding

# Add the path to bbob_pproc
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc.pproc2 import DataSetList
from bbob_pproc import ppperfprof

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
            opts, args = getopt.getopt(argv, "hvo:",
                                       ["help", "output-dir",
                                        "pickle", "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        verbose = False
        outputdir = 'defaultoutputdirectory'

        #Process options
        for o, a in opts:
            if o in ("-v","--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-o", "--output-dir"):
                outputdir = a
            else:
                assert False, "unhandled option"

        dsList = DataSetList(args)

        if not dsList:
            sys.exit()

        # Write the pickle files

        # Get the target function values depending on the function
        # target = dict(...)

        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
            if verbose:
                print 'Folder %s was created.' % (outputdir)

        # Performance profiles
        dictDim = dsList.dictByDim()
        for d, entries in dictDim.iteritems():
            ppperfprof.main(entries, target=1e-8, outputdir=outputdir,
                            info=('%02d' % d), verbose=verbose)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
   sys.exit(main())