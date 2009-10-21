#!/usr/bin/env python

"""Process data and generates pickle data files from raw data files.
   Synopsis:
      python path_to_folder/bbob_pproc/generate_pickle.py [OPTIONS] FILE_NAME FOLDER_NAME...
    Help:
      python path_to_folder/bbob_pproc/generate_pickle.py -h

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

from bbob_pproc.compall import dataoutput
from bbob_pproc.pproc import DataSetList

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#FUNCTION DEFINITIONS

def usage():
    print main.__doc__


def main(argv=None):
    """
    Keyword arguments:
    argv -- list of strings containing arguments. If not provided,
    sys.argv is accessed.

    argv should list either names of info files or folders containing info
    files. Furthermore, argv can begin with, in any order, facultative option
    flags listed below.

        -h, --help

            display this message

        -v, --verbose
 
            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

    Exceptions raised:
    Usage -- Gives back a usage message.

    Examples:

    * Calling the minirun.py interface from the command line:

        $ python bbob_pproc/generate_pickle.py -h

        $ python bbob_pproc/generate_pickle.py experiment2/*.info

    """

    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, "hv",
                                       ["help", "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        verbose = False

        #Process options
        for o, a in opts:
            if o in ("-v","--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            else:
                assert False, "unhandled option"

        # Write the pickle files if needed!
        dsList = DataSetList(args)
        dataoutput.outputPickle(dsList, verbose=verbose)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
   sys.exit(main())
