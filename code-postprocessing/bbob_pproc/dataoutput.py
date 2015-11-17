#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for outputting python-formatted data.

1 file per solver per function and per dimension (unit experiment).

Example use:

* from the shell, assuming folder FOLDER contains raw experimental
  data::

    $ python pathtococo/python/bbob_pproc/dataoutput.py FOLDER

    Searching in FOLDER ...
    Searching in FOLDER/data_f1 ...
    ...
    Found ... file(s)!
    Processing FOLDER/....info.
    ...
    Saved pickle in FOLDER-pickle/....pickle.

This creates folder :file:`FOLDER-pickle` with python formatted files to
use with COCO.

"""

from __future__ import absolute_import

import os
import sys
import pickle
import warnings
import getopt

if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc.pproc import DataSetList

from pdb import set_trace

__all__ = ['main']

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def outputPickle(dsList, verbose=True):
    """Generates pickle files from a DataSetList."""
    dictAlg = dsList.dictByAlg()
    for alg, entries in dictAlg.iteritems():
        #if not _isListed(alg):
        #    updateAlgorithmInfo(alg)
        entries.pickle(verbose=verbose)

def usage():
    print main.__doc__

def main(argv=None):
    """Generate python-formatted data from raw BBOB experimental data.

    The raw experimental data (files with the extension :file:`info`
    pointing to files with extension :file:`dat` and :file:`tdat`) are
    post-processed and stored in a more condensed way as files with the
    extension :file:`pickle`.
    Supposing the raw data are stored in folder :file:`mydata`, the new
    pickle files will be put in folder :file:`mydata-pickle`.

    :keyword list argv: strings containing options and arguments. If not
                        provided, sys.argv is accessed.

    *argv* should list either names of info files or folders containing
    info files.
    Furthermore, *argv* can begin with, in any order, facultative option
    flags listed below.

        -h, --help

            display this message

    :exception Usage: Gives back a usage message.

    Examples:

    * Calling the dataoutput.py interface from the command line::

        $ python bbob_pproc/dataoutput.py experiment2/*.info

    * Loading this package and calling the main from the command line
      (requires that the path to this package is in the search path)::

        $ python -m bbob_pproc.dataoutput -h

      This will print out this help message.

    * From the python interactive shell (requires that the path to this
      package is in python search path)::

        >> import bbob_pproc as bb
        >> bb.dataoutput.main('folder1')

    """
    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, "h",
                                       ["help"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        verbose = False

        #Process options
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                sys.exit()
            else:
                assert False, "unhandled option"

        if (not verbose):
            warnings.simplefilter('ignore')

        dsList = DataSetList(args)
        outputPickle(dsList, verbose=True)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
