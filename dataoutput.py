#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generates output either pickle files or elaborate data files. 1 file per
solver per function and per dimension (unit experiment).

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

# Will read in this file where to put the pickle files.
infofilename = 'algorithmshortinfos.txt'
infofile = os.path.join(os.path.split(__file__)[0], 'compall', infofilename)
algShortInfos = {}
algLongInfos = {}
algPlotInfos = {}
isAlgorithminfosFound = True
try:
    f = open(infofile,'r')
    for i, line in enumerate(f):
        if len(line) == 0 or line.startswith('%') or line.isspace() :
            continue
        try:
            algShortInfo, algId, comment, plotinfo = line.strip().split(':', 3)
            algShortInfos[(algId, comment)] = algShortInfo
            algLongInfos.setdefault(algShortInfo, []).append((algId, comment))
            # Could have multiple entries...
            algPlotInfos[(algId, comment)] = eval(plotinfo)
        except ValueError:
            # Occurs when the split line does not result in 4 elements.
            txt = ("\n  Line %d in %s\n  is not formatted correctly " % (i, infofile)
                   +"(see documentation of bbob_pproc.dataoutput.main)\n  "
                   +"and will be disregarded:\n    > %s" % (line))
            warnings.warn(txt)
    f.close()

except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    isAlgorithminfosFound = False
    print 'Could not find file', infofile, \
          'Will not generate any output.'

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def updateAlgorithmInfo(alg, verbose=True):
    """Input one pair of algorithm id and comment and update the text file.
    """

    try:
        f = open(infofile, 'a')
        if not alg in algLongInfos:
            algShortInfos[alg] = alg[0]
            algLongInfos.setdefault(alg[0], []).append(alg)
            f.write(':'.join([algShortInfos[alg],
                             ':'.join(alg),
                            '{"label":"%s"}\n' %  algShortInfos[alg]]))
            if verbose:
                print ('A new entry for %s was added in %s.'
                       % (alg, infofile))
        else:
            raise Usage('The entry %s is already listed' % alg)
    except:
        raise Usage('There was a problem here.')
    else:
        f.close()

def isListed(alg):
    res = True
    if not (alg in algLongInfos or alg in algShortInfos):
        warntxt = ('The algorithm %s is not an entry in %s.' %(alg, infofile))
        warnings.warn(warntxt)
        res = False
    return res


def outputPickle(dsList, verbose=True):
    """Generates pickle files from a DataSetList."""
    dictAlg = dsList.dictByAlg()
    for alg, entries in dictAlg.iteritems():
        if not isListed(alg):
            updateAlgorithmInfo(alg)

        if not os.path.exists(algShortInfos[alg]):
            os.mkdir(algShortInfos[alg])

        entries.pickle(outputdir=algShortInfos[alg], verbose=verbose)


def outputDataFiles(dsList, verbose=True):
    """Generates data files from a DataSetList."""
    dictAlg = dsList.dictByAlg()
    for alg, entries in dictAlg.iteritems():
        # TODO: entries.dataf(outputdiralgInfos[alg], verbose=verbose)
        pass


def usage():
    print main.__doc__


def main(argv=None):
    """
    Keyword arguments:
    argv -- list of strings containing options and arguments. If not provided,
    sys.argv is accessed.

    argv should list either names of info files or folders containing info
    files or folders containing pickle files (preferred).
    Furthermore, argv can begin with, in any order, facultative option
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

        $ python bbob_pproc/dataoutput.py -v

        $ python bbob_pproc/dataoutput.py experiment2/*.info


    * Loading this package and calling the main from the command line
      (requires that the path to this package is in python search path):

        $ python -m bbob_pproc.dataoutput -h

    This will print out this help message.

    * From the python interactive shell (requires that the path to this
      package is in python search path):

        >>> from bbob_pproc import dataoutput
        >>> dataoutput.main('folder1')

    This will execute the post-processing on the index files found in folder1.

    If you need to process new data, you must add a line in the file
    algorithmshortinfos.txt
    The line in question must have 4 fields separated by colon (:) character.
    The 1st must be the name of the folder which will contain the
    post-processed pickle data file, the 2nd is the exact string used as algId
    in the info files, the 3rd is the exact string for the comment. The 4th
    will be a python dictionary which will be use for the plotting.
    If different comment lines (3rd field) have been used for a single
    algorithm, there should be a line in algorithmshortinfos.txt corresponding
    to each of these.
    The line will be added automatically if it does not exist, the data will
    be put in a folder bearing the name of the algorithm.

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

        dsList = DataSetList(args)
        outputPickle(dsList, verbose=verbose)
        sys.exit()

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())
