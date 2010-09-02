#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for outputting python-formatted data.
1 file per solver per function and per dimension (unit experiment).
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

infofilename = 'algorithmshortinfos.txt'
infofile = os.path.join(os.path.split(__file__)[0], 'compall', infofilename)
algPlotInfos = {}
isAlgorithminfosFound = True
try:
    f = open(infofile,'r')
    for i, line in enumerate(f):
        if len(line) == 0 or line.startswith('%') or line.isspace() :
            continue
        try:
            algId, comment, plotinfo = line.strip().split(':', 2)
            # hack
            while comment[0] != '%':
                tmp, newcomment = comment.split(':', 1)
                algId = ':'.join((algId, tmp))
                comment = newcomment
            while plotinfo[0] != '{':
                tmp, newplotinfo = plotinfo.split(':', 1)
                comment = ':'.join((comment, tmp))
                plotinfo = newplotinfo
            algPlotInfos[(algId, comment)] = eval(plotinfo)
        except ValueError:
            # Occurs when the split line does not result in 3 elements.
            txt = ("\n  Line %d in %s\n  is not formatted correctly " % (i, infofile)
                   +"(see documentation of bbob_pproc.dataoutput.main)\n  "
                   +"and will be disregarded:\n    > %s" % (line))
            warnings.warn(txt)
    f.close()

except IOError, (errno, strerror):
    print "I/O error(%s): %s" % (errno, strerror)
    isAlgorithminfosFound = False
    print 'Could not find file: ', infofile

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def updateAlgorithmInfo(alg, verbose=True):
    """Input one pair of algorithm id and comment and update the text file.
    """

    try:
        f = open(infofile, 'a')
        if not alg in algPlotInfos:
            plotinfo = '{"label":"%s"}' % alg[0]
            #TODO find a default color and line style to use.
            #plotinfo = '{"label":"%s", "color": "c", "ls": "--"}' % alg[0]
            algPlotInfos[alg] = eval(plotinfo)
            f.write(':'.join(alg) + ':' + plotinfo + '\n')
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
    if not (alg in algPlotInfos):
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

        entries.pickle(verbose=verbose)

def usage():
    print main.__doc__

def main(argv=None):
    """Generate python-formatted data from raw BBOB experimental data.

    The raw experimental data (files with the extension 'info' pointing to
    files with extension 'dat' and 'tdat') are post-processed and stored in a
    more condensed way as files with the extension 'pickle'. Supposing the
    raw data are stored in folder 'mydata', the new pickle files will be put in
    folder 'mydata-pickle'.

    Running this will also add an entry in file algorithmshortinfos.txt if it
    does not exist already.
    algorithmshortinfos.txt is a file which contain meta-information that are
    used by modules from the bbob_pproc.compall package.
    The new entry in algorithmshortinfos.txt is represented as a new line
    appended at the end of the file.
    The line in question will have 3 fields separated by colon (:) character.
    The 1st field must be the exact string used as algId in the info files in
    your data, the 2nd the exact string for the comment. The 3rd will be
    a python dictionary which will be used for the plotting.

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

    * Calling the dataoutput.py interface from the command line:

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

        if (not verbose):
            warnings.simplefilter('ignore')

        dsList = DataSetList(args)
        outputPickle(dsList, verbose=verbose)
        sys.exit()

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())

