#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for checking data sets.

Synopsis:
    ``python path_to_folder/bbob_pproc/sanitycheck.py [OPTIONS] FOLDER``

Help:
    ``python path_to_folder/bbob_pproc/sanitycheck.py -h``

"""

from __future__ import absolute_import
from __future__ import with_statement # This isn't required in Python 2.6

import os
import sys
import warnings
import getopt
import pickle
import ast
from pdb import set_trace

# Add the path to bbob_pproc
if __name__ == "__main__":
    # os.path.split is system independent
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc import findfiles
from bbob_pproc.pproc import DataSetList
from bbob_pproc.pproc import parseinfo
from bbob_pproc.readalign import split

# Used by getopt:
shortoptlist = "hvpfo:"
longoptlist = ["help", "output-dir=", "noisy", "noise-free", "tab-only",
               "fig-only", "rld-only", "los-only", "crafting-effort=",
               "pickle", "verbose", "settings=", "conv"]

"""Use cases:

* Check instances and repetitions
* Ill-terminated runs -> do something?
* Default algId and comment line? -> edit?
* Some statistics?

"""

crit_attr = ('DIM', 'Precision', 'algId')
correct_instances2010 = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,
                         11:1, 12:1, 13:1, 14:1, 15:1}
correct_instances2009 = {1:3, 2:3, 3:3, 4:3, 5:3}

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


#FUNCTION DEFINITIONS

def checkinfofile(filename, verbose=True):
    """Check the integrity of info files."""
    
    filepath = os.path.split(filename)[0]
    if verbose:
        print 'Checking %s' % filename

    def check_datfiles(s):
        """Check data from 3rd line in index entry.

        These are the lines with data files and run times information.

        """
        parts = s.split(', ')
        datfiles = []
        trials = []
        for elem in parts:
            if elem.endswith('dat'):
                filename = elem.replace('\\', os.sep)
                # *nix data to Windows processing
                filename = filename.replace('/', os.sep)
                root, ext = os.path.splitext(elem)
                root = os.path.join(filepath, root)
                dat = os.path.join(root + '.dat')
                tdat = os.path.join(root + '.tdat')
                if not (os.path.exists(dat) and os.path.exists(tdat)):
                    raise IOError
                else:
                    if verbose:
                        print 'Found data files %s.dat and %s.tdat' % (root, root)
                datfiles.extend((dat, tdat))
            else:
                if not ':' in elem:
                    warnings.warn('Caught an ill-finalized run in %s'
                                  % (filename))
                    trials.append(ast.literal_eval(elem))
                else:
                    itrial, info = elem.split(':', 1)
                    trials.append(ast.literal_eval(itrial))
                    readmaxevals, readfinalf = info.split('|', 1)
                    try:
                        float(readmaxevals)
                        float(readfinalf)
                    except ValueError:
                        raise
        return datfiles, trials

    with open(filename) as f:
        for i, line in enumerate(f):
            # Checking line by line
            tmp = i % 3
            msg = ''
            if tmp == 0.:
                try:
                    info = dict(parseinfo(line))
                except:
                    msg = ('Cannot parse key=value pairs (check that string '
                           'values are in-between brackets)')
            elif tmp == 1.:
                if not line.strip().startswith('%'):
                    msg = "Line does not start with '%' character."
            elif tmp == 2.:
                datfiles, trials = check_datfiles(line)
                # check data files' integrity
                split(datfiles)
                # check instances
                if not is_correct_instances(trials):
                    msg = ('The instances listed do not respect the '
                           'specifications: one repetition for each of '
                           'instances 1 to 15.')
            if msg:
                raise Usage('Problem in file %s, line %d: %s' % (filename, i+1, msg))

            # Checking by groups of 3 lines (1 index entry)
            if tmp == 2:
                miss_attr = list(i for i in crit_attr if not info.has_key(i))
                if miss_attr:
                    msg = ('File %s, entry l%d-%d is missing the following'
                           'keys: %s.' % (filename, i-2, i, ', '.join(miss_attr)))
                    raise Usage(msg)
                else:
                    if verbose:
                        print ('File %s, entry l%d-%d is ok.' %
                               (filename, i-2, i))

def is_correct_instances(trials, verbose=True):
    """Check instances and number of repetitions."""

    tmp = dict((j, trials.count(j)) for j in set(trials))
    return (tmp == correct_instances2010 or tmp == correct_instances2009)

def somestatistics():
    """Do some statistics over the data."""
    pass

def usage():
    print main.__doc__

def main(argv=None):
    """Main routine for COCO data checking procedure.
    
    The routine will stop at the first problem encountered.
    
    """
    if argv is None:
        argv = sys.argv[1:]
        # The zero-th input argument which is the name of the calling script is
        # disregarded.

    try:
        try:
            opts, args = getopt.getopt(argv, shortoptlist, longoptlist)
        except getopt.error, msg:
             raise Usage(msg)
        if not (args):
            usage()
            sys.exit()

        #Process options
        verbose = False
        for o, a in opts:
            if o in ('-v', '--verbose'):
                verbose = True

        print 'COCO Checking procedure: This may take a couple of minutes.'

        filelist = list()
        for i in args:
            if os.path.isdir(i):
                filelist.extend(findfiles.main(i, verbose))
            elif os.path.isfile(i):
                filelist.append(i)
            else:
                txt = 'Input file or folder %s could not be found.' % i
                raise Usage(txt)

        for i in filelist:
            try:
                if verbose:
                    print 'Checking %s.' % i
                extension = os.path.splitext(i)[1]
                if extension == '.info':
                    checkinfofile(i, verbose)
                elif extension == '.pickle':
                    # cocofy(i)
                    with open(i) as f:
                        ds = pickle.load(f)
                    if not is_correct_instances(ds.instancenumbers):
                        msg = ('File %s: The instances listed do not respect '
                               'the specifications BBOB-2009 or BBOB-2010.' % i)
                        raise Usage(msg)
            except Usage, err:
                print >>sys.stderr, err.msg
                continue
        print '... Done.'

    except Usage, err:
        print >>sys.stderr, err.msg
        #print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
