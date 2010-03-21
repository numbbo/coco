#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic routines for figure generation."""

from operator import itemgetter
from itertools import groupby
from matplotlib import pyplot as plt

def saveFigure(filename, figFormat=('eps', 'pdf'), verbose=True):
    """Save figure into a file.
    Need to make sure the file location is available.
    """

    #TODO: make sure the file is writeable.
    if isinstance(figFormat, basestring):
        plt.savefig(filename + '.' + figFormat, dpi = 300,
                    format=figFormat)
        if verbose:
            print 'Wrote figure in %s.' %(filename + '.' + figFormat)
    else:
        if not isinstance(figFormat, basestring):
            for entry in figFormat:
                plt.savefig(filename + '.' + entry, dpi = 300,
                            format=entry)
                if verbose:
                    print 'Wrote figure in %s.' %(filename + '.' + entry)

def consecutiveNumbers(data):
    """Find runs of consecutive numbers using groupby. 
    The key to the solution is differencing with a range so that consecutive
    numbers all appear in same group.
    Interesting for determining ranges of functions...
    """

    res = []
    for k, g in groupby(enumerate(data), lambda (i,x):i-x):
        tmp = list(str(i) for i in map(itemgetter(1), g))
        if len(tmp) <= 2 :
            res.append(', '.join(tmp))
        else:
            res.append('-'.join((tmp[0], tmp[-1])))

    return ', '.join(res)
