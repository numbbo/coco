#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic routines for figure generation."""

from matplotlib import pyplot as plt

def saveFigure(filename, figFormat=('eps', 'pdf'), verbose=True):
    """Save figure into a file.
    Need to make sure the file location is available.
    """

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
