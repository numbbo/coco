#! /usr/bin/env python

# Creates run length distribution figures.

from __future__ import absolute_import

import scipy
import matplotlib.pyplot as plt
from pdb import set_trace

def sortIndexEntries(indexEntries):
    """Puts the indexEntry instances into bins for the distribution."""

    if not indexEntries:
        return ()
    sorted = {}
    #The bins correspond to whether the dimension is greater than 10 or not.
    for i in indexEntries:
        if i.dim < 10:
            sorted.setdefault('dimlt10',[]).append(i)
        else:
            sorted.setdefault('dimgt10',[]).append(i)
    return sorted


def beautify(figHandle, figureName, maxEvalsFactor, fileFormat=('png','eps'),
             verbose=True):
    """Formats the figure of the run length distribution."""

    axisHandle = figHandle.gca()
    axisHandle.set_xscale('log')
    axisHandle.set_xlim((1.0,maxEvalsFactor))
    axisHandle.set_ylim((0.0,1.0))
    axisHandle.set_xlabel('log10(FEvals/DIM)')
    axisHandle.set_ylabel('Probability of success')
    # Grid options
    axisHandle.grid('True')
    xtic = axisHandle.get_xticks()
    newxtic = []
    for j in tmp:
        tmp2.append('%d' % round(scipy.log10(j)))
    axisHandle.set_xticklabels(newxtic)

    # Save figure
    for entry in fileFormat:
        plt.savefig(figureName + '.' + entry, dpi = 120,
                    format = entry)
        if verbose:
            print 'Wrote figure in %s.' %(figureName + '.' + entry)


def main(indexEntries, fvalueToReach, maxEvalsFactor=1e5, verbose=True):
    """Creates run length distributions from a sequence of indexEntries."""

    x = []
    nn = 0
    for i in indexEntries:
        if i.rlDistr.has_key(fvalueToReach):
            for j in i.rlDistr[fvalueToReach]:
                x.append(j/float(i.dim))
        nn += i.nbRuns

    x.sort()
    n = len(x)
    res = None
    if n > 0:
        x2 = scipy.hstack([scipy.repeat(x, 2), maxEvalsFactor])
        #not efficient if some vals are repeated a lot
        y2 = scipy.hstack([0.0, scipy.repeat(scipy.arange(1,n)/float(nn),2), 
                           n/float(nn),n/float(nn)])
        res = plt.plot(x2, y2)

    return res
