#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Single data set results output module."""

import matplotlib.pyplot as plt
import numpy

def beautify():
    a = plt.gca()
    a.set_xscale('log')
    a.set_yscale('log')
    a.grid()
    a.set_ylabel('log10 of Df')
    a.set_xlabel('log10 of run lengths')
    tmp = a.get_yticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    a.set_yticklabels(tmp2)
    tmp = a.get_xticks()
    tmp2 = []
    for i in tmp:
        tmp2.append('%d' % round(numpy.log10(i)))
    a.set_xticklabels(tmp2)
    # TODO: label size and line width...

def plot(dataset, kwargs={}):
    """Plot function values versus function evaluations."""

    res = []
    for i in dataset.funvals[:, 1:].transpose(): # loop over the rows of the transposed array
        h = plt.plot(dataset.funvals[:, 0], i, **kwargs)
        res.extend(h)

    #TODO: make sure each line have the same properties plot properties
    return res

def plot2(dataset):
    """Plot function values versus function evaluations.

    + markers for final function values and run lengths, dash max and min (or 10 and 90%?)
    different markers for... ? see toolplot scilab
    
    """

    #TODO
    pass

def generatefig(dsList):
    """Plot function values versus function evaluations for multiple sets."""

    for i in dsList:
        plot(i)
        beautify()

def main(dsList, outputdir, verbose=True):
    """Generate output image files of function values vs. function evaluations."""

    generatefigure(dsList)
    # TODO: save, close

