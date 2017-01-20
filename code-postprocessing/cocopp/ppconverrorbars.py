#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process data and generates some comparison results."""

from __future__ import absolute_import

import os, sys
import numpy
from pdb import set_trace
from . import toolsdivers

# Add the path to cocopp
if __name__ == "__main__":
    # append path without trailing '/cocopp', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from . import genericsettings, pproc
from .ppfig import save_figure, save_single_functions_html
from .toolsstats import prctile

import matplotlib.pyplot as plt

final_target = 1e-8  # comes from the original experimental setup
warned = False  # print just one warning and set to True

#FUNCTION DEFINITIONS


def rearrange(blist, flist):
    """Alligns the number of evaluations taken from the blist with the
       corresponding flist"""
    final_b = []
    final_f = []
    for i in range(0, len(blist)): #runs over dimensions
        erg_b = numpy.empty((0), float)
        erg_f = [numpy.empty((0), float), numpy.empty((0), float), numpy.empty((0), float)]
        for j in range(0, len(blist[i])): #runs over function evaluations
            erg_b = numpy.append(erg_b, blist[i][j])
            erg_f[0] = numpy.append(erg_f[0], numpy.median(flist[i][j]))
            erg_f[1] = numpy.append(erg_f[1], prctile(flist[i][j], [0.25]))
            erg_f[2] = numpy.append(erg_f[2], prctile(flist[i][j], [0.75]))
        final_b.append(erg_b)
        final_f.append(erg_f)
    return final_b, final_f


def beautify():
    toolsdivers.legend(loc=3)
    plt.grid(True)
    limits = plt.ylim()
    plt.ylim(max((limits[0], final_target)), limits[1])

def main(dictAlg, outputdir='.', parentHtmlFileName=None):
    """Main routine for generating convergence plots

    """
    global warned  # bind variable warned into this scope
    dictFun = pproc.dictAlgByFun(dictAlg)
    for l in dictFun:  # l appears to be the function id!?
        for i in dictFun[l]: # please, what is i??? appears to be the algorithm-key
            plt.figure()
            if 1 < 3:  # no algorithm name in filename, as everywhere else
                figurename = "ppconv_" + "f%03d" % l
            else:  # previous version with algorithm name, but this is not very practical later
                if type(i) in (list, tuple):
                    figurename = "ppconv_plot_" + i[0] + "_f" + str(l)
                else:
                    try:
                        figurename = "ppconv_plot_" + dictFun[l][i].algId + "_f" + str(l)
                    except AttributeError:  # this is a (rather desperate)
                                            # bug-fix attempt that works for
                                            # the unit test
                        figurename = "ppconv_plot_" + dictFun[l][i][0].algId + "_f" + str(l)
            plt.xlabel('number of function evaluations / dimension')
            plt.ylabel('Median of fitness')
            plt.grid()
            ax = plt.gca()
            ax.set_yscale("log")
            ax.set_xscale("log")
            for j in dictFun[l][i]: # please, what is j??? a dataset
                dimList_b = []
                dimList_f = []
                dimList_b.append(j.funvals[:, 0])
                dimList_f.append(j.funvals[:, 1:])
                bs, fs = rearrange(dimList_b, dimList_f)
                labeltext = str(j.dim) + "D"
                try:
                    if 11 < 3:
                        plt.errorbar(bs[0] / j.dim, fs[0][0],
                                     yerr=[fs[0][1], fs[0][2]],
                                     label=labeltext)
                    else:
                        plt.errorbar(bs[0] / j.dim, fs[0][0], label=labeltext)
                except FloatingPointError:  # that's a bit of a hack
                    if 1 < 3 or not warned:
                        print('Warning: floating point error when plotting errorbars, ignored')
                    warned = True
            beautify()
            save_figure(os.path.join(outputdir, figurename.replace(' ', '')),
                        genericsettings.getFigFormats())
            plt.close()
    try:
        algname = str(dictFun[l].keys()[0][0])
    except KeyError:
        algname = str(dictFun[l].keys()[0])
    save_single_functions_html(os.path.join(outputdir, 'ppconv'),
                               algname,
                               parentFileName=parentHtmlFileName)  # first try


if __name__ == "__main__":
    sys.exit(main())
