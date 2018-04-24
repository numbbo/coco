#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COmparing Continuous Optimisers (COCO) post-processing software

This package is meant to generate output figures and tables for the
benchmarking of continuous optimisers in the case of black-box
optimisation.
The post-processing tool takes as input data from experiments and
generates outputs that will be used in the generation of the LateX-
formatted article summarizing the experiments.

The main method of this package is `cocopp.main` (currently aliased to
`cocopp.rungeneric.main`, which is **the best place to look for getting
a quick start**). This method allows to use the post-processing
through a command-line interface.

To obtain more information on the use of this package from the python
interpreter, type ``help(cocopp.cococommands)``, however remark that
this info might not be entirely up-to-date.

"""

from __future__ import absolute_import

import matplotlib  # just to make sure the following is actually done first
matplotlib.use('Agg')  # To avoid window popup and use without X forwarding
del matplotlib

from numpy.random import seed as set_seed

from .cococommands import *  # outdated
from . import config
from . import findfiles
from . import rungeneric

from .rungeneric import main

import pkg_resources

__all__ = [# 'main',  # import nothing with "from cocopp import *"
           ]

__version__ = pkg_resources.require('cocopp')[0].version

data_archive = findfiles.COCODataArchive()
_data_archive = data_archive  # should go away but some tests rely on this

bbob = findfiles.COCOBBOBDataArchive()
bbob_noisy = findfiles.COCOBBOBNoisyDataArchive()
bbob_biobj = findfiles.COCOBBOBBiobjDataArchive()

# clean up namespace
del absolute_import, pkg_resources
# del bestalg, captions, comp2, compall, htmldesc, pickle, ppconverrorbars
# del ppfig, ppfigdim, ppfigparam, pplogloss, pprldistr, pproc, pptable
# del pptex, readalign, rungeneric1, rungenericmany, toolsdivers, toolsstats

# cococommands, config, data_archive, dataformatsettings, findfiles,
# genericsettings, info, load, main, rungeneric, set_seed, systeminfo, testbedsettings,
