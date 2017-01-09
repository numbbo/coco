#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COmparing Continuous Optimisers (COCO) post-processing software

This package is meant to generate output figures and tables for the
benchmarking of continuous optimisers in the case of black-box
optimisation.
The post-processing tool takes as input data from experiments and
generates outputs that will be used in the generation of the LateX-
formatted article summarizing the experiments.

The main method of this package is :py:func:`cocopp.rungeneric.main`
This method allows to use the post-processing through a command-line
interface.

To obtain more information on the use of this package from the python
interpreter, assuming this package has been imported as ``bb``, type:
``help(bb.cococommands)``

"""

from __future__ import absolute_import

import matplotlib  # just to make sure the following is actually done first
matplotlib.use('Agg')  # To avoid window popup and use without X forwarding

from .cococommands import *

from .rungeneric import main as main

from .__main__ import main as _main  # depreciated

import pkg_resources

__all__ = ['comp2', 'compall', 'eaf', 'main', 'ppfigdim', 'pplogloss',
           'pprldistr', 'pproc', 'ppsingle', 'pptable', 'rungeneric',
           'rungeneric1', 'rungeneric2', 'rungenericmany', 'algportfolio',
           'pptex', 'toolsstats']

__version__ = pkg_resources.require('cocopp')[0].version
