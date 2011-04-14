#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COmparing Continuous Optimisers (COCO) post-processing software

This package is meant to generate output figures and tables for the
benchmarking of continuous optimisers in the case of black-box
optimisation.
The post-processing tool takes as input data from experiments and
generates outputs that will be used in the generation of the LateX-
formatted article summarizing the experiments.

The main method of this package is :py:func:`bbob_pproc.rungeneric.main`
This method allows to use the post-processing through a command-line
interface.

To obtain more information on the use of this package from the python
interpreter, assuming this package has been imported as ``bb``, type:
``help(bb.cococommands)``

"""

from __future__ import absolute_import

import sys

from bbob_pproc.cococommands import *

from bbob_pproc.rungeneric import main as main

__all__  = ['comp2', 'compall', 'main', 'ppfigdim', 'pplogloss', 'pprldistr',
            'pproc', 'pptable', 'rungeneric', 'rungeneric1', 'rungeneric2',
            'rungenericmany']

__version__ = '10.7'
