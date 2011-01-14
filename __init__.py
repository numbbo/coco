#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COmparing Continuous Optimisers (COCO) post-processing software

This package is meant to generate output figures and tables for the
benchmarking of continuous optimisers in the case of black-box optimisation.
The post-processing tool takes as input data from experiments and generates
output that will be used in the generation of the LateX-formatted article
summarizing the experiments.

To obtain more information on the use of this package from the python shell, type:

>> help bbob_pproc.cococommands

"""

from __future__ import absolute_import

import sys

#from bbob_pproc.run import main as main
from bbob_pproc.cococommands import *

__version__ = '10.7'

