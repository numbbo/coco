#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COmparing Continuous Optimisers (COCO):

This package is meant to generate output figures and tables for the
benchmarking of continuous optimisers in the case of black-box optimisation.
The post-processing tool takes as input data from experiments and generates
output that will be used in the generation of the LateX-formatted article
summarizing the experiments.

"""
import sys
from bbob_pproc.run import Usage as Usage
from bbob_pproc.run import main as main

__all__  = ['readalign', 'pptex', 'pprldistr', 'findfiles',
            'main', 'ppfigdim', 'pplogloss', 'pproc', 'dataoutput', 'Usage']

if __name__ == "__main__":
    sys.exit(main())
