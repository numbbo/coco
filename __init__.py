#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Black-Box Optimization Benchmarking (BBOB) post processing tool:

The BBOB post-processing tool takes as input data from BBOB experiments and
generates output that will be used in the generation of the LateX-formatted
article summarizing the experiments.

"""
import sys
from bbob_pproc.run import main as main

__all__  = ['readalign', 'pptex', 'pprldistr', 'findfiles',
            'main', 'ppfigdim', 'pproc', 'dataoutput']

if __name__ == "__main__":
    sys.exit(main())
