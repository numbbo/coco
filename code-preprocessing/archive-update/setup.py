#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools
from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from Cython.Build import cythonize

extensions = []

print("Using Cython to build interface.")
    
extensions.append(Extension('archive',
                            sources=['archive.pyx', 'coco.c'],
                            include_dirs=[np.get_include()]))

from Cython.Distutils import build_ext
setup(
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},
    name='archive',
    version="0.1"
)