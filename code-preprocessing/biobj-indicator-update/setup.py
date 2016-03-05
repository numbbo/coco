#==============================================================================
# from distutils.core import setup
# from Cython.Distutils import build_ext
# from distutils.extension import Extension
# 
# import numpy
# 
# sourcefiles = ["archive.pyx", "coco.c"]
# 
# extensions = [Extension("coco_archive", 
#                         sources=sourcefiles,
#                         include_dirs=[numpy.get_include()])]
# 
# setup(
#     ext_modules = extensions,
#     cmdclass={'build_ext': build_ext},
# )
#==============================================================================
#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import setuptools
from distutils.core import setup
from distutils.extension import Extension
import numpy as np
from Cython.Build import cythonize

cmdclass = {}
extensions = []

print("NOTE: Using Cython to build interface.")
# we rename file interface.pyx to _interface.pyx to possibly avoid import error later
from Cython.Distutils import build_ext
cmdclass.update({'build_ext': build_ext})
interface_file = 'archive.pyx'
    
extensions.append(Extension('archive',
                            sources=[interface_file, 'coco.c'],
                            include_dirs=[np.get_include()]))

setup(
    ext_modules = extensions,
    cmdclass={'build_ext': build_ext},
    name = 'archive',
    version = "0.1",
    packages = ['archive']
)