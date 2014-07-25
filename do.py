#!/usr/bin/env python
## -*- mode: python -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import re
import os
import shutil
import tempfile

## Change to the root directory of repository and add our tools/
## subdirectory to system wide search path for modules.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('tools'))

from amalgamate import amalgamate
from cocoutils import make, run, python2, python3
from cocoutils import copy_file, expand_file, write_file
from cocoutils import hg_version, hg_revision

core_files = ['src/coco_benchmark.c', 'src/coco_random.c',
              'src/coco_generics.c', 'src/coco_c_runtime.c']

################################################################################
## C
def build_c():
    amalgamate(core_files + ['src/coco_c_runtime.c'],  'build/c/coco.c')
    copy_file('src/coco.h', 'build/c/coco.h')
    copy_file('src/bbob2009_testcases.txt', 'build/c/bbob2009_testcases.txt')
    write_file(hg_revision(), "build/c/REVISION")
    write_file(hg_version(), "build/c/VERSION")

def test_c():
    build_c()
    make("build/c", "clean")
    make("build/c", "all")
    run('build/c', ['./coco_test', 'bbob2009_testcases.txt'])

################################################################################
## Python 2
def build_python2():
    amalgamate(core_files + ['src/coco_c_runtime.c'],  'build/python/cython/coco.c')
    copy_file('src/coco.h', 'build/python/cython/coco.h')
    copy_file('src/bbob2009_testcases.txt', 'build/python/bbob2009_testcases.txt')
    expand_file('build/python/README.in', 'build/python/README',
                {'COCO_VERSION': hg_version()})
    expand_file('build/python/setup.py.in', 'build/python/setup.py',
                {'COCO_VERSION': hg_version()})
    ## Force distutils to use Cython
    os.environ['USE_CYTHON'] = 'true'
    python2('build/python', ['setup.py', 'sdist'])
    os.environ.pop('USE_CYTHON')

def test_python2():
    build_python2()
    python2('build/python', ['setup.py', 'check', '--metadata', '--strict'])
    ## Now install into a temporary location, run test and cleanup
    python_temp_home = tempfile.mkdtemp(prefix="coco")
    python_temp_lib = os.path.join(python_temp_home, "lib", "python")
    try:
        ## We setup a custom "homedir" here into which we install our
        ## coco extension and then use that temporary installation for
        ## the tests. Otherwise we would run the risk of contaminating
        ## the Python installation of the build/test machine.
        os.makedirs(python_temp_lib)
        os.environ['PYTHONPATH'] = python_temp_lib
        os.environ['USE_CYTHON'] = 'true'
        python2('build/python', ['setup.py', 'install', '--home', python_temp_home])
        python2('build/python', ['coco_test.py', 'bbob2009_testcases.txt'])
        os.environ.pop('USE_CYTHON')
        os.environ.pop('PYTHONPATH')
    finally:
        shutil.rmtree(python_temp_home)
        pass

################################################################################
## Python 3
def build_python3():
    amalgamate(core_files + ['src/coco_c_runtime.c'],  'build/python/cython/coco.c')
    copy_file('src/coco.h', 'build/python/cython/coco.h')
    copy_file('src/bbob2009_testcases.txt', 'build/python/bbob2009_testcases.txt')
    expand_file('build/python/README.in', 'build/python/README',
                {'COCO_VERSION': hg_version()})
    expand_file('build/python/setup.py.in', 'build/python/setup.py',
                {'COCO_VERSION': hg_version()})
    ## Force distutils to use Cython
    os.environ['USE_CYTHON'] = 'true'
    python3('build/python', ['setup.py', 'sdist'])
    os.environ.pop('USE_CYTHON')

def test_python3():
    build_python3()
    python3('build/python', ['setup.py', 'check', '--metadata', '--strict'])
    ## Now install into a temporary location, run test and cleanup
    python_temp_home = tempfile.mkdtemp(prefix="coco")
    python_temp_lib = os.path.join(python_temp_home, "lib", "python")
    try:
        ## We setup a custom "homedir" here into which we install our
        ## coco extension and then use that temporary installation for
        ## the tests. Otherwise we would run the risk of contaminating
        ## the Python installation of the build/test machine.
        os.makedirs(python_temp_lib)
        os.environ['PYTHONPATH'] = python_temp_lib
        os.environ['USE_CYTHON'] = 'true'
        python3('build/python', ['setup.py', 'install', '--home', python_temp_home])
        python3('build/python', ['coco_test.py', 'bbob2009_testcases.txt'])
        os.environ.pop('USE_CYTHON')
        os.environ.pop('PYTHONPATH')
    finally:
        shutil.rmtree(python_temp_home)
        pass

################################################################################
## R
def build_r():
    amalgamate(core_files + ['src/coco_r_runtime.c'],  'build/r/skel/src/coco.c')
    copy_file('src/coco.h', 'build/r/skel/src/coco.h')
    expand_file('build/r/skel/DESCRIPTION.in', 'build/r/skel/DESCRIPTION',
                {'COCO_VERSION': hg_version()})

def test_r():
    build_r()
    pass

################################################################################
## Global
def build():
    build_c()
    build_python2()
    build_python3()
    build_r()

def test():
    test_c()
    test_python2()
    test_python3()
    test_r()

def help():
    print("""COCO framework bootstrap tool.

Usage: do.py <command>

Available commands:

  build        - Build C, Python and R modules
  test         - Test C, Python and R modules
  build-c      - Build C framework
  build-python2 - Build Python 2 modules
  build-python3 - Build Python 3 modules
  build-r       - Build R package
  test-c        - Run minimal test of C components
  test-python2  - Run minimal test of Python 2 module
  test-python3  - Run minimal test of Python 3 module
  test-r  - Run minimal test of R package
""")

def main(args):
    if len(args) != 1:
        help()
        sys.exit(0)
    cmd = args[0]
    if cmd == 'build-c': build_c()
    elif cmd == 'test-c': test_c()
    elif cmd == 'build-python2': build_python()
    elif cmd == 'test-python2': test_python2()
    elif cmd == 'build-python3': build_python()
    elif cmd == 'test-python3': test_python3()
    elif cmd == 'build-r': build_r()
    elif cmd == 'test-r': test_r()
    elif cmd == 'build': build()
    elif cmd == 'test': test()
    else: help()

if __name__ == '__main__':
    main(sys.argv[1:])
