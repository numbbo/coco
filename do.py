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
import subprocess

## Change to the root directory of repository and add our tools/
## subdirectory to system wide search path for modules.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('tools'))

from amalgamate import amalgamate
from cocoutils import make, run, python, rscript
from cocoutils import copy_file, copy_tree, expand_file, write_file
from cocoutils import hg_version, hg_revision

core_files = ['src/coco_benchmark.c', 'src/coco_random.c', 'src/coco_generics.c']

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
    try:
        make("build/c", "clean")
        make("build/c", "all")
        run('build/c', ['./coco_test', 'bbob2009_testcases.txt'])
    except subprocess.CalledProcessError:
        sys.exit(-1)

def leak_check():
    build_c()
    os.environ['CFLAGS'] = '-g -Os'
    make("build/c", "clean")
    make("build/c", "all")
    valgrind_cmd = ['valgrind', '--track-origins=yes',
                    '--leak-check=full', '--error-exitcode=1',
                    './coco_test', 'bbob2009_testcases.txt']
    run('build/c', valgrind_cmd)
    
################################################################################
## Python 2
def build_python():
    amalgamate(core_files + ['src/coco_c_runtime.c'],  'build/python/cython/coco.c')
    copy_file('src/coco.h', 'build/python/cython/coco.h')
    copy_file('src/bbob2009_testcases.txt', 'build/python/bbob2009_testcases.txt')
    expand_file('build/python/README.in', 'build/python/README',
                {'COCO_VERSION': hg_version()})
    expand_file('build/python/setup.py.in', 'build/python/setup.py',
                {'COCO_VERSION': hg_version()})
    ## Force distutils to use Cython
    os.environ['USE_CYTHON'] = 'true'
    python('build/python', ['setup.py', 'sdist'])
    os.environ.pop('USE_CYTHON')

def test_python():
    build_python()
    python('build/python', ['setup.py', 'check', '--metadata', '--strict'])
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
        python('build/python', ['setup.py', 'install', '--home', python_temp_home])
        python('build/python', ['coco_test.py', 'bbob2009_testcases.txt'])
        os.environ.pop('USE_CYTHON')
        os.environ.pop('PYTHONPATH')
    except subprocess.CalledProcessError:
        pass
    finally:
        shutil.rmtree(python_temp_home)
        pass

################################################################################
## Python 2
def build_python2():
    os.environ['PYTHON'] = 'python2.7'
    build_python()
    os.environ.pop('PYTHON')

def test_python2():
    os.environ['PYTHON'] = 'python2.7'
    test_python()
    os.environ.pop('PYTHON')

################################################################################
## Python 3
def build_python3():
    os.environ['PYTHON'] = 'python3.4'
    build_python()
    os.environ.pop('PYTHON')

def test_python3():
    os.environ['PYTHON'] = 'python3.4'
    test_python()
    os.environ.pop('PYTHON')

################################################################################
## R
def build_r():
    copy_tree('build/r/skel', 'build/r/pkg')
    amalgamate(core_files + ['src/coco_r_runtime.c'],  'build/r/pkg/src/coco.c')
    copy_file('src/coco.h', 'build/r/pkg/src/coco.h')
    expand_file('build/r/pkg/DESCRIPTION.in', 'build/r/pkg/DESCRIPTION',
                {'COCO_VERSION': hg_version()})
    rscript('build/r/', ['tools/roxygenize'])
    run('build/r', ['R', 'CMD', 'build', 'pkg'])

def test_r():
    build_r()
    run('build/r', ['R', 'CMD', 'check', 'pkg'])
    pass

################################################################################
## Global
def build():
    build_c()
    build_python()
    build_r()

def test():
    test_c()
    test_python()
    test_r()

def help():
    print("""COCO framework bootstrap tool.

Usage: do.py <command>

Available commands:

  build        - Build C, Python and R modules
  test         - Test C, Python and R modules
  build-c      - Build C framework
  build-python - Build Python 2 modules
  build-r      - Build R package
  test-c       - Run minimal test of C components
  test-python  - Run minimal test of Python 2 module
  test-r       - Run minimal test of R package
  leak-check   - Check for memory leaks
""")

def main(args):
    if len(args) != 1:
        help()
        sys.exit(0)
    cmd = args[0]
    if cmd == 'build-c': build_c()
    elif cmd == 'test-c': test_c()
    elif cmd == 'build-python': build_python()
    elif cmd == 'test-python': test_python()
    elif cmd == 'build-r': build_r()
    elif cmd == 'test-r': test_r()
    elif cmd == 'build': build()
    elif cmd == 'test': test()
    elif cmd == 'leak-check': leak_check()
    else: help()

if __name__ == '__main__':
    main(sys.argv[1:])
