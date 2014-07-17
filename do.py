#!/usr/bin/env python2.7
## -*- mode: python -*-

import sys
import re
import os

## Change to the root directory of repository and add our tools/
## subdirectory to system wide search path for modules.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('tools'))

from amalgamate import amalgamate
from platform import make, run, python27
from platform import copy_file, expand_file, write_file
from platform import hg_version, hg_revision

core_files = ['src/coco_benchmark.c', 'src/coco_random.c', 
              'src/coco_generics.c', 'src/coco_c_runtime.c']

################################################################################
## C
def build_c():
    amalgamate(core_files + ['src/coco_c_runtime.c'],  'build/c/coco.c')
    copy_file('src/coco.h', 'build/c/coco.h')
    write_file(hg_revision(), "build/c/REVISION")
    write_file(hg_version(), "build/c/VERSION")

def test_c():
    build_c()
    make("build/c", "clean")
    make("build/c", "all")
    run('build/c', ['./test_bbob2009'])

################################################################################
## Python
def build_python():
    amalgamate(core_files + ['src/coco_c_runtime.c'],  'build/python/coco/coco.c')
    copy_file('src/coco.h', 'build/python/coco/coco.h')
    expand_file('build/python/README.in', 'build/python/README',
                {'COCO_VERSION': hg_version()})
    expand_file('build/python/setup.py.in', 'build/python/setup.py',
                {'COCO_VERSION': hg_version()})
    python27('build/python', ['setup.py', 'build'])
    python27('build/python', ['setup.py', 'bdist'])
    python27('build/python', ['setup.py', 'sdist'])

def test_python():
    build_python()
    python27('build/python', ['setup.py', 'check', '--metadata', '--strict'])
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

def release_c():
    build_c()

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
    print """COCO framework bootstrap tool.

Usage: do.py <command>

Available commands:

  build_c - Build C framework
  test_c  - Run minimal test of C components
  build   - Build C, Python and R modules
"""

def main(args):
    if len(args) != 1: 
        help()
        sys.exit(0)
    cmd = args[0]
    if cmd == 'build_c': build_c()
    elif cmd == 'test_c': test_c()
    elif cmd == 'build_python': build_python()
    elif cmd == 'test_python': test_python()
    elif cmd == 'build_r': build_r()
    elif cmd == 'test_r': test_r()
    elif cmd == 'build': build()
    elif cmd == 'test': test()
    else: help()

if __name__ == '__main__':
    main(sys.argv[1:])
