#!/usr/bin/env python
## -*- mode: python -*- 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
import tempfile
import subprocess
import platform
from subprocess import check_output, STDOUT

## Change to the root directory of repository and add our tools/
## subdirectory to system wide search path for modules.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('code-experiments/tools'))

from amalgamate import amalgamate
from cocoutils import make, run, python, rscript
from cocoutils import copy_file, copy_tree, expand_file, write_file
from cocoutils import git_version, git_revision

core_files = ['code-experiments/src/coco_suites.c',
              'code-experiments/src/coco_random.c',
              'code-experiments/src/coco_generics.c'
              ]

################################################################################
## Unit tests
def build_unittest():
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/test/coco.c', release)
    copy_file('code-experiments/test/lib/cmocka.dll', 'code-experiments/test/cmocka.dll')
    copy_file('code-experiments/test/lib/libcmocka.a', 'code-experiments/test/libcmocka.a')
    copy_file('code-experiments/src/coco.h', 'code-experiments/test/coco.h')
    write_file(git_revision(), "code-experiments/test/REVISION")
    write_file(git_version(), "code-experiments/test/VERSION")
    make("code-experiments/test", "clean")
    make("code-experiments/test", "all")

def test_unittest():
    build_unittest()
    try:
        run('code-experiments/test', ['test'])
    except subprocess.CalledProcessError:
        sys.exit(-1)
    
################################################################################
## Global
def build():
    builders = [
        build_unittest
    ]
    for builder in builders:
        try:
            builder()
        except:
            failed = str(builder)
            print("============")
            print('   ERROR: %s failed, call "./test.py %s" individually'
                    % (failed, failed[failed.find('build_'):].split()[0]) +
                  ' for a more detailed error report')
            print("============")

def test():
    test_unittest()

def help():
    print("""COCO framework testing tool.

Usage: test.py <command> <arguments>

Available commands:

  build          - Build unit tests
  test           - Test unit tests

To build a release version which does not include debugging information in the 
amalgamations set the environment variable COCO_RELEASE to 'true'.
""")

def main(args):
    if len(args) < 1:
        help()
        sys.exit(0)
    cmd = args[0].replace('_', '-')
    if cmd == 'build': build()
    elif cmd == 'test': test()
    else: help()

if __name__ == '__main__':
    release = os.getenv('COCO_RELEASE', 'false') == 'true'
    main(sys.argv[1:])
