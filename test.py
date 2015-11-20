#!/usr/bin/env python
## -*- mode: python -*- 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import subprocess
import platform

## Change to the root directory of repository and add our tools/
## subdirectory to system wide search path for modules.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('code-experiments/tools'))

from amalgamate import amalgamate
from cocoutils import make, run
from cocoutils import copy_file, write_file
from cocoutils import git_version, git_revision

core_files = ['code-experiments/src/coco_suites.c',
              'code-experiments/src/coco_random.c',
              'code-experiments/src/coco_generics.c'
              ]

################################################################################
## Unit tests
def build_unit_test():
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/test/unit-test/coco.c', release)

    libraryPath = '';
    fileName = ''
    if ('win32' in sys.platform):
        fileName = 'cmocka.dll'
        if '64' in platform.machine():
            libraryPath = 'code-experiments/test/unit-test/lib/win64'
        elif ('32' in platform.machine()) or ('x86' in platform.machine()):
            if 'cygwin' in os.environ['PATH']:
                libraryPath = 'code-experiments/test/unit-test/lib/win32_cygwin'
            else:
                libraryPath = 'code-experiments/test/unit-test/lib/win32_mingw'
    elif ('linux' in sys.platform):
        libraryPath = 'code-experiments/test/unit-test/lib/linux'
        fileName = 'libcmocka.so'
        
    if (len(libraryPath) > 0):
        copy_file(os.path.join(libraryPath, fileName), 
                  os.path.join('code-experiments/test/unit-test', fileName))
        
    copy_file('code-experiments/src/coco.h', 'code-experiments/test/unit-test/coco.h')
    write_file(git_revision(), "code-experiments/test/unit-test/REVISION")
    write_file(git_version(), "code-experiments/test/unit-test/VERSION")
    make("code-experiments/test/unit-test", "clean")
    make("code-experiments/test/unit-test", "all")

def test_unit_test():
    build_unit_test()
    try:
        run('code-experiments/test/unit-test', ['./unit_test'])
    except subprocess.CalledProcessError:
        sys.exit(-1)
    
################################################################################
## Global
def build():
    builders = [
        build_unit_test
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
    test_unit_test()

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
