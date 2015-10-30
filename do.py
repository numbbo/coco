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
from subprocess import call

## Change to the root directory of repository and add our tools/
## subdirectory to system wide search path for modules.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('tools'))

from amalgamate import amalgamate
from cocoutils import make, run, python, rscript
from cocoutils import copy_file, copy_tree, expand_file, write_file
from cocoutils import git_version, git_revision

core_files = ['src/coco_suites.c',
              'src/coco_random.c',
              'src/coco_generics.c'
              ]

################################################################################
## Examples
def build_examples():
    build_c()
    copy_file('build/c/coco.c', 'examples/bbob2009-c-cmaes/coco.c')
    copy_file('build/c/coco.h', 'examples/bbob2009-c-cmaes/coco.h')

################################################################################
## C
def build_c():
    global release
    amalgamate(core_files + ['src/coco_runtime_c.c'],  'build/c/coco.c', release)
    copy_file('src/coco.h', 'build/c/coco.h')
    copy_file('src/bbob2009_testcases.txt', 'build/c/bbob2009_testcases.txt')
    write_file(git_revision(), "build/c/REVISION")
    write_file(git_version(), "build/c/VERSION")
    make("build/c", "clean")
    make("build/c", "all")

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
    valgrind_cmd = ['valgrind', '--error-exitcode=1', '--track-origins=yes',
                    '--leak-check=full', '--show-reachable=yes',
                    './coco_test', 'bbob2009_testcases.txt']
    run('build/c', valgrind_cmd)
    
################################################################################
## Python 2
def _prep_python():
    global release
    amalgamate(core_files + ['src/coco_runtime_c.c'],  'build/python/cython/coco.c', 
               release)
    copy_file('src/coco.h', 'build/python/cython/coco.h')
    copy_file('src/bbob2009_testcases.txt', 'build/python/bbob2009_testcases.txt')
    expand_file('build/python/README.in', 'build/python/README',
                {'COCO_VERSION': git_version()}) # hg_version()})
    expand_file('build/python/setup.py.in', 'build/python/setup.py',
                {'COCO_VERSION': git_version()}) # hg_version()})

def build_python():
    _prep_python()
    ## Force distutils to use Cython
    os.environ['USE_CYTHON'] = 'true'
    # python('build/python', ['setup.py', 'sdist'])
    python('build/python', ['setup.py', 'install', '--user'])
    os.environ.pop('USE_CYTHON')

def run_python(script_filename):
    _prep_python()
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
        python('.', [script_filename])
        os.environ.pop('USE_CYTHON')
        os.environ.pop('PYTHONPATH')
    except subprocess.CalledProcessError:
        sys.exit(-1)
    finally:
        shutil.rmtree(python_temp_home)

def test_python():
    _prep_python()
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
        sys.exit(-1)
    finally:
        shutil.rmtree(python_temp_home)

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
    os.environ['PYTHON'] = 'python3'
    build_python()
    os.environ.pop('PYTHON')

def test_python3():
    os.environ['PYTHON'] = 'python3'
    test_python()
    os.environ.pop('PYTHON')

################################################################################
## R
def build_r():
    global release
    copy_tree('build/r/skel', 'build/r/pkg')
    amalgamate(core_files + ['src/coco_runtime_r.c'],  'build/r/pkg/src/coco.c',
               release)
    copy_file('src/coco.h', 'build/r/pkg/src/coco.h')
    expand_file('build/r/pkg/DESCRIPTION.in', 'build/r/pkg/DESCRIPTION',
                {'COCO_VERSION': hg_version()})  # FIXME: it seems that this doesn't work, because it writes '0.0.789' instead of 0.0.789
    rscript('build/r/', ['tools/roxygenize'])
    rusn('build/r', ['R', 'CMD', 'build', 'pkg'])

def test_r():
    build_r()
    run('build/r', ['R', 'CMD', 'check', 'pkg'])
    pass

################################################################################
## Matlab
def build_matlab():
    global release
    amalgamate(core_files + ['src/coco_runtime_r.c'],  'build/matlab/coco.c', release)
    copy_file('src/coco.h', 'build/matlab/coco.h')
    write_file(hg_revision(), "build/matlab/REVISION")
    write_file(hg_version(), "build/matlab/VERSION")
    run('build/matlab', ['matlab', '-nodisplay', '-nosplash', '-r', 'setup, exit'])
    
################################################################################
## Java
def build_java():
    global release
    amalgamate(core_files + ['src/coco_runtime_c.c'],  'build/java/coco.c', release)
    copy_file('src/coco.h', 'build/java/coco.h')
    write_file(git_revision(), "build/java/REVISION")
    write_file(git_version(), "build/java/VERSION")
    run('build/java', ['javac', 'JNIinterface.java'])
    run('build/java', ['javah', 'JNIinterface'])
    run('build/java', ['gcc', '-I/System/Library/Frameworks/JavaVM.framework/Headers',
                       '-c', 'JNIinterface.c'])
    run('build/java', ['gcc', '-dynamiclib', '-o', 'libJNIinterface.jnilib',
                       'JNIinterface.o'])
    run('build/java', ['javac', 'Problem.java'])
    run('build/java', ['javac', 'Benchmark.java'])
    run('build/java', ['javac', 'demo.java'])

################################################################################
## multiobjective Coco
def build_c_mo():  # added for the multiobjective case
    global release
    amalgamate(core_files + ['src/coco_runtime_c.c'],  'build/c/mo/coco.c', release)
    copy_file('src/coco.h', 'build/c/mo/coco.h')
    # copy_file('src/bbob2009_testcases.txt', 'build/c/bbob2009_testcases.txt')
    write_file(git_revision(), "build/c/mo/REVISION")
    write_file(git_version(), "build/c/mo/VERSION")
    make("build/c/mo", "clean")
    make("build/c/mo", "all")

def test_c_mo():
    build_c_mo()
    try:
        run('build/c/mo', ['./demo_mo'])
    except subprocess.CalledProcessError:
        sys.exit(-1)



################################################################################
## Global
def build():
    builders = [
        build_c,
        build_java, 
        build_matlab,
        # build_octave, 
        build_python,
        build_r,
        build_examples,
        build_c_mo
    ]
    for builder in builders:
        try:
            builder()
        except:
            failed = str(builder)
            print("============")
            print('   ERROR: %s failed, call "./do.py %s" individually'
                    % (failed, failed[failed.find('build_'):].split()[0]) +
                  ' for a more detailed error report')
            print("============")

def test():
    test_c()
    test_python()
    test_r()
    test_c_mo()

def help():
    print("""COCO framework bootstrap tool.

Usage: do.py <command> <arguments>

Available commands:

  build          - Build C, Python and R modules
  test           - Test C, Python and R modules
  build-c        - Build C framework
  build-python   - Build Python modules
  build-python2  - Build Python 2 modules
  build-python3  - Build Python 3 modules
  build-r        - Build R package
  build-matlab   - Build Matlab package
  build-examples - Update examples to latest framework code
  build-java     - Build Java package
  run-python     - Run a Python script with installed COCO module
                   Takes a single argument (name of Python script file)
  test-c         - Run minimal test of C components
  test-python    - Run minimal test of Python module
  test-python2   - Run minimal test of Python 2 module
  test-python3   - Run minimal test of Python 3 module
  test-r         - Run minimal test of R package
  leak-check     - Check for memory leaks

  build-c-mo     - Build multiobjective Coco in C
  test-c-mo      - Test multiobjective Coco in C

To build a release version which does not include debugging information in the 
amalgamations set the environment variable COCO_RELEASE to 'true'.
""")

def main(args):
    if len(args) < 1:
        help()
        sys.exit(0)
    cmd = args[0].replace('_', '-')
    if cmd == 'build-c': build_c()
    elif cmd == 'test-c': test_c()
    elif cmd == 'build-python': build_python()
    elif cmd == 'run-python': run_python(args[1])
    elif cmd == 'test-python': test_python()
    elif cmd == 'build-python2': build_python2()
    elif cmd == 'test-python2': test_python2()
    elif cmd == 'build-python3': build_python3()
    elif cmd == 'test-python3': test_python3()
    elif cmd == 'build-r': build_r()
    elif cmd == 'build-matlab': build_matlab()
    elif cmd == 'build-java': build_java()
    elif cmd == 'test-r': test_r()
    elif cmd == 'build-examples': build_examples()
    elif cmd == 'build': build()
    elif cmd == 'test': test()
    elif cmd == 'leak-check': leak_check()
    elif cmd == 'build-c-mo': build_c_mo() # added for the multiobjective case
    elif cmd == 'test-c-mo': test_c_mo()   # added for the multiobjective case
    else: help()

if __name__ == '__main__':
    release = os.getenv('COCO_RELEASE', 'false') == 'true'
    main(sys.argv[1:])
