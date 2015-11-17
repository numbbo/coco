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
import platform
from subprocess import call, check_output, STDOUT

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
## Examples
def build_examples():
    build_c()
    copy_file('code-experiments/build/c/coco.c', 'code-experiments/examples/bbob2009-c-cmaes/coco.c')
    copy_file('code-experiments/build/c/coco.h', 'code-experiments/examples/bbob2009-c-cmaes/coco.h')

#########################################################ex#######################
## C
def build_c():
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/build/c/coco.c', release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/c/coco.h')
    copy_file('code-experiments/src/bbob2009_testcases.txt', 'code-experiments/build/c/bbob2009_testcases.txt')
    write_file(git_revision(), "code-experiments/build/c/REVISION")
    write_file(git_version(), "code-experiments/build/c/VERSION")
    make("code-experiments/build/c", "clean")
    make("code-experiments/build/c", "all")

def test_c():
    build_c()
    try:
        make("code-experiments/build/c", "clean")
        make("code-experiments/build/c", "all")
        run('code-experiments/build/c', ['./coco_test', 'bbob2009_testcases.txt'])
    except subprocess.CalledProcessError:
        sys.exit(-1)

def leak_check():
    build_c()
    os.environ['CFLAGS'] = '-g -Os'
    make("code-experiments/build/c", "clean")
    make("code-experiments/build/c", "all")
    valgrind_cmd = ['valgrind', '--error-exitcode=1', '--track-origins=yes',
                    '--leak-check=full', '--show-reachable=yes',
                    './coco_test', 'bbob2009_testcases.txt']
    run('code-experiments/build/c', valgrind_cmd)
    
## C - multiobjective case
def build_c_mo():
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/build/c/mo/coco.c', release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/c/mo/coco.h')
    write_file(git_revision(), "code-experiments/build/c/mo/REVISION")
    write_file(git_version(), "code-experiments/build/c/mo/VERSION")
    make("code-experiments/build/c/mo", "clean")
    make("code-experiments/build/c/mo", "all")

def test_c_mo():
    build_c_mo()
    try:
        run('code-experiments/build/c/mo', ['./demo_mo', 'test'])
    except subprocess.CalledProcessError:
        sys.exit(-1)
    
################################################################################
## Python 2
def _prep_python():
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/build/python/cython/coco.c', 
               release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/python/cython/coco.h')
    copy_file('code-experiments/src/bbob2009_testcases.txt', 'code-experiments/build/python/bbob2009_testcases.txt')
    expand_file('code-experiments/build/python/README.in', 'code-experiments/build/python/README',
                {'COCO_VERSION': git_version()}) # hg_version()})
    expand_file('code-experiments/build/python/setup.py.in', 'code-experiments/build/python/setup.py',
                {'COCO_VERSION': git_version()}) # hg_version()})

def build_python():
    _prep_python()
    ## Force distutils to use Cython
    os.environ['USE_CYTHON'] = 'true'
    # python('code-experiments/build/python', ['setup.py', 'sdist'])
    python('code-experiments/build/python', ['setup.py', 'install', '--user'])
    os.environ.pop('USE_CYTHON')

def run_python(script_filename):
    _prep_python()
    python('code-experiments/build/python', ['setup.py', 'check', '--metadata', '--strict'])
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
        python('code-experiments/build/python', ['setup.py', 'install', '--home', python_temp_home])
        python('.', [script_filename])
        os.environ.pop('USE_CYTHON')
        os.environ.pop('PYTHONPATH')
    except subprocess.CalledProcessError:
        sys.exit(-1)
    finally:
        shutil.rmtree(python_temp_home)

def test_python():
    _prep_python()
    python('code-experiments/build/python', ['setup.py', 'check', '--metadata', '--strict'])
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
        python('code-experiments/build/python', ['setup.py', 'install', '--home', python_temp_home])
        python('code-experiments/build/python', ['coco_test.py', 'bbob2009_testcases.txt'])
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
    copy_tree('code-experiments/build/r/skel', 'code-experiments/build/r/pkg')
    amalgamate(core_files + ['code-experiments/src/coco_runtime_r.c'],  'code-experiments/build/r/pkg/src/coco.c',
               release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/r/pkg/src/coco.h')
    expand_file('code-experiments/build/r/pkg/DESCRIPTION.in', 'code-experiments/build/r/pkg/DESCRIPTION',
                {'COCO_VERSION': git_version()})  # FIXME: it seems that this doesn't work, because it writes '0.0.789' instead of 0.0.789
    rscript('code-experiments/build/r/', ['code-experiments/tools/roxygenize'])
    rusn('code-experiments/build/r', ['R', 'CMD', 'build', 'pkg'])

def test_r():
    build_r()
    run('code-experiments/build/r', ['R', 'CMD', 'check', 'pkg'])
    pass

################################################################################
## Matlab
def build_matlab():
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_r.c'],  'code-experiments/build/matlab/coco.c', release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/matlab/coco.h')
    write_file(git_revision(), "code-experiments/build/matlab/REVISION")
    write_file(git_version(), "code-experiments/build/matlab/VERSION")
    run('code-experiments/build/matlab', ['matlab', '-nodisplay', '-nosplash', '-r', 'setup, exit'])
    
################################################################################
## Java
def build_java():
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/build/java/coco.c', release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/java/coco.h')
    write_file(git_revision(), "code-experiments/build/java/REVISION")
    write_file(git_version(), "code-experiments/build/java/VERSION")
    run('code-experiments/build/java', ['javac', 'JNIinterface.java'])
    run('code-experiments/build/java', ['javah', 'JNIinterface'])
    
    # Finds the path to the headers jni.h and jni_md.h under (platform-dependent)
    # and compiles the JNIinterface library (compiler-dependent). So far, only
    # the following cases are covered:
    
    # 1. Windows with Cygwin (both 64-bit)
    # Note that 'win32' stands for both Windows 32-bit and 64-bit.
    # Since platform 'cygwin' does not work as expected, we need to look for it in the PATH.
    if ('win32' in sys.platform) and ('cygwin' in os.environ['PATH']):
        jdkpath = check_output(['where', 'javac'], stderr = STDOUT, 
                               env = os.environ, universal_newlines = True)  
        jdkpath1 = jdkpath.split("bin")[0] + 'include'
        jdkpath2 = jdkpath1 + '\\win32'
        
        if ('64' in platform.machine()):
            run('code-experiments/build/java', ['x86_64-w64-mingw32-gcc', '-I', jdkpath1, '-I', 
                               jdkpath2, '-shared', '-o', 'JNIinterface.dll', 
                               'JNIinterface.c'])
    
    # 2. Windows with Cygwin (both 32-bit)
        elif ('32' in platform.machine()) or ('x86' in platform.machine()):
            run('code-experiments/build/java', ['i686-w64-mingw32-gcc', '-Wl,--kill-at', '-I', 
                               jdkpath1, '-I', jdkpath2, '-shared', '-o', 
                               'JNIinterface.dll', 'JNIinterface.c'])
                               
    # 3. Windows without Cygwin
    elif ('win32' in sys.platform) and ('cygwin' not in os.environ['PATH']):
        jdkpath = check_output(['where', 'javac'], stderr = STDOUT, 
                               env = os.environ, universal_newlines = True)  
        jdkpath1 = jdkpath.split("bin")[0] + 'include'
        jdkpath2 = jdkpath1 + '\\win32'
        run('code-experiments/build/java', ['gcc', '-Wl,--kill-at', '-I', jdkpath1, '-I', jdkpath2, 
                           '-shared', '-o', 'JNIinterface.dll', 'JNIinterface.c'])
                           
    # 4. Linux
    elif ('linux' in sys.platform):
        jdkpath = check_output(['locate', 'jni.h'], stderr = STDOUT, 
                               env = os.environ, universal_newlines = True)   
        jdkpath1 = jdkpath.split("jni.h")[0]
        jdkpath2 = jdkpath1 + '/linux'
        run('code-experiments/build/java', ['gcc', '-I', jdkpath1, '-I', jdkpath2, '-c', 
                           'JNIinterface.c'])
        run('code-experiments/build/java', ['gcc', '-I', jdkpath1, '-I', jdkpath2, '-o', 
                           'libJNIinterface.so', '-shared', 'JNIinterface.c'])
                           
    # 5. Mac
    elif ('darwin' in sys.platform):
        jdkpath = '/System/Library/Frameworks/JavaVM.framework/Headers'
        run('code-experiments/build/java', ['gcc', '-I', jdkpath, '-c', 'JNIinterface.c'])
        run('code-experiments/build/java', ['gcc', '-dynamiclib', '-o', 'libJNIinterface.jnilib',
                           'JNIinterface.o'])
    
    run('code-experiments/build/java', ['javac', 'Problem.java'])
    run('code-experiments/build/java', ['javac', 'Benchmark.java'])
    run('code-experiments/build/java', ['javac', 'demo.java'])

def test_java():
    build_java()
    try:
        run('code-experiments/build/java', ['java', '-Djava.library.path=.', 'demo'])    
    except subprocess.CalledProcessError:
        sys.exit(-1)

################################################################################
## Global
def build():
    builders = [
        build_c,
        build_c_mo,
        build_matlab,
        # build_octave, 
        build_python,
        build_r,
        build_java, 
        build_examples
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
    test_c_mo()
    test_python()
    test_r()
    test_java()

def help():
    print("""COCO framework bootstrap tool.

Usage: do.py <command> <arguments>

Available commands:

  build          - Build C, Python and R modules
  test           - Test C, Python and R modules
  build-c        - Build C framework
  build-c-mo     - Build multiobjective C framework
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
  test-c-mo      - Run minimal test of multiobjective C components
  test-python    - Run minimal test of Python module
  test-python2   - Run minimal test of Python 2 module
  test-python3   - Run minimal test of Python 3 module
  test-r         - Run minimal test of R package
  test-java      - Run minimal test of Java package
  leak-check     - Check for memory leaks


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
    elif cmd == 'build-c-mo': build_c_mo() 
    elif cmd == 'test-c-mo': test_c_mo()   
    elif cmd == 'build-python': build_python()
    elif cmd == 'run-python': run_python(args[1])
    elif cmd == 'test-python': test_python()
    elif cmd == 'build-python2': build_python2()
    elif cmd == 'test-python2': test_python2()
    elif cmd == 'build-python3': build_python3()
    elif cmd == 'test-python3': test_python3()
    elif cmd == 'build-matlab': build_matlab()
    elif cmd == 'build-r': build_r()
    elif cmd == 'test-r': test_r()
    elif cmd == 'build-java': build_java()
    elif cmd == 'test-java': test_java()
    elif cmd == 'build-examples': build_examples()
    elif cmd == 'build': build()
    elif cmd == 'test': test()
    elif cmd == 'leak-check': leak_check()
    else: help()

if __name__ == '__main__':
    release = os.getenv('COCO_RELEASE', 'false') == 'true'
    main(sys.argv[1:])
