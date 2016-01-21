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
import time
from subprocess import STDOUT
import glob
from os.path import join

## Change to the root directory of repository and add our tools/
## subdirectory to system wide search path for modules.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(join('code-experiments', 'tools')))

from amalgamate import amalgamate
from cocoutils import make, run, python, check_output
from cocoutils import copy_file, expand_file, write_file
from cocoutils import git_version, git_revision

core_files = ['code-experiments/src/coco_generics.c',
              'code-experiments/src/coco_random.c',
              'code-experiments/src/coco_suite.c',
              'code-experiments/src/coco_observer.c'
              ]

################################################################################
## C
def build_c():
    """ Builds the C source code """
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/build/c/coco.c', release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/c/coco.h')
    copy_file('code-experiments/build/c/coco.c', 'code-experiments/examples/bbob2009-c-cmaes/coco.c')
    copy_file('code-experiments/build/c/coco.h', 'code-experiments/examples/bbob2009-c-cmaes/coco.h')
    write_file(git_revision(), "code-experiments/build/c/REVISION")
    write_file(git_version(), "code-experiments/build/c/VERSION")
    make("code-experiments/build/c", "clean")
    make("code-experiments/build/c", "all")

def run_c():
    """ Builds and runs the example experiment in C """
    build_c()
    try:
        run('code-experiments/build/c', ['./example_experiment'])
    except subprocess.CalledProcessError:
        sys.exit(-1)

def test_c():
    """ Builds and runs unit tests, integration tests and an example experiment test in C """
    build_c()
    # Perform unit tests
    build_c_unit_tests()
    run_c_unit_tests()
    # Perform integration tests
    build_c_integration_tests()
    run_c_integration_tests()
    # Perform example experiment tests
    build_c_example_tests()
    run_c_example_tests()
        
def test_c_unit():
    """ Builds and runs unit tests in C """
    build_c()
    # Perform unit tests
    build_c_unit_tests()
    run_c_unit_tests()
        
def test_c_integration():
    """ Builds and runs integration tests in C """
    build_c()
    # Perform integration tests
    build_c_integration_tests()
    run_c_integration_tests()
        
def test_c_example():
    """ Builds and runs an example experiment test in C """
    build_c()
    # Perform example tests
    build_c_example_tests()
    run_c_example_tests()
        
def build_c_unit_tests():
    """ Builds unit tests in C """
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
        fileName = 'libcmocka.so'
        if 'Ubuntu' in platform.linux_distribution():
            libraryPath = 'code-experiments/test/unit-test/lib/linux_ubuntu'
        elif 'Fedora' in platform.linux_distribution():
            libraryPath = 'code-experiments/test/unit-test/lib/linux_fedora'
    elif ('darwin' in sys.platform): #Mac
        libraryPath = 'code-experiments/test/unit-test/lib/macosx'
        fileName = 'libcmocka.dylib'
        
    if (len(libraryPath) > 0):
        copy_file(os.path.join(libraryPath, fileName), 
                  os.path.join('code-experiments/test/unit-test', fileName))        
    copy_file('code-experiments/build/c/coco.c', 'code-experiments/test/unit-test/coco.c')
    copy_file('code-experiments/src/coco.h', 'code-experiments/test/unit-test/coco.h')
    make("code-experiments/test/unit-test", "clean")
    make("code-experiments/test/unit-test", "all")

def run_c_unit_tests():
    """ Runs unit tests in C """
    try:
        run('code-experiments/test/unit-test', ['./unit_test'])
    except subprocess.CalledProcessError:
        sys.exit(-1)

def build_c_integration_tests():  
    """ Builds integration tests in C """ 
    copy_file('code-experiments/build/c/coco.c', 'code-experiments/test/integration-test/coco.c')
    copy_file('code-experiments/src/coco.h', 'code-experiments/test/integration-test/coco.h')
    copy_file('code-experiments/src/bbob2009_testcases.txt', 'code-experiments/test/integration-test/bbob2009_testcases.txt')
    make("code-experiments/test/integration-test", "clean")
    make("code-experiments/test/integration-test", "all")

def run_c_integration_tests():
    """ Runs integration tests in C """
    try:
        run('code-experiments/test/integration-test', ['./test_coco', 'bbob2009_testcases.txt'])
        run('code-experiments/test/integration-test', ['./test_instance_extraction'])
        run('code-experiments/test/integration-test', ['./test_biobj'])
    except subprocess.CalledProcessError:
        sys.exit(-1)
    
def build_c_example_tests():
    """ Builds an example experiment test in C """
    if os.path.exists('code-experiments/test/example-test'):
        shutil.rmtree('code-experiments/test/example-test')
        time.sleep(1) # Needed to avoid permission errors for os.makedirs
    os.makedirs('code-experiments/test/example-test') 
    copy_file('code-experiments/build/c/coco.c', 'code-experiments/test/example-test/coco.c')
    copy_file('code-experiments/src/coco.h', 'code-experiments/test/example-test/coco.h')
    copy_file('code-experiments/build/c/example_experiment.c', 'code-experiments/test/example-test/example_experiment.c')
    copy_file('code-experiments/build/c/Makefile.in', 'code-experiments/test/example-test/Makefile.in')
    copy_file('code-experiments/build/c/Makefile_win_gcc.in', 'code-experiments/test/example-test/Makefile_win_gcc.in')
    make("code-experiments/test/example-test", "clean")
    make("code-experiments/test/example-test", "all")
        
def run_c_example_tests():
    """ Runs an example experiment test in C """
    try:
        run('code-experiments/test/example-test', ['./example_experiment'])
    except subprocess.CalledProcessError:
        sys.exit(-1)

def leak_check():
    """ Performs a leak check in C """
    build_c()
    build_c_integration_tests()
    os.environ['CFLAGS'] = '-g -Os'
    valgrind_cmd = ['valgrind', '--error-exitcode=1', '--track-origins=yes',
                    '--leak-check=full', '--show-reachable=yes',
                    './test_coco', 'bbob2009_testcases.txt']
    run('code-experiments/test/integration-test', valgrind_cmd)
    valgrind_cmd = ['valgrind', '--error-exitcode=1', '--track-origins=yes',
                    '--leak-check=full', '--show-reachable=yes',
                    './test_biobj', 'leak_check']
    run('code-experiments/test/integration-test', valgrind_cmd)
    
################################################################################
## Python 2
def install_postprocessing():
    global release
    expand_file(join('code-postprocessing', 'setup.py.in'),
                join('code-postprocessing', 'setup.py'),
                {'COCO_VERSION': git_version()})
    # copy_tree('code-postprocessing/latex-templates', 'code-postprocessing/bbob_pproc/latex-templates')
    python('code-postprocessing', ['setup.py', 'install', '--user'])
    
def _prep_python():
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/build/python/cython/coco.c', 
               release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/python/cython/coco.h')
    copy_file('code-experiments/src/bbob2009_testcases.txt', 'code-experiments/build/python/bbob2009_testcases.txt')
    copy_file('code-experiments/build/python/README.md', 'code-experiments/build/python/README.txt')
    expand_file('code-experiments/build/python/setup.py.in', 'code-experiments/build/python/setup.py',
                {'COCO_VERSION': git_version()}) # hg_version()})
    # if 'darwin' in sys.platform:  # a hack to force cythoning
    #     run('code-experiments/build/python/cython', ['cython', 'interface.pyx'])

def build_python():
    _prep_python()
    ## Force distutils to use Cython
    # os.environ['USE_CYTHON'] = 'true'
    # python('code-experiments/build/python', ['setup.py', 'sdist'])
    python('code-experiments/build/python', ['setup.py', 'install', '--user'])
    # os.environ.pop('USE_CYTHON')

def run_python(test=True):
    """ Builds and installs the Python module `cocoex` and runs the
    `example_experiment.py` as a simple test case. """
    build_python()
    try:
        if test:
            run(os.path.join('code-experiments', 'build', 'python'), ['python', 'coco_test.py'])
        run(os.path.join('code-experiments', 'build', 'python'),
            ['python', 'example_experiment.py'])
    except subprocess.CalledProcessError:
        sys.exit(-1)

def run_sandbox_python(directory, script_filename=
                     os.path.join('code-experiments', 'build', 'python',
                                  'example_experiment.py')):
    """run a python script after building and installing `cocoex` in a new
    environment."""
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
        python(directory, [script_filename])
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
## Matlab
def build_matlab():
    """Builds MATLAB example in build/matlab/ but not the one in examples/."""
    
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/build/matlab/coco.c', release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/matlab/coco.h')
    write_file(git_revision(), "code-experiments/build/matlab/REVISION")
    write_file(git_version(), "code-experiments/build/matlab/VERSION")
    run('code-experiments/build/matlab', ['matlab', '-nodisplay', '-nosplash', '-r', 'setup, exit'])

    
def run_matlab():
    # remove the mex files for a clean compilation first
    for filename in glob.glob('code-experiments/build/matlab/*.mex*') :
        os.remove( filename )
    # amalgamate, copy, and build
    build_matlab()
    wait_for_compilation_to_finish('./code-experiments/build/matlab/cocoProblemIsValid')
    # run after compilation finished
    run('code-experiments/build/matlab', ['matlab', '-nodisplay', '-nosplash', '-r', 'exampleexperiment, exit'])

    
def is_compiled(filenameprefix):
    """Returns true iff a file 'filenameprefix.mex*' exists."""
    
    # get all files with the given prefix
    files = glob.glob(filenameprefix + '.*')
    # return true iff one of the files contains 'mex'
    ret = False
    for f in files:
        if '.mex' in f:
            ret = True
    return ret


def wait_for_compilation_to_finish(filenameprefix):
    """Waits until filenameprefix.c is compiled into a mex file.
    
    Needed because under Windows, a MATLAB call is typically non-blocking
    and thus, the experiments would be started before the compilation is over.
    """
    
    print('Wait for compilation to finish', end=''),
    while not is_compiled(filenameprefix):
        time.sleep(2)
        print('.', end='')
    print(' ')


def build_matlab_sms():
    global release
    join = os.path.join
    source_folder = join('code-experiments', 'build', 'matlab')
    destination_folder = join('code-experiments', 'examples',
                              'bbob-biobj-matlab-smsemoa')
    # amalgamate and copy files
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],
               join(destination_folder, 'coco.c'), release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/examples/bbob-biobj-matlab-smsemoa/coco.h')
    write_file(git_revision(), join(destination_folder, "REVISION"))
    write_file(git_version(), join(destination_folder, "VERSION"))
    files = ['cocoEvaluateFunction.c',
             'cocoObserver.c',
             'cocoObserverFree.c',
             'cocoProblemGetDimension.c',
             'cocoProblemGetEvaluations.c',
             'cocoProblemGetId.c',
             'cocoProblemGetLargestValuesOfInterest.c',
             'cocoProblemGetName.c',
             'cocoProblemGetNumberOfObjectives.c',
             'cocoProblemGetSmallestValuesOfInterest.c',
             'cocoProblemIsValid.c',
             'cocoSuite.c',
             'cocoSuiteFree.c',
             'cocoSuiteGetNextProblem.c']
    for file in files:
        copy_file(join(source_folder, file),
                  join(destination_folder, file))
    # compile
    run('code-experiments/examples/bbob-biobj-matlab-smsemoa', ['matlab', '-nodisplay', '-nosplash', '-r', 'setup, exit'])

def run_matlab_sms():
    print('CLEAN\t mex files from code-experiments/build/matlab/')
    # remove the mex files for a clean compilation first
    for filename in glob.glob('code-experiments/examples/bbob-biobj-matlab-smsemoa/*.mex*') :
        os.remove( filename )
    # amalgamate, copy, and build
    build_matlab_sms()
    wait_for_compilation_to_finish('./code-experiments/examples/bbob-biobj-matlab-smsemoa/paretofront')
    # run after compilation finished
    run('code-experiments/examples/bbob-biobj-matlab-smsemoa', ['matlab', '-nodisplay', '-nosplash', '-r', 'run_smsemoa_on_bbob_biobj, exit'])


################################################################################
## Octave
def build_octave():
    """Builds example in build/matlab/ with GNU Octave but not the one in examples/."""
    
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/build/matlab/coco.c', release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/matlab/coco.h')
    write_file(git_revision(), "code-experiments/build/matlab/REVISION")
    write_file(git_version(), "code-experiments/build/matlab/VERSION")
    run('code-experiments/build/matlab', ['octave', '--no-gui', 'setup.m'])

    
def run_octave():
    # remove the mex files for a clean compilation first
    print('CLEAN\t mex files from code-experiments/build/matlab/')
    for filename in glob.glob('code-experiments/build/matlab/*.mex*') :
        os.remove( filename )
    # amalgamate, copy, and build
    build_octave()
    run('code-experiments/build/matlab', ['octave', '--no-gui', 'exampleexperiment.m'])


################################################################################
## Java
def build_java():
    """ Builds the example experiment in Java """
    global release
    amalgamate(core_files + ['code-experiments/src/coco_runtime_c.c'],  'code-experiments/build/java/coco.c', release)
    copy_file('code-experiments/src/coco.h', 'code-experiments/build/java/coco.h')
    write_file(git_revision(), "code-experiments/build/java/REVISION")
    write_file(git_version(), "code-experiments/build/java/VERSION")
    run('code-experiments/build/java', ['javac', 'CocoJNI.java'])
    run('code-experiments/build/java', ['javah', 'CocoJNI'])
    
    # Finds the path to the headers jni.h and jni_md.h (platform-dependent)
    # and compiles the CocoJNI library (compiler-dependent). So far, only
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
                               jdkpath2, '-shared', '-o', 'CocoJNI.dll', 
                               'CocoJNI.c'])
    
    # 2. Windows with Cygwin (both 32-bit)
        elif ('32' in platform.machine()) or ('x86' in platform.machine()):
            run('code-experiments/build/java', ['i686-w64-mingw32-gcc', '-Wl,--kill-at', '-I', 
                               jdkpath1, '-I', jdkpath2, '-shared', '-o', 
                               'CocoJNI.dll', 'CocoJNI.c'])
                               
    # 3. Windows without Cygwin
    elif ('win32' in sys.platform) and ('cygwin' not in os.environ['PATH']):
        jdkpath = check_output(['where', 'javac'], stderr = STDOUT, 
                               env = os.environ, universal_newlines = True)  
        jdkpath1 = jdkpath.split("bin")[0] + 'include'
        jdkpath2 = jdkpath1 + '\\win32'
        run('code-experiments/build/java', ['gcc', '-Wl,--kill-at', '-I', jdkpath1, '-I', jdkpath2, 
                           '-shared', '-o', 'CocoJNI.dll', 'CocoJNI.c'])
                           
    # 4. Linux
    elif ('linux' in sys.platform):
        jdkpath = check_output(['locate', 'jni.h'], stderr = STDOUT, 
                               env = os.environ, universal_newlines = True)   
        jdkpath1 = jdkpath.split("jni.h")[0]
        jdkpath2 = jdkpath1 + '/linux'
        run('code-experiments/build/java', ['gcc', '-I', jdkpath1, '-I', jdkpath2, '-c', 
                           'CocoJNI.c'])
        run('code-experiments/build/java', ['gcc', '-I', jdkpath1, '-I', jdkpath2, '-o', 
                           'libCocoJNI.so', '-shared', 'CocoJNI.c'])
                           
    # 5. Mac
    elif ('darwin' in sys.platform):
        jdkpath = '/System/Library/Frameworks/JavaVM.framework/Headers'
        run('code-experiments/build/java', ['gcc', '-I', jdkpath, '-c', 'CocoJNI.c'])
        run('code-experiments/build/java', ['gcc', '-dynamiclib', '-o', 'libCocoJNI.jnilib',
                           'CocoJNI.o'])
    
    run('code-experiments/build/java', ['javac', 'Problem.java'])
    run('code-experiments/build/java', ['javac', 'Benchmark.java'])
    run('code-experiments/build/java', ['javac', 'Observer.java'])
    run('code-experiments/build/java', ['javac', 'Suite.java'])
    run('code-experiments/build/java', ['javac', 'ExampleExperiment.java'])

def run_java():
    """ Builds and runs the example experiment in Java """
    build_java()
    try:
        run('code-experiments/build/java', ['java', '-Djava.library.path=.', 'ExampleExperiment'])    
    except subprocess.CalledProcessError:
        sys.exit(-1)

def test_java():
    """ Builds and runs the test in Java, which is equal to the example experiment """
    build_java()
    try:
        run('code-experiments/build/java', ['java', '-Djava.library.path=.', 'ExampleExperiment'])    
    except subprocess.CalledProcessError:
        sys.exit(-1)

################################################################################
## Post processing
def test_postprocessing():
    install_postprocessing()
    python('code-postprocessing/bbob_pproc', ['__main__.py'])
    # python('code-postprocessing', ['-m', 'bbob_pproc'])

################################################################################
## Global
def build():
    builders = [
        build_c,
        #build_matlab,
        build_python,
        build_java, 
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

def run_all():
    run_c()
    run_java()
    run_python()
    
def test():
    test_c()
    test_java()
    test_python()

def help():
    print("""COCO framework bootstrap tool.

Usage: do.py <command> <arguments>

If you want to get going as quickly as possible do once

   python do.py run-<your-language>

and

    python do.py install-postprocessing

and you are all set.

Available commands for users:

  build-c              - Build C module
  build-java           - Build Java module
  build-matlab         - Build Matlab module
  build-matlab-sms     - Build SMS-EMOA example in Matlab
  build-octave         - Build Matlab module in Octave
  build-python         - Build Python modules
  build-python2        - Build Python 2 modules
  build-python3        - Build Python 3 modules
  install-postprocessing - Install postprocessing (user-locally)

  run-c                - Build and run example experiment in C
  run-java             - Build and run example experiment in Java
  run-matlab           - Build and run example experiment in MATLAB
  run-matlab-sms       - Build and run SMS-EMOA on bbob-biobj suite in MATLAB
  run-octave           - Build and run example experiment in Octave
  run-python           - Build and install COCO module and run tests and the
                         example experiment in Python, "no-tests" omits tests

Available commands for developers:

  build                - Build C, Java and Python modules
  run                  - Run example experiments in C, Java and Python
  test                 - Test C, Java and Python modules

  run-sandbox-python   - Run a Python script with installed COCO module
                         Takes a single argument (name of Python script file)
  
  test-c               - Build and run unit tests, integration tests 
                         and an example experiment test in C 
  test-c-unit          - Build and run unit tests in C
  test-c-integration   - Build and run integration tests in C
  test-c-example       - Build and run an example experiment test in C 
  test-java            - Build and run a test in Java
  test-python          - Build and run minimal test of Python module
  test-python2         - Build and run minimal test of Python 2 module
  test-python3         - Build and run minimal test of Python 3 module
  test-postprocessing  - Runs post-processing tests.
  leak-check           - Check for memory leaks in C


To build a release version which does not include debugging information in the 
amalgamations set the environment variable COCO_RELEASE to 'true'.
""")
def main(args):
    if len(args) < 1:
        help()
        sys.exit(0)
    cmd = args[0].replace('_', '-').lower()
    if cmd == 'build': build()
    elif cmd == 'run': run_all()
    elif cmd == 'test': test()
    elif cmd == 'build-c': build_c()
    elif cmd == 'build-java': build_java()
    elif cmd == 'build-matlab': build_matlab()
    elif cmd == 'build-matlab-sms': build_matlab_sms()
    elif cmd == 'build-octave': build_octave()    
    elif cmd == 'build-python': build_python()
    elif cmd == 'build-python2': build_python2()
    elif cmd == 'build-python3': build_python3()
    elif cmd == 'install-postprocessing': install_postprocessing()
    elif cmd == 'run-c': run_c()
    elif cmd == 'run-java': run_java()
    elif cmd == 'run-matlab': run_matlab()
    elif cmd == 'run-matlab-sms': run_matlab_sms()
    elif cmd == 'run-octave': run_octave()    
    elif cmd == 'run-python':
        run_python(False) if len(args) > 1 and args[1] == 'no-tests' else run_python()
    elif cmd == 'test-c': test_c()
    elif cmd == 'test-c-unit': test_c_unit()
    elif cmd == 'test-c-integration': test_c_integration()
    elif cmd == 'test-c-example': test_c_example()    
    elif cmd == 'test-java': test_java()
    elif cmd == 'test-python': test_python()
    elif cmd == 'test-python2': test_python2()
    elif cmd == 'test-python3': test_python3()
    elif cmd == 'test-postprocessing': test_postprocessing()
    elif cmd == 'leak-check': leak_check()
    else: help()

if __name__ == '__main__':
    release = os.getenv('COCO_RELEASE', 'false') == 'true'
    main(sys.argv[1:])
