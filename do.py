#!/usr/bin/env python
## -*- mode: python -*-
import os
import re
import sys
import glob
import time
import click
import shutil
import platform
import subprocess

from subprocess import STDOUT
from os.path import join, abspath


## Change to the root directory of repository and add our tools/
## subdirectory to system wide search path for modules.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(join('code-experiments', 'tools')))

from cocoutils import make, run, python, check_output # noqa: E402
from cocoutils import copy_file, expand_file, write_file # noqa: E402
from cocoutils import executable_path # noqa: E402
from cocoutils import git_version, git_revision # noqa: E402

CORE_FILES = ['code-experiments/src/coco_random.c',
              'code-experiments/src/coco_suite.c',
              'code-experiments/src/coco_observer.c',
              'code-experiments/src/coco_archive.c'
             ]

MATLAB_FILES = ['cocoCall.m', 'cocoEvaluateFunction.m', 'cocoObserver.m',
                'cocoObserverFree.m', 'cocoProblemAddObserver.m',
                'cocoProblemFinalTargetHit.m', 'cocoProblemFree.m',
                'cocoProblemGetDimension.m', 'cocoProblemGetEvaluations.m',
                'cocoProblemGetId.m', 'cocoProblemGetInitialSolution.m',
                'cocoProblemGetLargestValuesOfInterest.m',
                'cocoProblemGetName.m', 'cocoProblemGetNumberOfObjectives.m',
                'cocoProblemGetSmallestValuesOfInterest.m',
                'cocoProblemIsValid.m', 'cocoProblemRemoveObserver.m',
                'cocoSetLogLevel.m', 'cocoSuite.m', 'cocoSuiteFree.m',
                'cocoSuiteGetNextProblem.m', 'cocoSuiteGetProblem.m']

################################################################################

class Amalgator:
    def __init__(self, destination_file, release):
        self.release = release
        self.included_files = []
        self.destination_fd = open(destination_file, 'w')
        self.destination_fd.write("""
/************************************************************************
 * WARNING
 *
 * This file is an auto-generated amalgamation. Any changes made to this
 * file will be lost when it is regenerated!
 ************************************************************************/

""")

    def finish(self):
        self.destination_fd.close()

    def __del__(self):
        self.finish()

    def process_file(self, filename):
        if filename in self.included_files:
            return
        self.included_files.append(filename)
        with open(filename) as fd:
            line_number = 1
            if not self.release:
                self.destination_fd.write("#line %i \"%s\"\n" % (line_number, filename))
            for line in fd.readlines():
                ## Is this an include statement?
                matches = re.match("#include \"(.*)\"", line)
                if matches:
                    include_file = "/".join([os.path.dirname(filename), matches.group(1)])
                    ## Has this file not been included previously?
                    if include_file not in self.included_files:
                        self.process_file(include_file)
                    if not self.release:
                        self.destination_fd.write("#line %i \"%s\"\n" % 
                                                  (line_number + 1, filename))
                else:
                    self.destination_fd.write(line)
                line_number += 1


def amalgamate(source_files, destination_file, release=False):
    """Amalgamate a list of source files into a single unity build file

    Parameters
    ----------

    source_files
        List of files to amalgamate
    destination_file 
        Unity build file name
    release
        If False, embed information about the file name and line number
        where the source originated into the `destination_file`.
    """
    print("AML\t%s -> %s" % (str(source_files), destination_file))
    amalgator = Amalgator(destination_file, release)
    if isinstance(source_files, str):
        source_files = [source_files]
    for filename in source_files:
        amalgator.process_file(filename)
    amalgator.finish()


################################################################################
## CLI entry point
@click.group()
@click.option("-v", "--verbose", is_flag=True)
@click.option("-r", "--release", is_flag=True,
              help="Build and run release code (stripping most debugging information).")
@click.pass_context
def cli(ctx, verbose, release):
    """Universal meta build tool for the COCO platform"""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["release"] = release
    pass

################################################################################
## Global version managment
VERSION_FILES_WRITTEN = False

def _write_version_to_python(filename, version):
    print("VERSION\t%s" % (filename))
    with open(filename, "wt") as fd:
        fd.write("# file generated by do.py\n")
        fd.write("# don't change, don't track in version control!\n")
        fd.write(f'__version__ = version = "{version}"\n')

def _write_version_to_c_header(filename, version,
                               variable="coco_version",
                               guard="COCO_VERSION"):
    print("VERSION\t%s" % (filename))
    with open(filename, "wt") as fd:
        fd.write("/* file generated by do.py\n")
        fd.write(" * don't change, don't track in version control!\n")
        fd.write(" */\n\n")
        fd.write(f"#ifndef {guard}_H\n")
        fd.write(f"#define {guard}_H\n")
        fd.write("/**\n")
        fd.write("* @brief COCO's version.\n")
        fd.write("*\n")
        fd.write("* The version number is dervied from the latest tag in the\n")
        fd.write("* repository plus the number of commits after the tag.\n")
        fd.write("*/\n")
        fd.write("/**@{*/\n")
        fd.write(f'static const char { variable }[{ len(version)+1 }] = "{ version }";\n')
        fd.write("/**@}*/\n")
        fd.write("#endif\n")


@cli.command()
@click.pass_context
def write_version(ctx):
    """Write current version number into build files."""
    if "VERSION_FILES_WRITTEN" in ctx.obj:
        return
    COCO_VERSION = git_version(pep440=True)
    print(f"INFO\tVersion is {COCO_VERSION}")
    _write_version_to_c_header("code-experiments/src/coco_version.h", COCO_VERSION)
    
    write_file(COCO_VERSION, "code-experiments/build/c/VERSION")
    write_file(COCO_VERSION, "code-experiments/build/java/VERSION")
    _write_version_to_python("code-experiments/build/python/src/cocoex/_version.py", COCO_VERSION)
    write_file(COCO_VERSION, "code-experiments/build/matlab/VERSION")

    _write_version_to_python("code-postprocessing/cocopp/_version.py", COCO_VERSION)
    # Guard against inadvertent multiple calls
    ctx.obj["VERSION_FILES_WRITTEN"] = True


################################################################################
## C
@cli.command()
@click.pass_context
def amalgamate_c(ctx):
    """Only amalgamate the C files"""
    if "C_FILES_AMALGAMATED" in ctx.obj:
        return
    ctx.invoke(write_version)

    amalgamate(CORE_FILES + ['code-experiments/src/coco_runtime_c.c'],
               'code-experiments/build/c/coco.c', ctx.obj["release"])
    amalgamate('code-experiments/src/coco.h',
               'code-experiments/build/c/coco.h', ctx.obj["release"])
    copy_file('code-experiments/build/c/coco.c',
              'code-experiments/examples/bbob2009-c-cmaes/coco.c')
    copy_file('code-experiments/build/c/coco.h',
              'code-experiments/examples/bbob2009-c-cmaes/coco.h')
    ctx.obj["C_FILES_AMALGAMATED"] = True


@cli.command("build-c")
@click.pass_context
def build_c(ctx):
    """Builds the C code"""
    ctx.invoke(amalgamate_c)
    make("code-experiments/build/c", "clean", verbose=ctx.obj["verbose"])
    make("code-experiments/build/c", "all", verbose=ctx.obj["verbose"])


def _test_c_unit(verbose=False):
    """Builds and runs unit tests in C"""
    # Perform unit tests
    copy_file('code-experiments/build/c/coco.c', 'code-experiments/test/unit-test/coco.c')
    copy_file('code-experiments/build/c/coco.h', 'code-experiments/test/unit-test/coco.h')
    make("code-experiments/test/unit-test", "clean", verbose=verbose)
    make("code-experiments/test/unit-test", "all", verbose=verbose)

    try:
        run('code-experiments/test/unit-test', ['./unit_test'], verbose=verbose)
    except subprocess.CalledProcessError:
        sys.exit(-1)


def _test_c_integration(verbose):
    """Builds and runs integration tests in C"""
    copy_file('code-experiments/build/c/coco.c',
              'code-experiments/test/integration-test/coco.c')
    copy_file('code-experiments/build/c/coco.h',
              'code-experiments/test/integration-test/coco.h')
    copy_file('code-experiments/src/bbob2009_testcases.txt',
              'code-experiments/test/integration-test/bbob2009_testcases.txt')
    copy_file('code-experiments/src/bbob2009_testcases2.txt',
              'code-experiments/test/integration-test/bbob2009_testcases2.txt')
    make("code-experiments/test/integration-test", "clean", verbose=verbose)
    make("code-experiments/test/integration-test", "all", verbose=verbose)

    try:
        run('code-experiments/test/integration-test',
            ['./test_coco', 'bbob2009_testcases.txt'], verbose=verbose)
        run('code-experiments/test/integration-test',
            ['./test_coco', 'bbob2009_testcases2.txt'], verbose=verbose)
        run('code-experiments/test/integration-test',
            ['./test_instance_extraction'], verbose=verbose)
        run('code-experiments/test/integration-test',
            ['./test_biobj'], verbose=verbose)
        run('code-experiments/test/integration-test',
            ['./test_bbob-constrained'], verbose=verbose)
        run('code-experiments/test/integration-test',
            ['./test_bbob-largescale'], verbose=verbose)
        run('code-experiments/test/integration-test',
            ['./test_bbob-mixint'], verbose=verbose)
    except subprocess.CalledProcessError:
        sys.exit(-1)


def _test_c_example(verbose):
    """Builds and runs C example experiment"""
    ## FIXME: Run test in temp directory and discard results
    if os.path.exists('code-experiments/test/example-test'):
        shutil.rmtree('code-experiments/test/example-test')
        time.sleep(1)  # Needed to avoid permission errors for os.makedirs
    os.makedirs('code-experiments/test/example-test')
    copy_file('code-experiments/build/c/coco.c',
              'code-experiments/test/example-test/coco.c')
    copy_file('code-experiments/build/c/coco.h',
              'code-experiments/test/example-test/coco.h')
    copy_file('code-experiments/build/c/example_experiment.c',
              'code-experiments/test/example-test/example_experiment.c')
    copy_file('code-experiments/build/c/Makefile.in',
              'code-experiments/test/example-test/Makefile.in')
    copy_file('code-experiments/build/c/Makefile_win_gcc.in',
              'code-experiments/test/example-test/Makefile_win_gcc.in')
    make("code-experiments/test/example-test", "clean", verbose=verbose)
    make("code-experiments/test/example-test", "all", verbose=verbose)

    try:
        run('code-experiments/test/example-test',
            ['./example_experiment'],
            verbose=verbose)
    except subprocess.CalledProcessError:
        sys.exit(-1)


@cli.command()
@click.option("--unit", is_flag=True, 
              help="Run unit tests")
@click.option("--integration", is_flag=True, 
              help="Run integration tests")
@click.option("--example", is_flag=True, 
              help="Run examples")
@click.pass_context
def test_c(ctx, integration, unit, example):
    """Builds and tests the C code"""
    ctx.invoke(amalgamate_c)
    if unit:
        print("INFO\tRunning C unit tests...")
        _test_c_unit(ctx.obj["verbose"])
    if integration: 
        print("INFO\tRunning C integration tests...")
        _test_c_integration(ctx.obj["verbose"])
    if example:
        print("INFO\tRunning C example")
        _test_c_example(ctx.obj["verbose"])


################################################################################
## Python 2
def install_error(e):
    exception_message = e.output.splitlines()
    formatted_message = ["|" + " " * 77 + "|"]
    for line in exception_message:
        while len(line) > 75:
            formatted_message.append("| " + line[:75] + " |")
            line = line[75:]
        formatted_message.append("| " + line.ljust(75) + " |")
    print("""
An exception occurred while trying to install packages.

A common reason for this error is insufficient access rights
to the installation directory. The original exception message
is as follows:

/----------------------------< EXCEPTION MESSAGE >----------------------------\\
{0}
\\-----------------------------------------------------------------------------/

To fix an access rights issue, you may try the following:

- Run the same command with "install-user" as additional argument.
  To get further help run do.py without a specific command.

- On *nix systems or MacOS, run the same command with a preceded "sudo ".

- Gain write access to the installation directory by changing
  access permissions or gaining administrative access.

""".format("\n".join(formatted_message)))
    return True

@cli.command()
@click.pass_context
def amalgamate_python(ctx):
    """Amalgamate Python files"""
    ctx.invoke(write_version)
    amalgamate(CORE_FILES + ['code-experiments/src/coco_runtime_c.c'],
               'code-experiments/build/python/src/cocoex/coco.c',
               ctx.obj["release"])
    amalgamate('code-experiments/src/coco.h',
               'code-experiments/build/python/src/cocoex/coco.h',
               ctx.obj["release"])
    copy_file('code-experiments/src/bbob2009_testcases.txt',
              'code-experiments/build/python/bbob2009_testcases.txt')
    copy_file('code-experiments/src/bbob2009_testcases2.txt',
              'code-experiments/build/python/bbob2009_testcases2.txt')
    copy_file('code-experiments/build/python/README.md',
              'code-experiments/build/python/README.txt')




@cli.command()
@click.option("--skip-cocoex", is_flag=True,
              help="Skip installation of cocoex package")
@click.option("--skip-cocopp", is_flag=True,
              help="Skip installation of cocopp package")
@click.pass_context
def build_python(ctx, skip_cocoex, skip_cocopp):
    """Builds the Python packages (cocoex and cocopp)"""
    ctx.invoke(amalgamate_python)
    cmdline = ["-m", "build"]

    if not skip_cocoex:
        python('code-experiments/build/python', cmdline,
               verbose=ctx.obj["verbose"])
    if not skip_cocopp:
        python('code-postprocessing', cmdline,
               verbose=ctx.obj["verbose"])


@cli.command()
@click.pass_context
def test_python(ctx):
    """Builds and tests the cocoex Python package"""
    ctx.invoke(install_python, skip_cocoex=True)
    try:
        python('code-experiments/build/python/tests/', 
               ['coco_test.py'])
        python('code-experiments/build/python',
               ['example_experiment.py', 'bbob'])
    except subprocess.CalledProcessError:
        sys.exit(-1)


@cli.command()
@click.option("--skip-cocoex", is_flag=True,
              help="Skip installation of cocoex package")
@click.option("--skip-cocopp", is_flag=True,
              help="Skip installation of cocopp package")
@click.option("--user/--system", default=True, 
              help="Install python packages for current user or globally")
@click.pass_context
def install_python(ctx, skip_cocoex, skip_cocopp, user):
    """Installs the Python packages (cocoex and cocopp)"""
    install_options = []
    cmdline = ["-m", "pip", "install", ".", *install_options]
    if user:
        install_options.append("--user")
    if ctx.obj["verbose"]:
        install_options.append("--verbose")
    
    ctx.invoke(amalgamate_python)

    if not skip_cocoex:
        python('code-experiments/build/python', cmdline)
    if not skip_cocopp:
        python('code-postprocessing', cmdline)

@cli.command()
@click.option("--keep-cocoex", is_flag=True,
              help="Don't uninstall cocoex package")
@click.option("--keep-cocopp", is_flag=True,
              help="Dont't uninstall cocopp package")
@click.pass_context
def uninstall_python(ctx, keep_cocoex, keep_cocopp):
    """Uninstall the Python packages (cocoex and cocopp)"""
    opts = []
    if ctx.obj["verbose"]:
        opts.append("-v")
    if not keep_cocoex:
        python(".", ["-m", "pip", "uninstall", "cocoex", "-y", *opts]) 
    if not keep_cocopp:
        python(".", ["-m", "pip", "uninstall", "cocopp", "-y", *opts]) 


################################################################################
## Matlab

@cli.command
@click.pass_context
def amalgamate_matlab(ctx):
    """Amalgamate Matlab files"""
    write_version()
    amalgamate(CORE_FILES + ['code-experiments/src/coco_runtime_matlab.c'],
               'code-experiments/build/matlab/coco.c',
               ctx.obj["release"])
    amalgamate('code-experiments/src/coco.h',
               'code-experiments/build/matlab/coco.h',
               ctx.obj["release"])

@cli.command
@click.pass_context
def build_matlab(ctx,):
    """Builds MATLAB example in build/matlab/ but not the one in examples/."""
    ctx.invoke(amalgamate_matlab)
    run('code-experiments/build/matlab',
        ['matlab', '-nodisplay', '-nosplash', '-r', 'setup, exit'],
        verbose=ctx.obj["verbose"])

@cli.command
@click.pass_context
def test_matlab():
    """ Builds and runs the example experiment in build/matlab/ in MATLAB """
    print('CLEAN\tmex files from code-experiments/build/matlab/')
    # remove the mex files for a clean compilation first
    for filename in glob.glob('code-experiments/build/matlab/*.mex*'):
        os.remove(filename)
    # amalgamate, copy, and build
    ctx.invoke(build_matlab)
    ## FIXME: Necessary?
    wait_for_compilation_to_finish('./code-experiments/build/matlab/cocoCall')
    # run after compilation finished
    run('code-experiments/build/matlab',
        ['matlab', '-nodisplay', '-nosplash', '-r', 'exampleexperiment, exit'],
        verbose=ctx.obj["verbose"])


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

    print('Wait for compilation to finish', end='')
    while not is_compiled(filenameprefix):
        time.sleep(2)
        print('.', end='')
    print(' ')


@cli.command()
@click.pass_context
def build_matlab_sms(ctx):
    """Builds the SMS-EMOA in MATLAB """
    destination_folder = 'code-experiments/examples/bbob-biobj-matlab-smsemoa'
    # amalgamate and copy files
    write_version()
    amalgamate(CORE_FILES + ['code-experiments/src/coco_runtime_matlab.c'],
               join(destination_folder, 'coco.c'), 
               ctx.obj["release"])
    amalgamate('code-experiments/src/coco.h',
               join(destination_folder, 'coco.h'),
               ctx.obj["release"])
    for f in MATLAB_FILES:
        copy_file(join('code-experiments/build/matlab/', f), join(destination_folder, f))
    write_file(git_revision(), join(destination_folder, "REVISION"))
    write_file(git_version(), join(destination_folder, "VERSION"))
    copy_file('code-experiments/build/matlab/cocoCall.c', join(destination_folder, 'cocoCall.c'))
    # compile
    run(destination_folder, ['matlab', '-nodisplay', '-nosplash', '-r', 'setup, exit'])


@cli.command()
@click.pass_context
def run_matlab_sms(ctx):
    """ Builds and runs the SMS-EMOA in MATLAB """
    print('CLEAN\tmex files from code-experiments/examples/bbob-biobj-matlab-smsemoa/')
    # remove the mex files for a clean compilation first
    for filename in glob.glob('code-experiments/examples/bbob-biobj-matlab-smsemoa/*.mex*'):
        os.remove(filename)
    # amalgamate, copy, and build
    build_matlab_sms()
    wait_for_compilation_to_finish('./code-experiments/examples/bbob-biobj-matlab-smsemoa/paretofront')
    # run after compilation finished
    run('code-experiments/examples/bbob-biobj-matlab-smsemoa',
        ['matlab', '-nodisplay', '-nosplash', '-r', 'run_smsemoa_on_bbob_biobj, exit'],
        verbose=ctx.obj["verbose"])


################################################################################
## Octave
@cli.command()
@click.pass_context
def build_octave(ctx):
    """Builds example in build/matlab/ with GNU Octave."""
    ctx.invoke(write_version)
    amalgamate(CORE_FILES + ['code-experiments/src/coco_runtime_c.c'],
               'code-experiments/build/matlab/coco.c',
               ctx.obj["release"])
    amalgamate('code-experiments/src/coco.h', 
               'code-experiments/build/matlab/coco.h',
               ctx.obj["release"])
    write_file(git_revision(), "code-experiments/build/matlab/REVISION")
    write_file(git_version(), "code-experiments/build/matlab/VERSION")

    # make sure that under Windows, run_octave has been run at least once
    # before to provide the necessary octave_coco.bat file
    if 'win32' in sys.platform:
        run('code-experiments/build/matlab',
            ['octave_coco.bat', '--no-gui', 'setup.m'],
            verbose=ctx.obj["verbose"])
    else:
        run('code-experiments/build/matlab',
            ['octave', '--no-gui', 'setup.m'], 
            verbose=ctx.obj["verbose"])


@cli.command()
@click.pass_context
def test_octave(ctx):
    """Builds and tests Octave code"""
    ctx.invoke(build_octave)
    try:
        if 'win32' in sys.platform:
            run('code-experiments/build/matlab',
                ['octave_coco.bat', '--no-gui', 'exampleexperiment.m'],
                verbose=ctx.obj["verbose"])
        else:
            run('code-experiments/build/matlab',
                ['octave', '--no-gui', 'exampleexperiment.m'], 
                verbose=ctx.obj["verbose"])
    except subprocess.CalledProcessError:
        sys.exit(-1)


@cli.command()
@click.pass_context
def build_octave_sms(ctx):
    """Builds the SMS-EMOA in Octave """
    destination_folder = 'code-experiments/examples/bbob-biobj-matlab-smsemoa'
    # amalgamate and copy files
    write_version()
    amalgamate(CORE_FILES + ['code-experiments/src/coco_runtime_c.c'],
               join(destination_folder, 'coco.c'),
               ctx.obj["release"])
    amalgamate('code-experiments/src/coco.h',
               join(destination_folder, 'coco.h'),
               ctx.obj["release"])
    for f in MATLAB_FILES:
        copy_file(join('code-experiments/build/matlab/', f), join(destination_folder, f))
    write_file(git_revision(), join(destination_folder, "REVISION"))
    write_file(git_version(), join(destination_folder, "VERSION"))
    copy_file('code-experiments/build/matlab/cocoCall.c', join(destination_folder, 'cocoCall.c'))
    # compile
    if 'win32' in sys.platform:
        run(destination_folder, ['octave_coco.bat', '--no-gui', 'setup.m'])
    else:
        run(destination_folder, ['octave', '--no-gui', 'setup.m'])


def run_octave_sms():
    """ Builds and runs the SMS-EMOA in Octave

        Note: does not work yet with all Windows/Octave versions.
    """

    print('CLEAN\tmex files from code-experiments/examples/bbob-biobj-matlab-smsemoa/')
    destination_folder = 'code-experiments/examples/bbob-biobj-matlab-smsemoa'
    # remove the mex files for a clean compilation first
    for filename in glob.glob(join(destination_folder, '*.mex*')):
        os.remove(filename)
    # amalgamate, copy, and build
    build_octave_sms()
    wait_for_compilation_to_finish(join(destination_folder, 'paretofront'))
    # run after compilation finished
    if 'win32' in sys.platform:
        run(destination_folder,
            ['octave_coco.bat', '-nogui', 'run_smsemoa_on_bbob_biobj.m'],
            verbose=ctx.obj["verbose"])
    else:
        run(destination_folder,
            ['octave', '-nogui', 'run_smsemoa_on_bbob_biobj.m'],
            verbose=ctx.obj["verbose"])

################################################################################
## Java
@cli.command()
@click.pass_context
def build_java(ctx):
    """ Builds the example experiment in Java """
    ctx.invoke(write_version)
    amalgamate(CORE_FILES + ['code-experiments/src/coco_runtime_c.c'],
               'code-experiments/build/java/coco.c', 
               ctx.obj["release"])
    amalgamate('code-experiments/src/coco.h',
               'code-experiments/build/java/coco.h',
               ctx.obj["release"])

    javacpath = executable_path('javac')
    javahpath = executable_path('javah')
    if javacpath and javahpath:
        run('code-experiments/build/java', ['javac', '-classpath', '.', 'CocoJNI.java'], verbose=ctx.obj["verbose"])
        run('code-experiments/build/java', ['javah', '-classpath', '.', 'CocoJNI'], verbose=ctx.obj["verbose"])
    elif javacpath:
        run('code-experiments/build/java', ['javac', '-h', '.', 'CocoJNI.java'], verbose=ctx.obj["verbose"])
    else:
        raise RuntimeError('Can not find javac path!')


    # Finds the path to the headers jni.h and jni_md.h (platform-dependent)
    # and compiles the CocoJNI library (compiler-dependent). So far, only
    # the following cases are covered:

    # 1. Windows with Cygwin (both 64-bit)
    # Note that 'win32' stands for both Windows 32-bit and 64-bit.
    # Since platform 'cygwin' does not work as expected, we need to look for it in the PATH.
    if ('win32' in sys.platform) and ('cygwin' in os.environ['PATH']):
        jdkpath = check_output(['where', 'javac'], stderr=STDOUT,
                               env=os.environ, universal_newlines=True)
        jdkpath1 = jdkpath.split("bin")[0] + 'include'
        jdkpath2 = jdkpath1 + '\\win32'

        if '64' in platform.machine():
            run('code-experiments/build/java', ['x86_64-w64-mingw32-gcc', '-I',
                                                jdkpath1, '-I', jdkpath2, '-shared', '-o',
                                                'CocoJNI.dll', 'CocoJNI.c'], verbose=ctx.obj["verbose"])

            # 2. Windows with Cygwin (both 32-bit)
        elif '32' in platform.machine() or 'x86' in platform.machine():
            run('code-experiments/build/java', ['i686-w64-mingw32-gcc', '-Wl,--kill-at', '-I',
                                                jdkpath1, '-I', jdkpath2, '-shared', '-o',
                                                'CocoJNI.dll', 'CocoJNI.c'], verbose=ctx.obj["verbose"])

    # 3. Windows without Cygwin
    elif ('win32' in sys.platform) and ('cygwin' not in os.environ['PATH']):
        jdkpath = check_output(['where', 'javac'], stderr=STDOUT,
                               env=os.environ, universal_newlines=True)
        jdkpath1 = jdkpath.split("bin")[0] + 'include'
        jdkpath2 = jdkpath1 + '\\win32'
        run('code-experiments/build/java',
            ['gcc', '-Wl,--kill-at', '-I', jdkpath1, '-I', jdkpath2,
             '-shared', '-o', 'CocoJNI.dll', 'CocoJNI.c'],
            verbose=ctx.obj["verbose"])

    # 4. Linux
    elif 'linux' in sys.platform:
        # bad bad bad...
        #jdkpath = check_output(['locate', 'jni.h'], stderr=STDOUT,
        #                       env=os.environ, universal_newlines=True)
        #jdkpath1 = jdkpath.split("jni.h")[0]
        # better
        javapath = executable_path('java')
        if not javapath:
            raise RuntimeError('Can not find Java executable')
        jdkhome = abspath(join(javapath, os.pardir, os.pardir))
        if os.path.basename(jdkhome) == 'jre':
            jdkhome = join(jdkhome, os.pardir)
        jdkpath1 = join(jdkhome, 'include')
        jdkpath2 = jdkpath1 + '/linux'
        run('code-experiments/build/java',
            ['gcc', '-I', jdkpath1, '-I', jdkpath2, '-c', 'CocoJNI.c'],
            verbose=ctx.obj["verbose"])
        run('code-experiments/build/java',
            ['gcc', '-I', jdkpath1, '-I', jdkpath2, '-o',
             'libCocoJNI.so', '-fPIC', '-shared', 'CocoJNI.c'],
            verbose=ctx.obj["verbose"])

    # 5. Mac
    elif 'darwin' in sys.platform:
        jdkversion = check_output(['javac', '-version'], stderr=STDOUT,
                                  env=os.environ, universal_newlines=True)
        jdkversion = jdkversion.split()[1]
        jdkpath = '/System/Library/Frameworks/JavaVM.framework/Headers'
        jdkpath1 = ('/Library/Java/JavaVirtualMachines/jdk' +
                    jdkversion + '.jdk/Contents/Home/include')
        jdkpath2 = jdkpath1 + '/darwin'
        run('code-experiments/build/java',
            ['gcc', '-I', jdkpath, '-I', jdkpath1, '-I', jdkpath2, '-c', 'CocoJNI.c'],
            verbose=ctx.obj["verbose"])
        run('code-experiments/build/java',
            ['gcc', '-dynamiclib', '-o', 'libCocoJNI.jnilib', 'CocoJNI.o'],
            verbose=ctx.obj["verbose"])

    run('code-experiments/build/java', 
        ['javac', '-classpath', '.', 'Problem.java'],
        verbose=ctx.obj["verbose"])
    run('code-experiments/build/java', 
        ['javac', '-classpath', '.', 'Benchmark.java'],
        verbose=ctx.obj["verbose"])
    run('code-experiments/build/java', 
        ['javac', '-classpath', '.', 'Observer.java'],
        verbose=ctx.obj["verbose"])
    run('code-experiments/build/java', 
        ['javac', '-classpath', '.', 'Suite.java'],
        verbose=ctx.obj["verbose"])
    run('code-experiments/build/java', 
        ['javac', '-classpath', '.', 'ExampleExperiment.java'],
        verbose=ctx.obj["verbose"])


@cli.command()
@click.pass_context
def test_java(ctx):
    """ Builds and runs the test in Java, which is equal to the example experiment """
    ctx.invoke(build_java)
    try:
        run('code-experiments/build/java',
            ['java', '-Djava.library.path=.', 'ExampleExperiment'],
            verbose=ctx.obj["verbose"])
    except subprocess.CalledProcessError:
        sys.exit(-1)


################################################################################
## Rust
@cli.command()
@click.pass_context
def build_rust(ctx):
    """ Builds the example experiment in Rust """
    ctx.invoke(write_version)
    amalgamate(CORE_FILES + ['code-experiments/src/coco_runtime_c.c'],
               'code-experiments/build/rust/coco-sys/vendor/coco.c',
               ctx.obj["release"])
    amalgamate('code-experiments/src/coco.h',
               'code-experiments/build/rust/coco-sys/vendor/coco.h',
               ctx.obj["release"])
    copy_file('code-experiments/src/coco_internal.h',
              'code-experiments/build/rust/coco-sys/vendor/coco_internal.h')

    write_file(git_revision(), "code-experiments/build/rust/REVISION")
    write_file(git_version(), "code-experiments/build/rust/VERSION")

    cargo_path = executable_path('cargo')
    if not cargo_path:
        raise RuntimeError('Can not find cargo path')

    bindgen_path = executable_path('bindgen')
    if not bindgen_path:
        raise RuntimeError('Can not find bindgen path')

    bindgen_call = ['bindgen', 'wrapper.h', '-o', 'vendor/coco.rs',
                    '--blocklist-item', 'FP_NORMAL',
                    '--blocklist-item', 'FP_SUBNORMAL',
                    '--blocklist-item', 'FP_ZERO',
                    '--blocklist-item', 'FP_INFINITE',
                    '--blocklist-item', 'FP_NAN']
    run('code-experiments/build/rust/coco-sys', bindgen_call, verbose=ctx.obj["verbose"])

    run('code-experiments/build/rust/coco-sys', ['cargo', 'build'], verbose=ctx.obj["verbose"])
    run('code-experiments/build/rust', ['cargo', 'build'], verbose=ctx.obj["verbose"])


@cli.command()
@click.option("--unit", is_flag=True,
              help="Run unit tests con coco-sys and coco crate")
@click.option("--example", is_flag=True,
              help="Run Rust example")
@click.pass_context
def test_rust(ctx, unit, example):
    """ Builds and runs the test in Rust """
    ctx.invoke(build_rust)
    if unit:
        print("INFO\tRunning Rust unit tests...")
        try:
            run('code-experiments/build/rust/coco-sys',
                ['cargo', 'test'],
                verbose=ctx.obj["verbose"])
            run('code-experiments/build/rust',
                ['cargo', 'test'],
                verbose=ctx.obj["verbose"])
        except subprocess.CalledProcessError:
            sys.exit(-1)
    if example:
        print("INFO\tRunning Rust example...")
        try:
            run('code-experiments/build/rust',
                ['cargo', 'run', '--example', 'example-experiment'],
                verbose=ctx.obj["verbose"])
        except subprocess.CalledProcessError:
            sys.exit(-1)


################################################################################
## Post processing
def test_postprocessing(all_tests=False, package_install_option=()):
    # install_postprocessing(package_install_option = package_install_option)
    try:
        if all_tests:
            # run example experiment to have a recent data set to postprocess:
            build_python(package_install_option=package_install_option)
            python('code-experiments/build/python/', ['-c', '''
from __future__ import print_function
try:
    import example_experiment as ee
except Exception as e:
    print(e)
ee.SOLVER = ee.random_search  # which is default anyway
for ee.suite_name, ee.observer_options['result_folder'] in [
        ["bbob-biobj", "RS-bi"],  # use a short path for Jenkins
        ["bbob", "RS-bb"],
        ["bbob-constrained", "RS-co"],
        ["bbob-largescale", "RS-la"],
        ["bbob-mixint", "RS-mi"],
        ["bbob-biobj-mixint", "RS-bi-mi"]
    ]:
    print("  suite %s" % ee.suite_name, end=' ')  # these prints are swallowed
    if ee.suite_name in ee.cocoex.known_suite_names:
        print("testing into folder %s" % ee.observer_options['result_folder'])
        ee.main()
    else:
        print("is not known")
                '''], verbose=ctx.obj["verbose"])
            # now run all tests
            python('code-postprocessing/cocopp',
                   ['test.py', 'all', sys.executable], verbose=ctx.obj["verbose"])
        else:
            python('code-postprocessing/cocopp', ['test.py', sys.executable],
                   verbose=ctx.obj["verbose"])

        # also run the doctests in aRTAplots/generate_aRTA_plot.py:
        python('code-postprocessing/aRTAplots', ['generate_aRTA_plot.py'], verbose=ctx.obj["verbose"])
    except subprocess.CalledProcessError:
        sys.exit(-1)
    finally:
        # always remove folder of previously run experiments:
        for s in ['bi', 'bb', 'co', 'la', 'mi', 'bi-mi']:
            shutil.rmtree('code-experiments/build/python/exdata/RS-' + s,
                          ignore_errors=True)

def verify_postprocessing(package_install_option = ()):
    install_postprocessing(package_install_option = package_install_option)
    # This is not affected by the ctx.obj["verbose"] value. Verbose should always be True.
    python('code-postprocessing/cocopp', ['preparehtml.py', '-v'], verbose=True)


################################################################################
## Pre-processing
def install_preprocessing(package_install_option = ()):
    global RELEASE
    write_version()
    install_postprocessing(package_install_option=package_install_option)
    expand_file(join('code-preprocessing/archive-update', 'setup.py.in'),
                join('code-preprocessing/archive-update', 'setup.py'),
                {'COCO_VERSION': git_version(pep440=True)})
    build_python(package_install_option = package_install_option)
    amalgamate(CORE_FILES + ['code-experiments/src/coco_runtime_c.c'],
               'code-preprocessing/archive-update/interface/coco.c',
               RELEASE)
    amalgamate('code-experiments/src/coco.h',
               'code-preprocessing/archive-update/interface/coco.h',
               RELEASE)
    python('code-preprocessing/archive-update',
           ['setup.py', 'install'] + package_install_option,
           verbose=ctx.obj["verbose"], custom_exception_handler=install_error)


def test_preprocessing(package_install_option = ()):
    install_preprocessing(package_install_option = package_install_option)
    python('code-preprocessing/archive-update', ['-m', 'pytest'], verbose=ctx.obj["verbose"])
    python('code-preprocessing/log-reconstruction', ['-m', 'pytest'], verbose=ctx.obj["verbose"])

################################################################################
## Global
@cli.command
@click.pass_context
def build_all(ctx):
    """Build C, Java, Octave, Python and Rust."""
    ctx.invoke(build_c)
    ctx.invoke(build_java)
    ctx.invoke(build_octave)
    ctx.invoke(build_python)
    ctx.invoke(build_rust)


@cli.command
@click.pass_context
def test_all(ctx):
    """Build and test C, Java, Octave, Python and Rust."""
    ctx.invoke(test_c)
    ctx.invoke(test_java)
    ctx.invoke(test_octave)
    ctx.invoke(test_python)
    ctx.invoke(test_rust)


if __name__ == '__main__':
    cli()
