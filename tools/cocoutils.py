## -*- mode: python -*-

## Lots of utility functions to abstract away platform differences.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
from shutil import copyfile, copytree, rmtree
from subprocess import CalledProcessError, check_output, call, STDOUT

def hg(args):
    """Run a Mercurial command and return its output.

    All errors are deemed fatal and the system will quit."""
    full_command = ['hg']
    full_command.extend(args)
    try:
        output = check_output(full_command, env=os.environ, universal_newlines=True)
        output = output.rstrip()
    except CalledProcessError as e:
        print('Failed to execute hg.')
        raise
    return output

def is_dirty():
    """Return True if the current working copy has uncommited changes."""
    return hg(['hg', 'id', '-i'])[-1] == '+'

def hg_version():
    """ Derive the current version number from the latest tag and the
    number of (local) commits since the tagged revision. """
    return hg(['log', '-r', '.', '--template', '{latesttag}.{latesttagdistance}'])

def hg_revision():
    return hg(['id', '-i'])

def run(directory, args):
    print("RUN\t%s in %s" % (" ".join(args), directory))
    oldwd = os.getcwd()
    try:
        os.chdir(directory)
        output = check_output(args, stderr=STDOUT, env=os.environ, 
                              universal_newlines=True)
    except CalledProcessError as e:
        print("ERROR: return value=%i" % e.returncode)
        print(e.output)
        raise
    finally:
        os.chdir(oldwd)

def python(directory, args, env=None):
    print("PYTHON\t%s in %s" % (" ".join(args), directory))
    oldwd = os.getcwd()
    if os.environ.get('PYTHON') is not None:
        ## Use the Python interpreter specified in the PYTHON
        ## environment variable.
        full_command = [os.environ['PYTHON']]
    else:
        ## No interpreter specified. Use the Python interpreter that
        ## is used to execute this script.
        full_command = [sys.executable]
    full_command.extend(args)
    try:
        os.chdir(directory)
        output = check_output(full_command, stderr=STDOUT, env=os.environ,
                              universal_newlines=True)
    except CalledProcessError as e:
        print("ERROR: return value=%i" % e.returncode)
        print(e.output)
        raise
    finally:
        os.chdir(oldwd)

def rscript(directory, args, env=None):
    print("RSCRIPT\t%s in %s" % (" ".join(args), directory))
    oldwd = os.getcwd()
    if os.environ.get('RSCRIPT') is not None:
        ## Use the Rscript interpreter specified in the RSCRIPT
        ## environment variable.
        full_command = [os.environ['RSCRIPT']]
    else:
        ## No interpreter specified. Try to find an Rscript interpreter.
        full_command = ['Rscript']
    full_command.extend(args)
    try:
        os.chdir(directory)
        output = check_output(full_command, stderr=STDOUT, env=os.environ,
                              universal_newlines=True)
    except CalledProcessError as e:
        print("ERROR: return value=%i" % e.returncode)
        print(e.output)
        raise
    finally:
        os.chdir(oldwd)

def copy_file(source, destination):
    print("COPY\t%s -> %s" % (source, destination))
    copyfile(source, destination)

def copy_tree(source_directory, destination_directory):
    if os.path.isdir(destination_directory):
        rmtree(destination_directory)
    print("COPY\t%s -> %s" % (source_directory, destination_directory))
    copytree(source_directory, destination_directory)
    
def write_file(string, destination):
    print("WRITE\t%s" % destination)
    with open(destination, 'w') as fd:
        fd.write(string)

def make(directory, target):
    """Run make to build a target"""
    print("MAKE\t%s in %s" % (target, directory))
    oldwd = os.getcwd()
    try:
        os.chdir(directory)
        output = check_output(['make', target], stderr=STDOUT, env=os.environ,
                              universal_newlines=True)
    except CalledProcessError as e:
        print("ERROR: return value=%i" % e.returncode)
        print(e.output)
        raise
    finally:
        os.chdir(oldwd)

def expand_file(source, destination, dictionary):
    print("EXPAND\t%s to %s" % (source, destination))
    from string import Template
    with open(source, 'r') as fd:
        content = Template(fd.read())
        with open(destination, "w") as outfd:
            outfd.write(content.safe_substitute(dictionary))
