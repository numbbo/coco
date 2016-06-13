## -*- mode: python -*-

## Lots of utility functions to abstract away platform differences.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
from shutil import copyfile, copytree, rmtree
from subprocess import CalledProcessError, call, STDOUT

try:
    from subprocess import check_output
except ImportError:
    import subprocess
    def check_output(*popenargs, **kwargs):
        r"""Run command with arguments and return its output as a byte string.
        Backported from Python 2.7 as it's implemented as pure python on stdlib.
        >>> check_output(['/usr/bin/python', '--version'])
        Python 2.6.2

        WARNING: This method is also defined in ../../code-postprocessing/bbob_pproc/toolsdivers.py.
        If you change something you have to change it in both files.
        """
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            error = subprocess.CalledProcessError(retcode, cmd)
            error.output = output
            raise error
        return output


def check_output_with_print(verbose, *popenargs, **kwargs):
    output = check_output(*popenargs, **kwargs)
    if verbose:
        print(output)

    return output

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

def git(args):
    """Run a git command and return its output.

    All errors are deemed fatal and the system will quit.

    WARNING: This method is also defined in ../../code-postprocessing/bbob_pproc/toolsdivers.py.
    If you change something you have to change it in both files.
    """
    full_command = ['git']
    full_command.extend(args)
    try:
        output = check_output(full_command, env=os.environ,
                              stderr=STDOUT, universal_newlines=True)
        output = output.rstrip()
    except CalledProcessError as e:
        # print('Failed to execute "%s"' % str(full_command))
        raise
    return output

def is_dirty():
    """Return True if the current working copy has uncommited changes."""
    raise NotImplementedError()
    return hg(['hg', 'id', '-i'])[-1] == '+'

def git_version(pep440=False):
    """Return somewhat readible version number from git, like
    '0.1-6015-ga0a3769' if not pep440 else '0.1.6015'

    """
    try:
        res = git(['describe', '--tags'])
    except:
        res = os.path.split(os.getcwd())[-1]
    if pep440:
        while len(res) and res[0] not in '0123456789':
            res = res[1:]
        if '-' in res:
           return '.'.join(res.split('-')[:2])
        else:
            return res
    else:
        return res

def git_revision():
    """Return unreadible git revision identifier, like
    a0a3769da32436c27df84d1b9b0915447aebf4d0"""
    try:
        return git(['rev-parse', 'HEAD'])
    except:
        # print('git revision call failed')
        return ""

def run(directory, args, verbose=False):
    print("RUN\t%s in %s" % (" ".join(args), directory))
    oldwd = os.getcwd()
    try:
        os.chdir(directory)
        output = check_output_with_print(verbose, args, stderr=STDOUT, env=os.environ,
                              universal_newlines=True)
        # print(output)
    except CalledProcessError as e:
        print("ERROR: return value=%i" % e.returncode)
        print(e.output)
        raise
    finally:
        os.chdir(oldwd)

def python(directory, args, env=None, verbose=False):
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
        output = check_output_with_print(verbose, full_command, stderr=STDOUT, env=os.environ,
                              universal_newlines=True)
        # print(output)
    except CalledProcessError as e:
        print("ERROR: return value=%i" % e.returncode)
        print(e.output)
        raise
    finally:
        os.chdir(oldwd)

def rscript(directory, args, env=None, verbose=False):
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
        output = check_output_with_print(verbose, full_command, stderr=STDOUT, env=os.environ,
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
    """CAVEAT: this removes the destination tree if present!"""
    if os.path.isdir(destination_directory):
        rmtree(destination_directory)
    print("COPY\t%s -> %s" % (source_directory, destination_directory))
    copytree(source_directory, destination_directory)

def write_file(string, destination):
    print("WRITE\t%s" % destination)
    with open(destination, 'w') as fd:
        fd.write(string)

def make(directory, target, verbose=False):
    """Run make to build a target"""
    print("MAKE\t%s in %s" % (target, directory))
    oldwd = os.getcwd()
    try:
        os.chdir(directory)
        # prepare makefile(s)
        if ((('win32' in sys.platform) or ('win64' in sys.platform)) and
            ('cygwin' not in os.environ['PATH'])):
            # only if under Windows and without Cygwin, we need a specific
            # Windows makefile
            copy_file('Makefile_win_gcc.in', 'Makefile')
        else:
            copy_file('Makefile.in', 'Makefile')

        output = check_output_with_print(verbose, ['make', target], stderr=STDOUT, env=os.environ,
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
