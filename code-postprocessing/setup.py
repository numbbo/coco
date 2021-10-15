#!/usr/bin/env python
"""setup cocopp

Prepare distribution::

    make a git tag (on GitHub) to have a new distribution release number

    outcomment (activate) install requirements below (in this file)
    python setup.py check
    python setup.py sdist bdist_wheel --universal > dist_call_output.txt ; less dist_call_output.txt  # bdist_wininst

Check distribution and project description::

    tree build | less  # check that the build folders are clean
    twine check dist/*

Finally upload the distribution::

    twine upload dist/*2.4.x*  # to not upload outdated stuff

and (possibly) tag the version::

    git tag -a cocopp-2.3.x.xx -m "cocopp-2.3.x.xx uploaded to PyPI"

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

_name = 'cocopp'

try:
    from subprocess import check_output
except ImportError:
    import subprocess
    def check_output(*popenargs, **kwargs):
        r"""Run command with arguments and return its output as a byte string.
        Backported from Python 2.7 as it's implemented as pure python on stdlib.
        >>> check_output(['/usr/bin/python', '--version'])
        Python 2.6.2

        WARNING: This method is also defined in ../../code-postprocessing/cocopp/toolsdivers.py.
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

def git(args):
    """Run a git command and return its output.

    All errors are deemed fatal and the system will quit.

    WARNING: This method is also defined in ../../code-postprocessing/cocopp/toolsdivers.py.
    If you change something you have to change it in both files.
    """
    full_command = ['git']
    full_command.extend(args)
    try:
        output = check_output(full_command, env=os.environ,
                              universal_newlines=True)
        output = output.rstrip()
    except CalledProcessError as e:
        # print('Failed to execute "%s"' % str(full_command))
        raise
    return output

def git_version(pep440=False):
    """Return somewhat readible version number from git, like
    '0.1-6015-ga0a3769' if not pep440 else '0.1.6015'

    """
    # res = git(['describe', '--tags'])
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

def git_version_print(*args, **kwargs):
    res = git_version(*args, **kwargs)
    print("git_version:", res)
    return res

try:
    long_description = open('README.md', 'r').read()
except:
    long_description = "no long description found"

setup(
    name = _name,
    version = git_version_print(pep440=True),
    packages = [_name, _name + '.comp2', _name + '.compall'],
    package_dir = {_name: 'cocopp'},
    package_data={_name: ['*enchmarkshortinfos.txt',
                          '*enchmarkinfos.txt',
                          'best*algentries*.pickle',
                          'best*algentries*.pickle.gz',
                          'refalgs/best*.tar.gz',
                          'pprldistr2009*.pickle.gz',
                          'latex_commands_for_html.html',
    # this is not supposed to copy to the subfolder, see https://docs.python.org/2/distutils/setupscript.html
    # but it does.
                          'js/*', 'tth/*',
                          '../latex-templates/*.tex',
                          '../latex-templates/*.cls',
                          '../latex-templates/*.sty',
                          '../latex-templates/*.bib',
                          '../latex-templates/*.bst',
                          ]},
    url = 'https://github.com/numbbo/coco',
    license = 'BSD',
    maintainer = 'Dimo Brockhoff and Nikolaus Hansen',
    maintainer_email = 'dimo.brockhoff@inria.fr',
    # author = ['Nikolaus Hansen', 'Raymond Ros', 'Dejan Tusar'],
    description = 'Benchmarking framework for all types of black-box optimization algorithms, postprocessing. ',
    long_description = long_description,
    long_description_content_type =  'text/markdown',  # 'text/x-rst', # html doesn't exist,
    # THIS BREAKS TESTS BUT SHOULD BE OUTCOMMENTED TO MAKE A DISTRIBUTION:
    # install_requires = ['numpy>=1.7', 'matplotlib>=3.1'],
    classifiers = [
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Other Audience",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        # "Programming Language :: Python :: 2.6",
        # "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Framework :: IPython",
        "Framework :: Jupyter",
        "License :: OSI Approved :: BSD License"
    ]
)
