#! /usr/bin/env python
"""tentative make replacement to compile coco.c and example_experiment.c

Usage::

    python tentative-make.py
    python tentative-make.py clean
"""
# TODO:
# DONE:

from __future__ import division, print_function, absolute_import
import os, sys, time, shutil
import subprocess
import numpy as np

dry_run = False

# ============== VARIABLES ==============
substitutes = dict(
    CC = os.environ.get('CC', 'cc'),
    CCFLAGS = '-g -ggdb -std=c99 -pedantic -Wall -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare -Wconversion',
    LDFLAGS = os.environ.get('LDFLAGS', '') + ' -lm'
)

# ============== MAKE DEPENDENCY LIST ==============
make_dependencies = {
    'example_experiment': [['coco.o', 'example_experiment.o'],
         ['${CC} -o example_experiment coco.o example_experiment.o ${LDFLAGS}']],
    'coco.o': [['coco.h', 'coco.c'],
        ['${CC} -c ${CCFLAGS} -o coco.o coco.c']],
    'example_experiment.o': [['coco.h', 'coco.c', 'example_experiment.c'],
        ['${CC} -c ${CCFLAGS} -o example_experiment.o example_experiment.c']]
    }


# ============== CODE ==============
def substitute(cmd, substitutes=substitutes):
    """return `substitutes[something]` if `cmd == "${something}"` else `cmd`"""
    cmd = cmd.strip()
    if cmd.startswith('${'):
        try:
            return substitutes[cmd[2:-1]]
        except KeyError:
            print('WARNING: nothing found to substitute "%s"' % cmd)
    return cmd


def call(cmd, show_cmd=True, show_res=True):
    """call system command returning the result of...

    either `os.system` if `show_cmd`, or `subprocess.Popen`
    otherwise.

    `"${something}"` is substituted by `substitute("${something}")`.
    """
    if str(cmd) == cmd:
        cmd = cmd.split()
    # substitute
    for i in range(len(cmd)):
        cmd[i] = substitute(cmd[i])
    if show_cmd or dry_run:
        print(' '.join(cmd))
    # call
    if not dry_run:
        if show_res:
            return os.system(' '.join(cmd))
        else:
            return subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE).communicate()[0]


def newer(source, target):
    """return `False` if creation _and_ change time of `source`
    is successfully tested and found to be older than the one of
    `target`, else `True`. """
    try:
        if os.path.getmtime(source) < os.path.getmtime(target) or \
           os.path.getctime(source) < os.path.getctime(target):
            return False
    except:
        return True
    return True


def fake_make(target, dependencies=make_dependencies,
         message="make.py: target is up to date"):
    """replicate behavior of make for `target`"""
    for sub_target in dependencies[target][0]:
        if sub_target in dependencies:
            fake_make(sub_target, message="")
    if any(newer(dep, target) for dep in dependencies[target][0]):
        for cmd in dependencies[target][1]:
            call(cmd)
    elif message:
        print(message.replace('target', 'target "%s"' % target))


def clean(*args):
    for arg in args:
        try:
            os.remove(arg) if not dry_run else \
                print("DRY_RUN: os.remove(%s)" % arg)
        except:
            pass
            # print('file "%s" could not be removed' % arg)


if __name__ == "__main__":
    args = sys.argv[1:]  # remove script name from list
    if len(args) > 0 and sys.argv[0] in ('-h', '--help'):
        print(__doc__)
        sys.exit()

# ============== GLOBAL MAKE TARGETS ==============
    if len(args) == 0 or args[0] == 'all':
        fake_make('example_experiment', make_dependencies)
    elif args[0] == 'clean':
        clean('coco.o', 'example_experiment.o', 'example_experiment')
    else:
        print(__doc__)
