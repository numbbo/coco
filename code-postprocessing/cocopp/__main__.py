#!/usr/bin/env python
"""Calls rungeneric.py.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import sys

try:
    from . import rungeneric
    from . import genericsettings
    is_module = True
except:
    is_module = False
import matplotlib  # just to make sure the following is actually done first

matplotlib.use('Agg')  # To avoid window popup and use without X forwarding


def main():
    r"""Currently it does nothing.

    """

if __name__ == "__main__":
    """run either tests or rungeneric.main"""
    args = sys.argv[1:] if len(sys.argv) else []
    if not is_module:
        raise ValueError('try calling "python -m ..." instead of "python ..."')
    res = rungeneric.main(args)
    if genericsettings.test:
        print(res)
