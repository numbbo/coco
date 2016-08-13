#! /usr/bin/env python 
"""Replaces a string with another string, or a line starting with a string with 
another line, in a bunch of given files.   

Needs at least three arguments: oldstring, newstring and file(s). 
Examples:

    python foreachfile.py '5,5' '5;5' *.info
    python foreachfile.py line.startswith._debugging "_debugging = False"

In the latter case entire lines starting with "_degugging" are replaced 
(white spaces are not ignored, for the time being)
"""

from __future__ import absolute_import, print_function
import os, sys
from subprocess import call, check_output, CalledProcessError


def condition1(old, line):
    return old in line
def change1(line, old, new):
    """replace old with new if old in line. """
    return line.replace(old, new)

def condition2(old, line):
    return line.startswith(old)
def change2(line, old, new):
    """replace line with new if line.startswith(old)"""
    if line.startswith(old):
        return new + "\n"
    return line

condition = condition1
change = change1

def main(old, new, *files):
    global condition
    global change
    if old.startswith("line.startswith."):
        condition = condition2  # effects only console output
        change = change2
        old = old[16:]
        print('replace lines starting with "%s" with "%s"' % (old, new))
    else:
        print('replacing ' + old + ' with ' + new)

    counter = 0
    found = 0
    p = os.path
    for filename in files:
        # print(filename)
        counter += 1
        tfilename = p.join(p.dirname(filename), '__tmp__' + 
                           p.split(filename)[-1] + '__tmp__');
        if os.path.isfile(tfilename):
            os.remove(tfilename) # deal with rename on windows
        os.rename(filename, tfilename)
        with open(filename, 'a') as fp: # a is just in case
            for line in open(tfilename):
                if condition(old, line):
                    found += 1
                fp.write(change(line, old, new))
        sys.stdout.flush()  # for print
    print(counter, 'files visited,', found, 'times replaced')

if __name__ == "__main__":
    if len(sys.argv) < 4: 
        print(__doc__)
    else:
        main(*sys.argv[1:])
