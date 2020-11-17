#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys
import getopt
from shutil import move
from os import remove

"""Modify meta information in raw experimental data.

While comparing algorithms with the cocopp package, it is sometimes
needed to change the algorithm name (given as `algId` or `algorithm` 
in the :file`.info` files) or the algorithm comments after a run is
already finished (for example because two output folders contain results
for two different algorithms but with the same name). This script allows
to change these within a specified output folder.

written: db 28/01/2010
         db 26/06/2013 corrected documentation
         db 19/06/2017 updated docstrings, allow bbob-biobj(-ext) data sets
         db 16/11/2020 finally moved to python 3 :-)

"""

__all__ = ['main']


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def usage():
    print(main.__doc__)


def main(argv=None):
    """Main routine.

    This script allows to change algorithm name (algId/algorithm) and algorithm
    comment after a run finished, i.e., after an output folder has been
    created.

    :param seq argv: list of strings containing options and arguments.
                     If not provided, sys.argv is accessed.

    :py:data:`argv` should list an output folder (first argument) and
    additionally an algorithm name (2nd argument) and the algorithm
    comment (3rd argument).
    If only the output folder is given, the script asks for an algorithm
    name and a comment interactively.

        -h, --help

            display this message

    Examples:

    * Changing algorithm name and comments for given output folder from the
       command line::

        >> python changeAlgIdAndComment.py outfolder "CMA-ES" "CMA_with_lambda_100"

    * Changing algorithm name and comments for given output folder
       interactively::

        >> python changeAlgIdAndComment.py outputfolder

    """

    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, "h",
                                       ["help"])
        except:
            raise Usage(sys.exc_info()[0])

        if not args:
            usage()
            sys.exit()

        # Process options
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                sys.exit()
            else:
                assert False, "unhandled option"

        # check if all arguments are there and ask for them if not:
        if len(args) < 3:
            if len(args) < 2:
                name = input("You forgot to specify an algorithm name. " +
                             "Please enter one (algId):")
                args.append(name)
            comment = input("You forgot to specify a comment. Please " +
                            "enter one for algorithm " + args[1] + ":")
            args.append(comment)
        folder = args[0]
        # make sure that folder name ends with a '/' to be able to append
        # the file names afterwards
        if not folder.endswith('/'):
            folder = folder + '/'

        algId = args[1]
        comment = args[2]

        if not os.path.exists(folder):
            print("ERROR: folder " + folder + " does not exist!")
            sys.exit()
        if not os.path.isdir(folder):
            print("ERROR: " + folder + " is not a directory")
            sys.exit()

        # get all .info files in folder:
        FILES = []
        for (path, dirs, files) in os.walk(folder):
            for fname in files:
                if fname.endswith('.info'):
                    FILES.append(os.path.join(path, fname))

        for file in FILES:
            # open file to read and temp file to write
            infile = open(file, 'r')
            tempfile = open('temp.temp', 'w')
            while infile:
                line = infile.readline()
                if not line:
                    break

                # make sure that everything is copied:
                newline = line
                # check if something needs to be changed:
                if line.find('algId =') >= 0 or line.find('algorithm =') >= 0:
                    s = line.split(', ')
                    for i, word in enumerate(s):
                        if word.find('algId') >= 0:
                            # replace algId:
                            s[i] = "algId = '" + algId + "'"
                            if len(s) == i+1:
                                # algId or algorithm last entry in this line
                                s[i] = s[i] + "\n"
                        elif word.find('algorithm') >= 0:
                            # replace algId:
                            s[i] = "algorithm = '" + algId + "'"
                            if len(s) == i+1:
                                # algId or algorithm last entry in this line
                                s[i] = s[i] + "\n"
                    newline = ", ".join(s)

                else:
                    s = line.split()
                    if s[0] == '%':
                        newline = "% " + comment + "\n"

                tempfile.write(newline)

            infile.close()
            tempfile.close()
            # remove old file and rename temp file accordingly
            remove(file)
            move('temp.temp', file)

            print(file + " changed")

        sys.exit()

    except IOError as e:
        errno, strerror = e.args
        print('>> ' + strerror + e.msg)
        print('>> ' + errno)
        print('for help use -h or --help')
        return 2


if __name__ == "__main__":
    sys.exit(main())

