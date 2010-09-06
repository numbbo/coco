#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Routine to organize the output folder in which runcompall puts all the output.
"""

import os
import sys
import glob
from pdb import set_trace

def do(dirnames):
    """ moves images into different folders, only the overall RL distributions remain in the root folder"""
    if not hasattr(dirnames, '__iter__'):
        dirnames = (dirnames, )
    for dirname in dirnames:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        filenames = []
        filenames.append(glob.glob(os.path.join(dirname, '*_f*')))
        filenames.append(glob.glob(os.path.join(dirname, '*_ert*')))
        filenames.append(glob.glob(os.path.join(dirname, '*allerts.*')))
        #filenames is a list of 3 lists of files
        filedst = ('FIXED-F', 'FIXED-RT', 'ERT')
        for i, filen in enumerate(filenames):
            if filen: # is not empty
                foldername = os.path.join(dirname, filedst[i])
                if not os.path.exists(foldername):
                    os.makedirs(foldername)
                for f in filen:
                    try:
                        dst = os.path.join(foldername, os.path.split(f)[-1])
                        os.rename(f, dst)
                    except OSError:
                        # Either dst is a folder or the OS is Windows and dst exists
                        print "%s cannot be moved to %s.\n" % (f, dst)

if 1 < 3 and __name__ == "__main__":
    # Input: folders where to apply above method do
    if len(sys.argv) < 2: 
        print 'Need at least one directory argument, where pictures are going to be organized into folders'
    do(sys.argv[1:])
