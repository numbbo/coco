#! /usr/bin/env python

"""Routine to organize the output folder in which runcompall puts all the output.
"""

import os
import sys
import shutil
import glob
from pdb import set_trace

def do(dirnames): 
    """ moves images into different folders, only the overall RL distributions remain in the root folder"""
    if not hasattr(dirnames, '__iter__'):
        dirnames = (dirnames, )
    for dirname in dirnames:
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        fixedf = glob.glob(os.path.join(dirname, '*_f*'))
        fixedrt = glob.glob(os.path.join(dirname, '*_ert*'))
        ert = glob.glob(os.path.join(dirname, '*allerts.*'))
        if fixedf:
            foldername = os.path.join(dirname, 'FIXED-F')
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            for file in fixedf:
                shutil.move(file, foldername)
        if fixedrt:
            foldername = os.path.join(dirname, 'FIXED-RT')
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            for file in fixedrt:
                shutil.move(file, foldername)
        if ert:
            foldername = os.path.join(dirname, 'ERT')
            if not os.path.exists(foldername):
                os.makedirs(foldername)
            for file in ert:
                shutil.move(file, foldername)

if 1 < 3 and __name__ == "__main__":
    # Input: folders where to apply above method do
    if len(sys.argv) < 2: 
        print 'Need at least one directory argument, where pictures are going to be organized into folders'
    do(sys.argv[1:])
