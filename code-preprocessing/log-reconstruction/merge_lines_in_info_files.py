#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Merges lines with the same file link in .info files such that the
# postprocessing works correctly.
#
# The code takes a folder as argument in which all .info files are found
# and in which the instances on different lines that start with the same
# function, dimension, and filename, are put on a single line.
#
# This script is only needed for data sets that have been reconstructed
# from the archive and the experiments of which have written the data for
# different instances of the same function/dimension pair into separate
# folders but the same .dat files.
#
# written by Dimo Brockhoff 2016

from __future__ import absolute_import, division, print_function, unicode_literals
try: range = xrange  # let range always be an iterator
except NameError: pass
import numpy as np  # "pip install numpy" installs numpy
import sys
import shutil

import bbob_pproc
from bbob_pproc import findfiles


def merge_lines_in(filename, inputfolder, outputfolder):
    towrite = ""
    instancedict = {} # will finally contain string with all instances for each
                      # function/dimension/filename pair
    with open(filename) as f:
        for line in f:
            if "function" not in line:
                towrite = towrite + line
            else:
                splitline = line.split()
                if (splitline[2], splitline[5], splitline[6]) in instancedict:
                    instancedict[(splitline[2], splitline[5], splitline[6])] = (
                        (instancedict[(splitline[2], splitline[5], splitline[6])]).replace('\n', '')
                        + line.split(".dat")[1].replace('\n', ''))
                else:
                    instancedict[(splitline[2], splitline[5], splitline[6])] = (
                        line.split(".dat,")[1].replace('\n', ''))

    outputfilename = outputfolder + filename.split(inputfolder)[1]
    with open(outputfilename, 'w') as f:
        f.write(towrite)
        for idx, key in enumerate(instancedict):
            f.write("function =  %s dim =  %s %s%s\n" % (key[0], key[1], key[2], instancedict[key]))
    
        

if __name__ == '__main__':
    """Merges lines of .info files that contain different instances but the
       same filename.

       Reconstructs the .info files within a given folder and writes everything
       into a clean new folder with the given output folder name.
    """
    
    if not len(sys.argv) == 3:
        print(r'Usage:\n python merge_lines_in_info_files.py FOLDERNAME OUTPUTFOLDERNAME')
    else:
        inputfolder = sys.argv[1]
        outputfolder = sys.argv[2]
        try:
            shutil.copytree(inputfolder, outputfolder)
        except:
            print("Problem while copying folder %s to %s" % (inputfolder, outputfolder))
            e = sys.exc_info()[0]
            print("   Error: %s" % e)
        
        filelist = findfiles.main(sys.argv[1])
        for f in filelist:
            print("Processing %s..." % f)
            merge_lines_in(f, inputfolder, outputfolder)