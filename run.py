#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Main module for post-processing the data of one algorithm.
   Calls the main function of bbob_pproc with arguments from the
   command line. Executes the BBOB postprocessing on the given
   filename and folder arguments, using all found .info files.
Synopsis:
    python path_to_folder/bbob_pproc/run.py [OPTIONS] FILE_NAME FOLDER_NAME...
Help:
    python path_to_folder/bbob_pproc/run.py -h

"""

from __future__ import absolute_import

import os
import sys
import warnings
import getopt
from pdb import set_trace
import matplotlib
matplotlib.use('Agg') # To avoid window popup and use without X forwarding

# Add the path to bbob_pproc
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc import pptex, findfiles
from bbob_pproc.bbob2010 import pprldistr, ppfigdim, pplogloss
from bbob_pproc.pproc import DataSetList

import matplotlib.pyplot as plt

# GLOBAL VARIABLES used in the routines defining desired output  for BBOB 2009.
instancesOfInterest = {1:3, 2:3, 3:3, 4:3, 5:3}
instancesOfInterest2010 = {1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1,
                           11:1, 12:1, 13:1, 14:1, 15:1}

# function-dependent target function values: hard coded here before we come up
# with something smarter. It is supposed the number of level of difficulties
# are the same, it is just the target function value that differs.

tabDimsOfInterest = (5, 20)    # dimension which are displayed in the tables
# tabValsOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
#tabValsOfInterest = (10, 1.0, 1e-1, 1e-3, 1e-5, 1.0e-8)
tabValsOfInterest = ({1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10,
                      9: 10, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10,
                      16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10,
                      23: 10, 24: 10, 101: 10, 102: 10, 103: 10, 104: 10,
                      105: 10, 106: 10, 107: 10, 108: 10, 109: 10, 110: 10,
                      111: 10, 112: 10, 113: 10, 114: 10, 115: 10, 116: 10,
                      117: 10, 118: 10, 119: 10, 120: 10, 121: 10, 122: 10,
                      123: 10, 124: 10, 125: 10, 126: 10, 127: 10, 128: 10,
                      129: 10, 130: 10},
                     {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
                      8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0,
                      14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0,
                      20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 101: 1.0,
                      102: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0,
                      107: 1.0, 108: 1.0, 109: 1.0, 110: 1.0, 111: 1.0,
                      112: 1.0, 113: 1.0, 114: 1.0, 115: 1.0, 116: 1.0,
                      117: 1.0, 118: 1.0, 119: 1.0, 120: 1.0, 121: 1.0,
                      122: 1.0, 123: 1.0, 124: 1.0, 125: 1.0, 126: 1.0,
                      127: 1.0, 128: 1.0, 129: 1.0, 130: 1.0},
                      {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1,
                       8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1, 12: 0.1, 13: 0.1,
                       14: 0.1, 15: 0.1, 16: 0.1, 17: 0.1, 18: 0.1, 19: 0.1,
                       20: 0.1, 21: 0.1, 22: 0.1, 23: 0.1, 24: 0.1, 101: 0.1,
                       102: 0.1, 103: 0.1, 104: 0.1, 105: 0.1, 106: 0.1,
                       107: 0.1, 108: 0.1, 109: 0.1, 110: 0.1, 111: 0.1,
                       112: 0.1, 113: 0.1, 114: 0.1, 115: 0.1, 116: 0.1,
                       117: 0.1, 118: 0.1, 119: 0.1, 120: 0.1, 121: 0.1,
                       122: 0.1, 123: 0.1, 124: 0.1, 125: 0.1, 126: 0.1,
                       127: 0.1, 128: 0.1, 129: 0.1, 130: 0.1},
                      {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001, 5: 0.001,
                       6: 0.001, 7: 0.001, 8: 0.001, 9: 0.001, 10: 0.001,
                       11: 0.001, 12: 0.001, 13: 0.001, 14: 0.001, 15: 0.001,
                       16: 0.001, 17: 0.001, 18: 0.001, 19: 0.001, 20: 0.001,
                       21: 0.001, 22: 0.001, 23: 0.001, 24: 0.001, 101: 0.001,
                       102: 0.001, 103: 0.001, 104: 0.001, 105: 0.001,
                       106: 0.001, 107: 0.001, 108: 0.001, 109: 0.001,
                       110: 0.001, 111: 0.001, 112: 0.001, 113: 0.001,
                       114: 0.001, 115: 0.001, 116: 0.001, 117: 0.001,
                       118: 0.001, 119: 0.001, 120: 0.001, 121: 0.001,
                       122: 0.001, 123: 0.001, 124: 0.001, 125: 0.001,
                       126: 0.001, 127: 0.001, 128: 0.001, 129: 0.001,
                       130: 0.001},
                      {1: 1e-05, 2: 1e-05, 3: 1e-05, 4: 1e-05, 5: 1e-05,
                       6: 1e-05, 7: 1e-05, 8: 1e-05, 9: 1e-05, 10: 1e-05,
                       11: 1e-05, 12: 1e-05, 13: 1e-05, 14: 1e-05, 15: 1e-05,
                       16: 1e-05, 17: 1e-05, 18: 1e-05, 19: 1e-05, 20: 1e-05,
                       21: 1e-05, 22: 1e-05, 23: 1e-05, 24: 1e-05, 101: 1e-05,
                       102: 1e-05, 103: 1e-05, 104: 1e-05, 105: 1e-05,
                       106: 1e-05, 107: 1e-05, 108: 1e-05, 109: 1e-05,
                       110: 1e-05, 111: 1e-05, 112: 1e-05, 113: 1e-05,
                       114: 1e-05, 115: 1e-05, 116: 1e-05, 117: 1e-05,
                       118: 1e-05, 119: 1e-05, 120: 1e-05, 121: 1e-05,
                       122: 1e-05, 123: 1e-05, 124: 1e-05, 125: 1e-05,
                       126: 1e-05, 127: 1e-05, 128: 1e-05, 129: 1e-05,
                       130: 1e-05},
                      {1: 1e-08, 2: 1e-08, 3: 1e-08, 4: 1e-08, 5: 1e-08,
                       6: 1e-08, 7: 1e-08, 8: 1e-08, 9: 1e-08, 10: 1e-08,
                       11: 1e-08, 12: 1e-08, 13: 1e-08, 14: 1e-08, 15: 1e-08,
                       16: 1e-08, 17: 1e-08, 18: 1e-08, 19: 1e-08, 20: 1e-08,
                       21: 1e-08, 22: 1e-08, 23: 1e-08, 24: 1e-08, 101: 1e-08,
                       102: 1e-08, 103: 1e-08, 104: 1e-08, 105: 1e-08,
                       106: 1e-08, 107: 1e-08, 108: 1e-08, 109: 1e-08,
                       110: 1e-08, 111: 1e-08, 112: 1e-08, 113: 1e-08,
                       114: 1e-08, 115: 1e-08, 116: 1e-08, 117: 1e-08,
                       118: 1e-08, 119: 1e-08, 120: 1e-08, 121: 1e-08,
                       122: 1e-08, 123: 1e-08, 124: 1e-08, 125: 1e-08,
                       126: 1e-08, 127: 1e-08, 128: 1e-08, 129: 1e-08,
                       130: 1e-08})

#figValsOfInterest = (10, 1e-1, 1e-4, 1e-8)
#figValsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8)
figValsOfInterest = ({1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10,
                      9: 10, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10,
                      16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10,
                      23: 10, 24: 10, 101: 10, 102: 10, 103: 10, 104: 10,
                      105: 10, 106: 10, 107: 10, 108: 10, 109: 10, 110: 10,
                      111: 10, 112: 10, 113: 10, 114: 10, 115: 10, 116: 10,
                      117: 10, 118: 10, 119: 10, 120: 10, 121: 10, 122: 10,
                      123: 10, 124: 10, 125: 10, 126: 10, 127: 10, 128: 10,
                      129: 10, 130: 10},
                     {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0,
                      8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0,
                      14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0,
                      20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 101: 1.0,
                      102: 1.0, 103: 1.0, 104: 1.0, 105: 1.0, 106: 1.0,
                      107: 1.0, 108: 1.0, 109: 1.0, 110: 1.0, 111: 1.0,
                      112: 1.0, 113: 1.0, 114: 1.0, 115: 1.0, 116: 1.0,
                      117: 1.0, 118: 1.0, 119: 1.0, 120: 1.0, 121: 1.0,
                      122: 1.0, 123: 1.0, 124: 1.0, 125: 1.0, 126: 1.0,
                      127: 1.0, 128: 1.0, 129: 1.0, 130: 1.0},
                      {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1,
                       8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1, 12: 0.1, 13: 0.1,
                       14: 0.1, 15: 0.1, 16: 0.1, 17: 0.1, 18: 0.1, 19: 0.1,
                       20: 0.1, 21: 0.1, 22: 0.1, 23: 0.1, 24: 0.1, 101: 0.1,
                       102: 0.1, 103: 0.1, 104: 0.1, 105: 0.1, 106: 0.1,
                       107: 0.1, 108: 0.1, 109: 0.1, 110: 0.1, 111: 0.1,
                       112: 0.1, 113: 0.1, 114: 0.1, 115: 0.1, 116: 0.1,
                       117: 0.1, 118: 0.1, 119: 0.1, 120: 0.1, 121: 0.1,
                       122: 0.1, 123: 0.1, 124: 0.1, 125: 0.1, 126: 0.1,
                       127: 0.1, 128: 0.1, 129: 0.1, 130: 0.1},
                      {1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01,
                       6: 0.01, 7: 0.01, 8: 0.01, 9: 0.01, 10: 0.01,
                       11: 0.01, 12: 0.01, 13: 0.01, 14: 0.01, 15: 0.01,
                       16: 0.01, 17: 0.01, 18: 0.01, 19: 0.01, 20: 0.01,
                       21: 0.01, 22: 0.01, 23: 0.01, 24: 0.01, 101: 0.01,
                       102: 0.01, 103: 0.01, 104: 0.01, 105: 0.01,
                       106: 0.01, 107: 0.01, 108: 0.01, 109: 0.01,
                       110: 0.01, 111: 0.01, 112: 0.01, 113: 0.01,
                       114: 0.01, 115: 0.01, 116: 0.01, 117: 0.01,
                       118: 0.01, 119: 0.01, 120: 0.01, 121: 0.01,
                       122: 0.01, 123: 0.01, 124: 0.01, 125: 0.01,
                       126: 0.01, 127: 0.01, 128: 0.01, 129: 0.01,
                       130: 0.01},
                      {1: 0.001, 2: 0.001, 3: 0.001, 4: 0.001, 5: 0.001,
                       6: 0.001, 7: 0.001, 8: 0.001, 9: 0.001, 10: 0.001,
                       11: 0.001, 12: 0.001, 13: 0.001, 14: 0.001, 15: 0.001,
                       16: 0.001, 17: 0.001, 18: 0.001, 19: 0.001, 20: 0.001,
                       21: 0.001, 22: 0.001, 23: 0.001, 24: 0.001, 101: 0.001,
                       102: 0.001, 103: 0.001, 104: 0.001, 105: 0.001,
                       106: 0.001, 107: 0.001, 108: 0.001, 109: 0.001,
                       110: 0.001, 111: 0.001, 112: 0.001, 113: 0.001,
                       114: 0.001, 115: 0.001, 116: 0.001, 117: 0.001,
                       118: 0.001, 119: 0.001, 120: 0.001, 121: 0.001,
                       122: 0.001, 123: 0.001, 124: 0.001, 125: 0.001,
                       126: 0.001, 127: 0.001, 128: 0.001, 129: 0.001,
                       130: 0.001},
                      {1: 1e-05, 2: 1e-05, 3: 1e-05, 4: 1e-05, 5: 1e-05,
                       6: 1e-05, 7: 1e-05, 8: 1e-05, 9: 1e-05, 10: 1e-05,
                       11: 1e-05, 12: 1e-05, 13: 1e-05, 14: 1e-05, 15: 1e-05,
                       16: 1e-05, 17: 1e-05, 18: 1e-05, 19: 1e-05, 20: 1e-05,
                       21: 1e-05, 22: 1e-05, 23: 1e-05, 24: 1e-05, 101: 1e-05,
                       102: 1e-05, 103: 1e-05, 104: 1e-05, 105: 1e-05,
                       106: 1e-05, 107: 1e-05, 108: 1e-05, 109: 1e-05,
                       110: 1e-05, 111: 1e-05, 112: 1e-05, 113: 1e-05,
                       114: 1e-05, 115: 1e-05, 116: 1e-05, 117: 1e-05,
                       118: 1e-05, 119: 1e-05, 120: 1e-05, 121: 1e-05,
                       122: 1e-05, 123: 1e-05, 124: 1e-05, 125: 1e-05,
                       126: 1e-05, 127: 1e-05, 128: 1e-05, 129: 1e-05,
                       130: 1e-05},
                      {1: 1e-08, 2: 1e-08, 3: 1e-08, 4: 1e-08, 5: 1e-08,
                       6: 1e-08, 7: 1e-08, 8: 1e-08, 9: 1e-08, 10: 1e-08,
                       11: 1e-08, 12: 1e-08, 13: 1e-08, 14: 1e-08, 15: 1e-08,
                       16: 1e-08, 17: 1e-08, 18: 1e-08, 19: 1e-08, 20: 1e-08,
                       21: 1e-08, 22: 1e-08, 23: 1e-08, 24: 1e-08, 101: 1e-08,
                       102: 1e-08, 103: 1e-08, 104: 1e-08, 105: 1e-08,
                       106: 1e-08, 107: 1e-08, 108: 1e-08, 109: 1e-08,
                       110: 1e-08, 111: 1e-08, 112: 1e-08, 113: 1e-08,
                       114: 1e-08, 115: 1e-08, 116: 1e-08, 117: 1e-08,
                       118: 1e-08, 119: 1e-08, 120: 1e-08, 121: 1e-08,
                       122: 1e-08, 123: 1e-08, 124: 1e-08, 125: 1e-08,
                       126: 1e-08, 127: 1e-08, 128: 1e-08, 129: 1e-08,
                       130: 1e-08})

rldDimsOfInterest = (5, 20)
#rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8)
rldValsOfInterest = ({1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10,
                      9: 10, 10: 10, 11: 10, 12: 10, 13: 10, 14: 10, 15: 10,
                      16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 10, 22: 10,
                      23: 10, 24: 10, 101: 10, 102: 10, 103: 10, 104: 10,
                      105: 10, 106: 10, 107: 10, 108: 10, 109: 10, 110: 10,
                      111: 10, 112: 10, 113: 10, 114: 10, 115: 10, 116: 10,
                      117: 10, 118: 10, 119: 10, 120: 10, 121: 10, 122: 10,
                      123: 10, 124: 10, 125: 10, 126: 10, 127: 10, 128: 10,
                      129: 10, 130: 10},
                      {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1,
                       8: 0.1, 9: 0.1, 10: 0.1, 11: 0.1, 12: 0.1, 13: 0.1,
                       14: 0.1, 15: 0.1, 16: 0.1, 17: 0.1, 18: 0.1, 19: 0.1,
                       20: 0.1, 21: 0.1, 22: 0.1, 23: 0.1, 24: 0.1, 101: 0.1,
                       102: 0.1, 103: 0.1, 104: 0.1, 105: 0.1, 106: 0.1,
                       107: 0.1, 108: 0.1, 109: 0.1, 110: 0.1, 111: 0.1,
                       112: 0.1, 113: 0.1, 114: 0.1, 115: 0.1, 116: 0.1,
                       117: 0.1, 118: 0.1, 119: 0.1, 120: 0.1, 121: 0.1,
                       122: 0.1, 123: 0.1, 124: 0.1, 125: 0.1, 126: 0.1,
                       127: 0.1, 128: 0.1, 129: 0.1, 130: 0.1},
                      {1: 1e-04, 2: 1e-04, 3: 1e-04, 4: 1e-04, 5: 1e-04,
                       6: 1e-04, 7: 1e-04, 8: 1e-04, 9: 1e-04, 10: 1e-04,
                       11: 1e-04, 12: 1e-04, 13: 1e-04, 14: 1e-04, 15: 1e-04,
                       16: 1e-04, 17: 1e-04, 18: 1e-04, 19: 1e-04, 20: 1e-04,
                       21: 1e-04, 22: 1e-04, 23: 1e-04, 24: 1e-04, 101: 1e-04,
                       102: 1e-04, 103: 1e-04, 104: 1e-04, 105: 1e-04,
                       106: 1e-04, 107: 1e-04, 108: 1e-04, 109: 1e-04,
                       110: 1e-04, 111: 1e-04, 112: 1e-04, 113: 1e-04,
                       114: 1e-04, 115: 1e-04, 116: 1e-04, 117: 1e-04,
                       118: 1e-04, 119: 1e-04, 120: 1e-04, 121: 1e-04,
                       122: 1e-04, 123: 1e-04, 124: 1e-04, 125: 1e-04,
                       126: 1e-04, 127: 1e-04, 128: 1e-04, 129: 1e-04,
                       130: 1e-04},
                      {1: 1e-08, 2: 1e-08, 3: 1e-08, 4: 1e-08, 5: 1e-08,
                       6: 1e-08, 7: 1e-08, 8: 1e-08, 9: 1e-08, 10: 1e-08,
                       11: 1e-08, 12: 1e-08, 13: 1e-08, 14: 1e-08, 15: 1e-08,
                       16: 1e-08, 17: 1e-08, 18: 1e-08, 19: 1e-08, 20: 1e-08,
                       21: 1e-08, 22: 1e-08, 23: 1e-08, 24: 1e-08, 101: 1e-08,
                       102: 1e-08, 103: 1e-08, 104: 1e-08, 105: 1e-08,
                       106: 1e-08, 107: 1e-08, 108: 1e-08, 109: 1e-08,
                       110: 1e-08, 111: 1e-08, 112: 1e-08, 113: 1e-08,
                       114: 1e-08, 115: 1e-08, 116: 1e-08, 117: 1e-08,
                       118: 1e-08, 119: 1e-08, 120: 1e-08, 121: 1e-08,
                       122: 1e-08, 123: 1e-08, 124: 1e-08, 125: 1e-08,
                       126: 1e-08, 127: 1e-08, 128: 1e-08, 129: 1e-08,
                       130: 1e-08})
#Put backward to have the legend in the same order as the lines.

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


#FUNCTION DEFINITIONS

def usage():
    print main.__doc__


def main(argv=None):
    """Generates from BBOB experiment data some outputs for a tex document.

    Provided with some index entries (found in files with the 'info' extension)
    this routine outputs figure and TeX files in the folder 'ppdata' needed for
    the compilation of  latex document templateBBOBarticle.tex. These output
    files will contain performance tables, performance scaling figures and
    empirical cumulative distribution figures. On subsequent executions, new
    files will be added to the output directory, overwriting existing older
    files in the process.

    Keyword arguments:
    argv -- list of strings containing options and arguments. If not given,
    sys.argv is accessed.

    argv should list either names of info files or folders containing info
    files. argv can also contain post-processed pickle files generated by this
    routine. Furthermore, argv can begin with, in any order, facultative option
    flags listed below.

        -h, --help

            display this message

        -v, --verbose

            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

        -p, --pickle

            generates pickle post processed data files.

        -o, --output-dir OUTPUTDIR

            change the default output directory ('ppdata') to OUTPUTDIR

        --crafting-effort=VALUE

            sets the crafting effort to VALUE. Otherwise the user will be
            prompted. This flag is useful when running this script in batch.

        -f, --final

            lengthens the bootstrapping process used as dispersion measure in
            the tables generation. This might at least double the time of the
            whole post-processing. Please use this option when generating your
            final paper.

        --tab-only, --fig-only, --rld-only, --los-only

            these options can be used to output respectively the tex tables,
            convergence and ENFEs graphs figures, run length distribution
            figures, ERT loss ratio figures only. A combination of any two of
            these options results in no output.

    Exceptions raised:
    Usage -- Gives back a usage message.

    Examples:

    * Calling the run.py interface from the command line:

        $ python bbob_pproc/run.py -v experiment1

    will post-process the folder experiment1 and all its containing data,
    base on the found .info files in the folder. The result will appear
    in folder ppdata. The -v option adds verbosity.

        $ python bbob_pproc/run.py -o otherppdata experiment2/*.info

    This will execute the post-processing on the info files found in
    experiment2. The result will be located in the alternative location
    otherppdata.

    * Loading this package and calling the main from the command line
      (requires that the path to this package is in python search path):

        $ python -m bbob_pproc -h

    This will print out this help message.

    * From the python interactive shell (requires that the path to this
      package is in python search path):

        >>> import bbob_pproc
        >>> bbob_pproc.main('-o outputfolder folder1'.split())

    This will execute the post-processing on the index files found in folder1.
    The -o option changes the output folder from the default ppdata to
    outputfolder.

    """

    if argv is None:
        argv = sys.argv[1:]
        # The zero-th input argument which is the name of the calling script is
        # disregarded.

    try:

        try:
            opts, args = getopt.getopt(argv, "hvpfo:",
                                       ["help", "output-dir=",
                                        "tab-only", "fig-only", "rld-only",
                                        "los-only", "crafting-effort=",
                                        "pickle", "verbose", "final"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        inputCrE = None
        isfigure = True
        istab = True
        isrldistr = True
        islogloss = True
        isPostProcessed = False
        isPickled = False
        isDraft = True
        verbose = False
        outputdir = 'ppdata'

        #Process options
        for o, a in opts:
            if o in ("-v","--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-p", "--pickle"):
                isPickled = True
            elif o in ("-o", "--output-dir"):
                outputdir = a
            elif o in ("-f", "--final"):
                isDraft = False
            #The next 3 are for testing purpose
            elif o == "--tab-only":
                isfigure = False
                isrldistr = False
                islogloss = False
            elif o == "--fig-only":
                istab = False
                isrldistr = False
                islogloss = False
            elif o == "--rld-only":
                istab = False
                isfigure = False
                islogloss = False
            elif o == "--los-only":
                istab = False
                isfigure = False
                isrldistr = False
            elif o == "--crafting-effort":
                try:
                    inputCrE = float(a)
                except ValueError:
                    raise Usage('Expect a valid float for flag crafting-effort.')
            else:
                assert False, "unhandled option"

        if (not verbose):
            warnings.simplefilter('ignore')

        print ("BBOB Post-processing: will generate post-processing " +
               "data in folder %s" % outputdir)
        print "  this might take several minutes."

        filelist = list()
        for i in args:
            if os.path.isdir(i):
                filelist.extend(findfiles.main(i, verbose))
            elif os.path.isfile(i):
                filelist.append(i)
            else:
                txt = 'Input file or folder %s could not be found.' % i
                raise Usage(txt)

        dsList = DataSetList(filelist, verbose)

        if not dsList:
            raise Usage("Nothing to do: post-processing stopped.")

        if (verbose):
            for i in dsList:
                if (dict((j, i.itrials.count(j)) for j in set(i.itrials)) !=
                    instancesOfInterest2010):
                    warnings.warn('The data of %s do not list ' %(i) +
                                  'the correct instances ' +
                                  'of function F%d.' %(i.funcId))

                # BBOB 2009 Checking
                #if ((dict((j, i.itrials.count(j)) for j in set(i.itrials)) !=
                    #instancesOfInterest) and
                    #(dict((j, i.itrials.count(j)) for j in set(i.itrials)) !=
                    #instancesOfInterest2010)):
                    #warnings.warn('The data of %s do not list ' %(i) +
                                  #'the correct instances ' +
                                  #'of function F%d or the ' %(i.funcId) +
                                  #'correct number of trials for each.')

        dictAlg = dsList.dictByAlg()
        if len(dictAlg) > 1:
            warnings.warn('Data with multiple algId %s ' % (dictAlg) +
                          'will be processed together.')
            #TODO: in this case, all is well as long as for a given problem
            #(given dimension and function) there is a single instance of
            #DataSet associated. If there are more than one, the first one only
            #will be considered... which is probably not what one would expect.
            #TODO: put some errors where this case would be a problem.

        if isfigure or istab or isrldistr or islogloss:
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
                if verbose:
                    print 'Folder %s was created.' % (outputdir)

        if isPickled:
            dsList.pickle(verbose=verbose)

        plt.rc("axes", labelsize=24, titlesize=24)
        plt.rc("xtick", labelsize=20)
        plt.rc("ytick", labelsize=20)
        plt.rc("font", size=20)
        plt.rc("legend", fontsize=20)

        if isfigure:
            ppfigdim.main(dsList, figValsOfInterest, outputdir,
                          verbose)
            print "Scaling figures done."

        if istab:
            dictFunc = dsList.dictByFunc()
            for fun, sliceFun in dictFunc.items():
                dictDim = sliceFun.dictByDim()
                tmp = []
                for dim in tabDimsOfInterest:
                    try:
                        if len(dictDim[dim]) > 1:
                            warnings.warn('Func: %d, DIM %d: ' % (fun, dim) +
                                          'multiple index entries. Will only '+
                                          'process the first ' +
                                          '%s.' % dictDim[dim][0])
                        tmp.append(dictDim[dim][0])
                    except KeyError:
                        pass
                if tmp:
                    filename = os.path.join(outputdir,'ppdata_f%d' % fun)
                    pptex.main(tmp, tabValsOfInterest, filename, isDraft,
                               verbose)
            print "TeX tables",
            if isDraft:
                print ("(draft) done. To get final version tables, please "
                       "use the -f option with run.py")
            else:
                print "done."

        if isrldistr:
            dictNoise = dsList.dictByNoise()
            if len(dictNoise) > 1:
                warnings.warn('Data for functions from both the noisy and '
                              'non-noisy testbeds have been found. Their '
                              'results will be mixed in the "all functions" '
                              'ECDF figures.')
            dictDim = dsList.dictByDim()
            for dim in rldDimsOfInterest:
                try:
                    sliceDim = dictDim[dim]
                    pprldistr.main(sliceDim, rldValsOfInterest, True,
                                   outputdir, 'dim%02dall' % dim, verbose)
                    dictNoise = sliceDim.dictByNoise()
                    for noise, sliceNoise in dictNoise.iteritems():
                        pprldistr.main(sliceNoise, rldValsOfInterest, True,
                                       outputdir, 'dim%02d%s' % (dim, noise),
                                       verbose)
                    dictFG = sliceDim.dictByFuncGroup()
                    for fGroup, sliceFuncGroup in dictFG.items():
                        pprldistr.main(sliceFuncGroup, rldValsOfInterest, True,
                                       outputdir, 'dim%02d%s' % (dim, fGroup),
                                       verbose)

                    pprldistr.fmax = None #Resetting the max final value
                    pprldistr.evalfmax = None #Resetting the max #fevalsfactor
                except KeyError:
                    pass
            print "ECDF graphs done."

        if islogloss:
            for ng, sliceNoise in dsList.dictByNoise().iteritems():
                if ng == 'noiselessall':
                    testbed = 'noiseless'
                elif ng == 'nzall':
                    testbed = 'noisy'
                txt = ("Please input crafting effort value "
                       + "for %s testbed:\n  CrE = " % testbed)
                CrE = inputCrE
                while CrE is None:
                    try:
                        CrE = float(input(txt))
                    except (SyntaxError, NameError, ValueError):
                        print "Float value required."
                dictDim = sliceNoise.dictByDim()
                for d in rldDimsOfInterest:
                    try:
                        sliceDim = dictDim[d]
                    except KeyError:
                        continue
                    info = 'dim%02d%s' % (d, ng)
                    pplogloss.main(sliceDim, CrE, True, outputdir, info,
                                   verbose=verbose)
                    pplogloss.generateTable(sliceDim, CrE, outputdir, info,
                                            verbose=verbose)
                    for fGroup, sliceFuncGroup in sliceDim.dictByFuncGroup().iteritems():
                        info = 'dim%02d%s' % (d, fGroup)
                        pplogloss.main(sliceFuncGroup, CrE, True, outputdir, info,
                                       verbose=verbose)
                    pplogloss.evalfmax = None #Resetting the max #fevalsfactor

            print "ERT loss ratio figures and tables done."

        if isfigure or istab or isrldistr or islogloss:
            print "Output data written to folder %s." % outputdir

        plt.rcdefaults()

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2


if __name__ == "__main__":
   sys.exit(main())
