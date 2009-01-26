#!/usr/bin/env python

"""Calls the main function from bbob_pproc with arguments from the
   command line.
Synopsis:
  python path_to_folder/bbob_pproc/run.py [OPTIONS] FILE_NAME FOLDER_NAME...
Help: 
  python path_to_folder/bbob_pproc/run.py -h  
"""

# this script should probably replace ../bbob_pproc.py in future? 

import os
import sys

(filepath,filename) = os.path.split(sys.argv[0])
# append path without trailing '/bbob_pproc'
sys.path.append(filepath[:-1-len(filepath.split(os.sep)[-1])]) 

import bbob_pproc

def main():
    bbob_pproc.main(sys.argv)

if __name__ == "__main__":
   sys.exit(main())
