#!/usr/bin/env python

"""Calls the main function of bbob_pproc with arguments from the
   command line. Executes the BBOB postprocessing on the given
   filename and folder arguments, using all found .info files. 
Synopsis:
  python path_to_folder/bbob_pproc/run.py [OPTIONS] FILE_NAME FOLDER_NAME...
Help: 
  python path_to_folder/bbob_pproc/run.py -h  
"""

# this script should probably replace ../bbob_pproc.py in future? 

import os
import sys

(filepath, filename) = os.path.split(sys.argv[0])

# append path without trailing '/bbob_pproc', using os.sep fails in mingw32 
sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0]) 

import bbob_pproc

def main():
    pass
    bbob_pproc.main(sys.argv)

if __name__ == "__main__":
   sys.exit(main())
