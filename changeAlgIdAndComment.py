#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import getopt
from shutil import move
from os import remove, close

"""While comparing algorithms with the bbob_proc package, it is sometimes
		needed to change the algorithm name (given as algId in the .info files)
		or the algorithm comments after a	BBOB run is already finished (for
		example because two output folders contain results for two different
		algorithms but with the same name). This script allows to change these
		things within a specified BBOB output folder.

		written: db 28/01/2010

"""

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def usage():
    print main.__doc__

def main(argv=None):
    """
    This script allows to change algorithm name (algId) and algorithm comment
    after a BBOB run finished, i.e., after an output folder has been created.
    
    Keyword arguments:
    argv -- list of strings containing options and arguments. If not provided,
    sys.argv is accessed.

    argv should list an output folder (first argument) and additionally an
    algorithm name (2nd argument) and the algorithm comment (3rd argument).
    If only the output folder is given, the script asks for an algorithm name
    and a comment interactively.

        -h, --help

            display this message

        -v, --verbose
 
            verbose mode, prints out operations. When not in verbose mode, no
            output is to be expected, except for errors.

    Examples:

    * Changing algorithm name and comments for given output folder from the
       command line:

        $ python bbob_pproc/changeAlgoName.py outfolder "CMA-ES" "CMA_with_lambda_100"

    * Changing algorithm name and comments for given output folder
       interactively:

        $ python bbob_pproc/changeAlgoName.py outputfolder

    """


    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, "hv",
                                       ["help", "verbose"])
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        verbose = False

        #Process options
        for o, a in opts:
            if o in ("-v","--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            else:
                assert False, "unhandled option"
        
        # check if all arguments are there and ask for them if not:
        if len(args) < 3:
            if len(args) < 2:
                name = raw_input("You forgot to specify an algorithm name. Please enter one (algId):")
                args.append(name)
            comment = raw_input("You forgot to specify a comment. Please enter one for algorithm " + args[1] + ":")
            args.append(comment)
        folder = args[0]
        # make sure that folder name ends with a '/' to be able to append
        # the file names afterwards
        if not folder.endswith('/'):
            folder = folder + '/'
        
        algId = args[1]
        comment = args[2]
        
        if not os.path.exists(folder):
            print "ERROR: folder " + folder + " does not exist!"
            sys.exit()
        if not os.path.isdir(folder):
            print "ERROR: " + folder + " is not a directory"
            sys.exit()

        # get all .info files in folder:
        FILES = []
        dirList=os.listdir(folder)
        for fname in dirList:
            if fname.endswith('.info'):
                FILES.append(fname)
        
        for file in FILES:
            # open file to read and temp file to write
            infile = open(folder + file,'r')
            tempfile = open('temp.temp','w')
            while infile:
                line = infile.readline()
                if not line:
                    break
                
                # make sure that everything is copied:
                newline = line
                # check if something needs to be changed:
                if line.find('algId') >= 0:
                    s = line.split()
                    n = 0 # compute position of 'algId'
                    for word in s:
                        n = n+1
                        if word=='algId':
                            break    
                            
                    # replace algId:
                    s = s[0:n+1]
                    s.append("'" + algId + "'\n")
                    newline = " ".join(s)
                else:
                    s = line.split()
                    if '%'==s[0]:
                        newline = "% " + comment + "\n"
                
                tempfile.write(newline)    
                    
            infile.close()
            tempfile.close()
            # remove old file and rename temp file accordingly
            remove(folder + file)
            move('temp.temp', folder + file)
            
            print(folder + file + " changed")
        
        sys.exit()

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2


if __name__ == "__main__":
    sys.exit(main())

