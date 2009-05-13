#! /usr/bin/env python

import os
import numpy

origdirectory = os.getcwd()  # just a help for executing the script in ipython without changing the directory every time
# location of the folder with the *.tdat files
folder = '../../../mydata/'
os.chdir(folder)
directory = os.getcwd()
# find all *.tdat files
tdatFiles = list()
for root, dirs, files in os.walk(directory):
    for elem in files:
        if elem.endswith('.tdat'):
            tdatFiles.append(os.path.join(root, elem))
# change back the starting directory
os.chdir(origdirectory)
print len(tdatFiles)
# returns list (unsorted) of all possible files

# set values
dims = [2]     # dimensions of interest
funcs = range(101,111)  # functions of interest

# loop over dimensions
for dim in dims:

    # loop over functions
    for func in funcs:
        # tag to find 'right' files
        tag = 'f' + str(func) + '_DIM' + str(dim) + '.tdat'
        # sample counter
        j = -1  

        # loop over files
        for file in tdatFiles:
                      
            # only files with the correct tag will be processed
            if file.endswith(tag):
                f = open(file)
                print f.name
  
                # loop to gather all data          
                while True:
                    
                    # read current line
                    line = f.readline()

                    if len(line) == 0:
                        # end-of-file is reached
                        tdatFiles.remove(file) # file is not needed anymore can be removed from list
                        break
                    
                    # split line 
                    tmp = line.split()

                    if tmp[0] == '%':
                        # skip comment line and reset counters
                        i = 0    # counter for evals
                        j = j+1  # counter for samples

                        if j == 0:
                            # initialization of matrices
                            fvalue = numpy.zeros((1,1))
                            feval = numpy.zeros((1,1))  # feval is nor really needed but helps for debugging
                        else:
                            # increase size to account for more samples
                            fvalue = numpy.hstack((fvalue,numpy.zeros((fvalue.shape[0],1))))
                            feval = numpy.hstack((feval,numpy.zeros((feval.shape[0],1))))
                    elif int(tmp[0]) <=  dim*2*10**i:
                        # create temporary values                       
                        fvalue_tmp = float(tmp[2])
                        feval_tmp = int(tmp[0])
                    else:
                        # save temporary values to arrays and increase counter
                        if i+1 > fvalue.shape[0]:
                            # check the current dimension of the arrays and increase if necessary
                            fvalue = numpy.vstack((fvalue,numpy.zeros((1,fvalue.shape[1]))))
                            feval = numpy.vstack((feval,numpy.zeros((1,feval.shape[1]))))
                        fvalue[i,j] = fvalue_tmp
                        feval[i,j] = feval_tmp
                        i = i+1

        # determine best and median
        best = numpy.min(fvalue,axis = 1)
        median = numpy.median(fvalue,axis = 1)
          
        # print out values
        print func
        print best
        print median 
        print numpy.min(feval,axis=1)
        print numpy.median(feval,axis=1)
                   
# write out in table for latex
# maybe use the writeTable function from bbob_pproc 


