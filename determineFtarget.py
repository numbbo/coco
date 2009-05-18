#! /usr/bin/env python
# coding: utf-8

import os
import numpy
import sys

# add path to bbob_pproc  
filepath = os.getcwd()
sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc import pproc2
from bbob_pproc import pptex

from pdb import set_trace

class FunTarget:
    """ Determines the best and median function value from the data of all 
        algorithms for ERT between 2*Dim*10**(i-1) and 2*Dim*10**i for i = 0,1,2,... 
        The search stops if the minimal target function values of 1e-8 is reached
        or no more data for ERT>2*Dim*10**i exist.

        Class Attributes:
        minFtarget - minimal function value over all algorithms for given 
                     dimension, function, and ERT (array)
        medianFtarget - median function value over all algorithms for given 
                       dimension, function, and ERT (array)
        ert - corresponds to ERT/DIM for minFtarget (array)
    """
   
    def __init__(self,dataset,dim):

        # initialize
        i = 0
        self.minFtarget = numpy.array([])
        self.medianFtarget = numpy.array([])
        self.ert = numpy.array([])
        maxErtAll = 0

        while True:

            targetValue = numpy.array([])

            # search all algorithms
            for alg in dataset:

                # set maxErtAll for termination
                if i == 0 and alg.ert[-1] > maxErtAll:
                    maxErtAll = alg.ert[-1]

                id = 0
                while id < len(alg.ert):
                    tmp = 0
                    if alg.ert[id] <= 2*dim*10**i:
                        id += 1
                    elif alg.ert[id] > 2*dim*10**i:
                        tmp = alg.target[id-1]
                        break # target value is reached
                
                if tmp == 0:
                    # if no ERT is larger than 2*dim*10**i take
                    # the best value (last entry in alg.target)
                    tmp = alg.target[-1]

                # add to list of targetvalues
                targetValue = numpy.append(targetValue,tmp)

            # determine min and median and set attributes
            self.minFtarget = numpy.append(self.minFtarget,numpy.min(targetValue))
            self.medianFtarget = numpy.append(self.medianFtarget,numpy.median(targetValue))
            self.ert = numpy.append(self.ert,10**i)             
            
            # check termination conditions
            if numpy.min(targetValue) <= 1e-8 or maxErtAll < 2*dim*10**i:
                break 

            # increase counter
            i += 1

def writeTable(data,dim):

    print data

    # parameter
    fontSize = 'tiny'   
    
    # define header and format of columns
    header = ['$evals/D$'] 
    format = ['%d']
    for id in range(0,len(data)):
        header.append('$f_{' + str(data[id]['funcId']) + '}$')
        format.append('%1.1e')
    print header

    tabColumns = '@{$\;$}r@{$\;$}'
    tabColumns += ('|' + (len(header) - 1) * '@{$\;$}c@{$\;$}')

    # open file
    filename = 'ftarget_dim' + str(dim)
    try:
        f = open(filename + '.tex','w')
    except ValueError:
        print 'Error opening '+ filename +'.tex'

    # Write tabular environment
    f.write('\\begin{' + fontSize + '} \n')
    f.write('\\begin{tabular}{' + tabColumns + '} \n')

    # write header
    f.write(' & '.join(header) + '\\\\ \n \hline \n')

    # write data
    i = 0
    maxLength = 0
    while True:
        # set_trace()
        tableData = [10**i]
        for fun in range(0,len(data)):
            try:
                tableData.append(data[fun]['min'][i])
            except:
                tableData.append('--')

            if i == 0 and len(data[fun]['ert']) > maxLength:
                maxLength = len(data[fun]['ert'])

        print tableData
        pptex.writeArray(f,tableData,format, 'scriptstyle')
        if maxLength > i+1:
            i += 1
        else:
            break

    # finish writing
    f.write('\end{tabular} \n')
    f.write('\caption{target function value (min) for increasing problem difficulty in '+ str(dim) +'-D')  
    f.write('\end{' + fontSize + '} \n')

    # close file
    f.close()

    if not f.closed:
        raise ValueError, ('File ' + filename +'.tex is not closed!')


def main(directory,dims, funcs):
    """From a directory which contains the data of the algorithms
       the minimum reached target value and the median reached target
       value for all algorithms will be determined for all dimensions
       and functions. 

       Input parameter:
       directory - directory containing the data (list)
       dims - Dimensions of interest (list)
       funcs - dfunctions of interest (list)
    """
    
    # directory containing the data
    # directory = ['/home/fst/coco/BBOB/mydata/noisySPSA1e4', '/home/fst/coco/BBOB/mydata/noisySPSAHA1e4']

    # functions and dimension of interest
    # dims = [10]
    # funcs = [101]

    # create dataset
    datasetfull = pproc2.DataSetList(directory,verbose=False)

    # loop over dimension and functions
    for dim in dims:

        # create list which contains min and median values across all 
        # algorithms for all functions
        ftarget = list()
        
        for fun in funcs:
                
            # create list which only contains entries with dimension = dim
            # for function = fun   
            dataset = list()                     
            for elem in datasetfull:        
                if elem.dim == dim and elem.funcId == fun:
                    dataset.append(elem)
                    datasetfull.remove(elem)
                
            if len(dataset) == 0:
                raise ValueError, ('No entry found for dim = %g' %dim 
                                  + ' and function = f%g!' %fun)

            # get min and median values 
            print dataset  
            tmp = FunTarget(dataset,dim)
            print tmp.minFtarget
            #print tmp.medianFtarget
            #print tmp.ert
            ftarget.append({'dim':dim,'funcId':fun,'min':tmp.minFtarget,'median':tmp.medianFtarget,'ert':tmp.ert})

        # write data into table
        writeTable(ftarget,dim)

    #return ftarget
 
    
    
