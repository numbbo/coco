#! /usr/bin/env python
# coding: utf-8

import os
import numpy
import sys

# add path to bbob_pproc  
filepath = os.getcwd()
sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc import pproc2
# from bbob_pproc import pptex

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
    """ Write data in tex-files for creating tables 
    """

    #print data

    # parameter
    fontSize = 'tiny'   
    
    # define header and format of columns
    header = ['$evals/D$'] 
    format = ['%1e']
    for id in range(0,len(data)):
        header.append('$f_{' + str(data[id]['funcId']) + '}$')
        format.append('%1.1e')

    # create latex string for table definiton
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
    # write each row separately
    while True:
        tableData = [10**i]
        # create data for each function
        for fun in range(0,len(data)):
            try:
                tableData.append(data[fun]['min'][i])
            except:
                # if no entry exist write nan
                tableData.append(numpy.nan)
    
            # create termination condition
            if i == 0 and len(data[fun]['ert']) > maxLength:
                maxLength = len(data[fun]['ert'])

        #print tableData

        # write row in latex format
        writeArray(f,tableData,format, 'scriptstyle')

        # check termination condition
        if maxLength > i+1:
            i += 1
        else:
            break

    # finish writing and write caption
    f.write('\end{tabular} \n')
    f.write('\caption{target function value (min) for increasing problem difficulty in '+ str(dim) +'-D} \n')  
    f.write('\end{' + fontSize + '} \n')

    # close file
    f.close()
    if not f.closed:
        raise ValueError, ('File ' + filename +'.tex is not closed!')

def writeArray(file, vector, format, fontSize, sep=' & ', linesep='\\\\ \n',
               suppress_entry=None):
    """ Writes components of an numeric array in LaTex file with additional
        Tex-formating features. Negative numbers are printed positive but in
        italics.

        Inputs:
        file - file in which the output is written to
        vector - 1d-array with only numeric entries
        format - format specifier for each column.
            CAVE: numbers are only printed correctly,
                  if format specifies two numbers of prec.
        fontSize - size of characters in math environment

        Optional Inputs:
        sep - string which is written between the numeric elements
        format - format for the numeric values (e.g. 'e','f')
        suppress_entry - list of boolean of len of vector, if true
           a '.' is written. Useful to not repeat the same line of
           function values again.
    """

    # TODO (see CAVE above): I think the written numbers are only correct, if
    # the input format specifies two numbers of precision. Otherwise the
    # rounding procedure is wrong.

    # handle input arg
    if suppress_entry is None:
        suppress_entry = len(vector) * (False,)

    # Loop through vector
    for i, x in enumerate(vector):

        # Filter entries to suppress, nan, inf...
        if suppress_entry[i]:
            tmp2 = '.'
        elif numpy.isinf(x):
            tmp2 = '\infty'
        elif numpy.isnan(x):
            tmp2 = '-'

        elif format[i].endswith('e'):

            # Split number and sign+exponent
            try:
                tmp = str(format[i]%x).split('e')
            except TypeError:
                print format[i]
                print x
                print type(x)
                print type(format[i])

            # Generate Latex entry
            # It is assumed that all entries range between 10e-9 and 10e9
            if i == 0:  # ert/dim values
                if x <= 100 :
                    tmp2 = str(int(round(x)))  # tmp[0][0]
                else:
                    tmp2 = (tmp[0][0] + '\\!\\mathrm{\\hspace{0.10em}e}' +
                            tmp[1][-1])
            else:   # target function values
                if x > 1:
                    sgn = '+'
                else:
                    sgn = '-'
                tmp2 = tmp[0] + '\\mathrm{\\hspace{0.10em}e}' + sgn + tmp[1][-1]
        else:
            tmp2 = str(format[i]%x)

        tmp2 = '$' + '\\' + fontSize + tmp2 + '$'

        # Print in between separator or end of line separator
        if i != len(vector)-1:
            tmp2 = tmp2 + sep
        else:
            tmp2 = tmp2 + linesep

        # Write to file
        file.write(tmp2)


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
            #print dataset  
            tmp = FunTarget(dataset,dim)
            #print tmp.minFtarget
            #print tmp.medianFtarget
            #print tmp.ert
            ftarget.append({'dim':dim,'funcId':fun,'min':tmp.minFtarget,'median':tmp.medianFtarget,'ert':tmp.ert})

        # write data into table
        writeTable(ftarget,dim)
    
