#! /usr/bin/env python
# coding: utf-8

import os
import numpy
import sys
from pdb import set_trace
import getopt

# add path to bbob_pproc  
#filepath = '/home/fst/coco/BBOB/code/python/bbob_pproc/'
#sys.path.append(os.path.join(filepath, os.path.pardir))
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    # Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc import pproc2

### Class Definitions ###

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
        self.minFtarget = []
        self.medianFtarget = []
        self.ert = []
        maxErtAll = 0

        # loop through all ERT values (2*dim*10**i)
        while True:

            targetValue = []

            # search all algorithms
            for alg in dataset:

                # set maxErtAll for termination
                if i == 0 and alg.ert[-1] > maxErtAll:
                    maxErtAll = alg.ert[-1]

                id = 0
                tmp = numpy.nan
                while id < len(alg.ert):
                    if alg.ert[id] <= 2*dim*10**i:
                        id += 1
                    elif alg.ert[id] > 2*dim*10**i:  # why this? 
                        # minimum for one algorithm
                        tmp = min(alg.target[:id])                            
                        break # target value is reached
                
                # if no ERT value is available
                if numpy.isnan(tmp):
                    tmp = min(alg.target)  

                # add to list of targetvalues
                targetValue.append(tmp)

            # determine min and median for all algorithm
            self.minFtarget.append(numpy.min(targetValue))
            self.medianFtarget.append(numpy.median(targetValue))
            self.ert.append(10**i)             
            
            # check termination conditions
            if numpy.min(targetValue) <= 1e-8 or maxErtAll < 2*dim*10**i or i>0:
                break 

            # increase counter
            i += 1
        
### Function definitons ###

def usage():
    print main.__doc__

def writeTable(data,dim,suffix=None, whichvalue = 'min'):
    """ Write data in tex-files for creating tables.

        Inputvalues:
        data - values which are to be processed (list)
        dim - corresponding dimension. There will be
              at least one table per dimension.
        suffix - If not all data can fit within one side
                 2 files (*_part1.tex and *_part2.tex) will
                 be created. 
        whichvalue - determines wheter the min ('min') values
                     or the median ('median') values are displayed. 
    """

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
    if suffix is None:
        filename = whichvalue + 'ftarget_dim' + str(dim) 
    else:
        filename = whichvalue + 'ftarget_dim' + str(dim) + '_part' + str(suffix)
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
                tableData.append(data[fun][whichvalue][i]) 
            except:
                # if no entry exist write nan
                tableData.append(numpy.nan)
    
            # create termination condition
            if i == 0 and len(data[fun]['ert']) > maxLength:
                maxLength = len(data[fun]['ert'])

        #print tableData

        # write row in latex format
        writeArray(f,tableData,format,'scriptstyle')

        # check termination condition
        if maxLength > i+1:
            i += 1
        else:
            break

    # finish writing and write caption
    f.write('\end{tabular} \n')
    if suffix is None or suffix == 2:
        f.write('\caption{target function value ('+ whichvalue +
                ') for increasing problem difficulty in '+ str(dim) +'-D} \n')  
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


def main(argv=None):
    """From a directory which contains the data of the algorithms
       the minimum reached target value and the median reached target
       value for all algorithms will be determined for all dimensions
       and functions. 

       Input parameter:
       argv - list of strings containing options and arguments. If not given,
              sys.argv is accessed. The first argument is the directory which
              contains the data files.
       flags:
       -h,--help           - displays help
       -d,--dimensions DIM - dimension(s) of interest
       -f,--functions FUN  - function(s) of interest
       --noisefree         - noisefree function set 
       --noisy             - noisy function set
       -v,--verbose        - verbose output
       Either the flag -f FUN or -nf or -n should be set!
    """

    if argv is None:
        argv = sys.argv[1:]
        
    try: 
        opts, args = getopt.getopt(argv, "hvd:f:",["help", "dimensions=","functions=","noisy","noisefree","verbose"])

    except getopt.error, msg:
        raise Usage(msg)

    if not (args):
        usage()
        sys.exit()

    
    verboseflag = False
    dims = list()
    funcs = list()
    directory = argv[-1]  # directory which contains data  

    # Process options
    for o, a in opts:
        if o in ("-h","--help"):
            usage()
            sys.exit()
        elif o in ("-d", "--dimensions"):
            dims.append(int(a))
        elif o in ("-f", "--functions"):
            funcs.append(int(a))
        elif o in ("--noisy"):
            funcs = range(101,131)
        elif o in ("--noisefree"):
            funcs = range(1,25)
        elif o in ("-v","--verbose"):
            verboseflag = True
        else:
            assert False, "unhandled option"

    if len(dims) == 0:
        raise ValueError,('No dimension(s) specified!')
    if len(funcs) == 0:
        raise ValueError,('No function(s) specified!')

    # partition data since not all functions can be displayed in 
    # one table
    partition = [1]
    half = len(funcs)
    if half > 12:
        partition.append(2)
        half = int(round(len(funcs)/2))
    
    # create dataset
    datasetfull = pproc2.DataSetList(directory,verbose = verboseflag)

    # loop over dimension and functions
    for dim in dims:

        # use partition
        for p in partition:

            # create list which contains min and median values across all 
            # algorithms for all functions
            ftarget = list()
        
            for fun in funcs[0+int((p-1)*half):int(p*half)]:
                
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
                #print fun
                tmp = FunTarget(dataset,dim)
                #print tmp.minFtarget
                #print tmp.medianFtarget
                #print tmp.ert
                ftarget.append({'dim':dim,'funcId':fun,'min':tmp.minFtarget,'median':tmp.medianFtarget,'ert':tmp.ert})

            # write data into table
            writeTable(ftarget,dim,p,whichvalue = 'min')
            writeTable(ftarget,dim,p,whichvalue = 'median')

if __name__ == "__main__":
    sys.exit(main())
    
