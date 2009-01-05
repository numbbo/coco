#! /usr/bin/env python

# Creates tex-file with table entries.

import os
import sys
import numpy as n
from pdb import set_trace


class Error(Exception):
    """ Base class for errors. """
    pass

class WrongInputSizeError(Error):
    """ Error if an array has the wrong size for the following operation.
        Returns a message containing the size of the array and the required
        size.

    """

    def __init__(self,arrName, arrSize, reqSize):
        self.arrName = arrName
        self.arrSize = arrSize
        self.reqSize = reqSize

    def __str__(self):
        message = 'The size of %s is %s. One dimension must be of length %s!' %\
                  (self.arrName,str(self.arrSize), str(self.reqSize))
        return repr(message)



def writeTable(data, fileName, header = list(), fontSize = 'tiny',
               fontSize2 = 'scriptscriptstyle', format = list(), keep_open = 0,
               verbose = True):
    """Writes data of array in a *.tex file. This file then can be used
       in LaTex as input for tables. The file must be placed inside a
       \begin{table} ... \end{table} environment.

       Mandatory inputs:
       data - sorted 2d-array containing the data values (array)
       fileName - name of output file (string)
       caption - caption for the table (string)

       Optional inputs:
       header - header for the columns of the table (string, list)
       fontSize - size of fonts as they appear in the LaTex table (string)
       fontSize2 - size of the fonts in the math environment (string)
       format - format specifier for each column
       keep_open - if nonzero the file will not be closed

    """

    # Input checking and reshape data
    if len(format) != data.shape[0] and len(format) != data.shape[1]:
        raise WrongInputSizeError('data',data.shape,len(format))
    elif len(format) != len(header) :
        raise WrongInputSizeError('header',len(header),len(format))
    elif len(format) != data.shape[0]:
        data = n.transpose(data)

    # Generate LaTex commands for vertical lines and aligment of the entries.
    # Vertical lines appear after column 'FES' and '$P_{\mathrm{s}}$'
    tabColumns ='@{$\,$}l@{$\,$}' + '|' + 4*'@{$\,$}l@{$\,$}' + '|'
    tabColumns = tabColumns + 5*'@{$\,$}l@{$\,$}'

    # Create and open output file
    f = open(fileName + '.tex','a')

    # Write tabular environment
    f.write('\\begin{table} \n')
    f.write('\\centering \n')
    f.write('\\begin{' + fontSize + '} \n')
    f.write('\\begin{tabular}{' + tabColumns + '} \n')

    # Write first row containing the header of the table columns
    f.write(' & '.join(header) + '\\\\ \n \hline \n')

    # Write data
    for i in range(0,data.shape[1]):
        writeArray(f, data[:,i], format, fontSize2)

    # Finish writing the table and close file.
    f.write('\end{tabular} \n')
    f.write('\end{' + fontSize + '} \n')
    f.write('\end{table} \n')
    #f.write('\caption{{'+ '\\' + fontSize + ' ' + caption + '}} \n')

    # Check if file should be closed
    if keep_open == 0:
        f.close()

    if verbose:
        print 'Wrote in %s.' %(fileName+'.tex')

def writeTable2(data, filename, entryList, header = list(), fontSize = 'scriptsize',
                fontSize2 = 'scriptstyle', format = list(), keep_open = 0,
                width = 2, verbose = True):
    """Writes data of array in a *.tex file. This file then can be used
       in LaTex as input for tables. The file must be placed inside a
       \begin{table} ... \end{table} environment.

       Mandatory inputs:
       data - sorted 2d-array containing the data values (array)
       filename - name of output file (string)
       entry - IndexEntry containing the attributes of the current index entry

       Optional inputs:
       header - header for the columns of the table (string, list)
       fontSize - size of fonts as they appear in the LaTex table (string)
       fontSize2 - size of the fonts in the math environment (string)
       format - format specifier for each column
       keep_open - if nonzero the file will not be closed
       width -  number of dimension (from the same function) which are
                displayed in the table. The first column (Ft) is only
                displayed once.

    """

    # Assemble header for whole table (contains more than 1 dimension)
    header = header + (width-1)*header[1:]
    format = format + (width-1)*format[1:]

    # Input checking and reshape data
    if len(format) != data.shape[0] and len(format) != data.shape[1]:
        raise WrongInputSizeError('data',data.shape,len(format))
    elif len(format) != len(header) :
        raise WrongInputSizeError('header',len(header),len(format))
    elif len(format) != data.shape[0]:
        data = n.transpose(data)

    # Generate LaTex commands for vertical lines and aligment of the entries.
    # Vertical lines appear after column 'Ft' and '$P_{\mathrm{s}}$'
    tabColumns ='@{$\,$}c@{$\,$}|' + 4 * '@{$\,$}c@{$\,$}' + '|'
    tabColumns = tabColumns + 5 * '@{$\,$}c@{$\,$}' +'|'
    tabColumns = tabColumns + 4 * '@{$\,$}c@{$\,$}' +'|'
    tabColumns = tabColumns + 5 * '@{$\,$}c@{$\,$}'

    # Create output file
    try:
        os.listdir(filename.split('/')[0]).index(filename.split('/')[1] + '.tex')
        print 'Overwrite old file %s!' %(filename + '.tex')
        f = open(filename + '.tex','w')
    except ValueError:    
        f = open(filename + '.tex','w')

    # Write tabular environment
    #f.write('\\begin{table*} \n')
    f.write('\\centering \n')
    f.write('\\begin{' + fontSize + '} \n')
    f.write('\\begin{tabular}{' + tabColumns + '} \n')

    # Write first two rows containing the info of the table columns
    for i in range(0,width):
        caption = 'N = ' + str(entryList[i].dim) + ',FId = ' + str(entryList[i].funcId)
        caption = caption + ',max. FEvals = ' + str(entryList[i].maxEvals)
        caption = caption + ',Nruns = ' + str(entryList[i].nbRuns)
        if i != width - 1:
            f.write('& \multicolumn{' + str((len(format)-1)/width) + '}{@{$\,$}c|@{$\,$}}{' + caption + '}')
        else:
            f.write('& \multicolumn{' + str((len(format)-1)/width) + '}{@{$\,$}c@{$\,$}}{' + caption + '}')
    f.write('\\\\ \n')
    f.write('\hline \n')
    f.write(' & '.join(header) + '\\\\ \n \hline \n')

    # Write data
    for i in range(0,data.shape[1]):
        writeArray(f, data[:,i], format, fontSize2)

    # Finish writing the table and close file.
    f.write('\end{tabular} \n')
    f.write('\end{' + fontSize + '} \n')
    #f.write('\end{table*} \n')

    # Check if file should be closed
    if keep_open == 0:
        f.close()

    if verbose:
        print 'Wrote in %s.' %(filename+'.tex')

def writeArray(file, vector, format, fontSize, sep = ' & ',linesep = '\\\\ \n'):
    """ Writes components of an numeric array in LaTex file with additional
        Tex-formating features.

        Inputs:
        file - file in which the output is written to
        vector - 1d-array with only numeric entries
        format - format specifier for each column
        fontSize - size of characters in math environment

        Optional Inputs:
        sep - string which is written between the numeric elements
        format - format for the numeric values (e.g. 'e','f')

    """

    # Loop through vector
    for id,x in enumerate(vector):

        #print type(x)
        #print len(vector)

        # Filter nan entries
        if n.isinf(x):
            tmp2 = '\infty'
        elif n.isnan(x):
            tmp2 = '-'

        elif format[id].endswith('e'):

            # Split number and sign+exponent
            try:
            	tmp = str(format[id]%x).split('e')
            except TypeError:
            	print format[id]
            	print x
            	print type(x)
            	print type(format[id])

            # Generate Latex entry
            # It is assumed that all entries range between 10e-9 and 10e9

            if id == 0:
                tmp2 = '\\!' + tmp[1][0] + '\\!' + tmp[1][2]

            else:
                tmp2 = tmp[0] + '\\mathrm{\\hspace{0.10em}e}' + tmp[1][2]
        else:
            tmp2 = str(format[id]%x)

        tmp2 = '$' + '\\' + fontSize + tmp2 + '$'

        # Print in between separator or end of line separator
        if id != len(vector)-1:
            tmp2 = tmp2 + sep
        else:
            tmp2 = tmp2 + linesep

        # Write to file
        file.write(tmp2)

