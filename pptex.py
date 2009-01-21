#! /usr/bin/env python

# Creates tex-file with table entries.

import os
import sys
import scipy

from bbob_pproc import bootstrap
from pdb import set_trace

samplesize = 15 #Bootstrap sample size
header = ['$\Delta f$', '$\ENFEs$', '10\%', '90\%', '$\#$',
          'best', '$2^\mathrm{nd}$', 'med.', '$2^\mathrm{nd}\,$w.', 'worst']
format = ['%1.1e', '%1.1e', '%1.1e', '%1.1e', '%d',
          '%1.1e', '%1.1e', '%1.1e', '%1.1e', '%1.1e']
ranksOfInterest = (1, 2)
maxEvalsFactor = 1e4 #Warning! this appears in pprldistr as well!

idxNbSuccRuns = (4, 13) #TODO: global variable?
#TODO: this global variables depend on ranksOfInterest


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
        data = scipy.transpose(data)

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

def writeTable2(data, filename, entryList, header=list(), fontSize='scriptsize',
                fontSize2='scriptstyle', format=list(), keep_open=0,
                width=2, verbose=True):
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

    #Why does this function needs entryList?

    # Assemble header for whole table (contains more than 1 dimension)
    header = header + (width-1)*header[1:]
    format = format + (width-1)*format[1:]

    # Input checking and reshape data
    if len(format) != data.shape[0] and len(format) != data.shape[1]:
        raise WrongInputSizeError('data',data.shape,len(format))
    elif len(format) != len(header) :
        raise WrongInputSizeError('header',len(header),len(format))
    elif len(format) != data.shape[0]:
        data = scipy.transpose(data)

    # Generate LaTex commands for vertical lines and aligment of the entries.
    # Vertical lines appear after column 'Ft' and '$P_{\mathrm{s}}$'
    tabColumns ='@{$\,$}c@{$\,$}|' + 4 * '@{$\,$}c@{$\,$}' + ''
    tabColumns = tabColumns + 5 * '@{$\,$}c@{$\,$}' + '|'
    tabColumns = tabColumns + 4 * '@{$\,$}c@{$\,$}' + ''
    tabColumns = tabColumns + 5 * '@{$\,$}c@{$\,$}'

    # Create output file
    if verbose:
        if os.path.exists(filename):
            print 'Overwrite old file %s!' %(filename + '.tex')

    try:
        f = open(filename + '.tex','w')
    except ValueError:
        print 'Error opening ' + filename + '.tex'

    # Write tabular environment
    f.write('\\begin{' + fontSize + '} \n')
    f.write('\\begin{tabular}{' + tabColumns + '} \n')

    # Write first two rows containing the info of the table columns
    for i in range(0, width):
        caption = ('\\textbf{\\textit{f}\\raisebox{-0.35ex}{' + str(entryList[i].funcId) +
                   '} in ' + str(entryList[i].dim)) + '-D}'
        caption = caption + ', Nruns = ' + str(entryList[i].nbRuns)
        maxEvals = min(entryList[i].mMaxEvals, entryList[i].dim * maxEvalsFactor)
        caption = caption + ', max.\,FEvals = ' + str(int(maxEvals))
        if i != width - 1:
            f.write('& \multicolumn{' + str((len(format)-1)/width) + '}{@{$\,$}c|@{$\,$}}{' + caption + '}')
        else:
            f.write('& \multicolumn{' + str((len(format)-1)/width) + '}{@{$\,$}c@{$\,$}}{' + caption + '}')
    f.write('\\\\ \n')
    # f.write('\hline \n')
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

    # TODO: I think the written numbers are only correct, if the input format specifies
    #       two numbers of precision. Otherwise the rounding procedure is wrong. 

    # Loop through vector
    for id, x in enumerate(vector):
        #print type(x)
        #print len(vector)

        # Filter nan entries
        if scipy.isinf(x):
            tmp2 = '\infty'
        elif scipy.isnan(x):
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

            if id == 0:  # Delta f value
                if x >= 1 and x <= 100 and x == round(x):
                    tmp2 = str(int(round(x)))  # tmp[0][0]
                else:
                    sgn = '+'  # don't assume that a + sign is present
                    if x < 1:
                        sgn = '-'
                    tmp2 = (tmp[0][0] + '\\!\\mathrm{\\hspace{0.10em}e}' +
                            sgn + tmp[1][-1])                            
            else:
                if x < 0:
                    tmp2 = ('\\mathit{' + tmp[0][1] + tmp[0][3] + '}' +
                            '\\hspace{0.08em}e')
                    # tmp[0][1] + tmp[0][3]: tmp[0][0] is the sign

                    #TODO: hack because we change the number format
                    tmp2 += '\\mathit{%+d}' % (int(tmp[1]) - 1)
                else:
                    tmp2 = tmp[0] + '\\mathrm{\\hspace{0.10em}e}' + tmp[1][-1] 
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


def sortIndexEntries(indexEntries, dimOfInterest):
    """From a list of IndexEntry, returs a post-processed sorted dictionary."""
    sortByFunc = {}
    for elem in indexEntries:
        sortByFunc.setdefault(elem.funcId,{})
        if (dimOfInterest.count(elem.dim) != 0):
            sortByFunc[elem.funcId][elem.dim] = elem

    return sortByFunc


def generateData(indexEntry, targetFuncValues):
    """Returns data to be plotted from indexEntry and the target function values."""

    res = []
    it = iter(indexEntry.hData)
    i = it.next()
    curLine = []

    maxEvals = min(indexEntry.mMaxEvals, indexEntry.dim * maxEvalsFactor)

    #set_trace()
    for targetF in targetFuncValues:
        while i[0] > targetF:
            try:
                i = it.next()
            except(StopIteration):
                isFinished = True
                break
        success = []
        for j in range(1, indexEntry.nbRuns+1):
            success.append(i[indexEntry.nbRuns+j] <= targetF and 
                           i[j] <= maxEvals)

        N = scipy.sort(i[1:indexEntry.nbRuns + 1])

        sp1m = bootstrap.sp1(N, issuccessful=success)
        dispersionSP1 = bootstrap.draw(N, [10,90], samplesize=samplesize, 
                                       func=bootstrap.sp1, 
                                       args=[0,success])[0]
        curLine = [targetF, sp1m[0], dispersionSP1[0],
                   dispersionSP1[1], sp1m[2]]

        tmp = []
        vals = scipy.sort(indexEntry.vData[-1, indexEntry.nbRuns+1:])
        #set_trace()
        for j in ranksOfInterest:
            if sp1m[2] >= j:
                tmp.append(N[j-1])
            else:
                tmp.append(-vals[j-1]) #minimization

        if sp1m[2] > float(indexEntry.nbRuns)/2:
            tmp.append(bootstrap.prctile(N, 50, issorted=True)[0])
        else:
            tmp.append(-bootstrap.prctile(vals, 50, issorted=True)[0])

        for j in reversed(ranksOfInterest):
            if sp1m[2] > indexEntry.nbRuns-j:
                tmp.append(N[-j])
            else:
                tmp.append(-vals[-j])

        curLine.extend(tmp)
        res.append(curLine)
            #set_trace()
    return scipy.vstack(res)


def main(indexEntries, dimOfInterest, valOfInterests, outputdir, verbose):
    """From a list of IndexEntry and a prefix, returns latex tabulars in files.
    """

    sortByFunc = sortIndexEntries(indexEntries, dimOfInterest)

    for func in sortByFunc:
        filename = os.path.join(outputdir,'ppdata_f%d' % (func))
        # initialize matrix containing the table entries
        tabData = scipy.zeros(0)
        entryList = list()     # list of entries which are displayed in the table

        for dim in sorted(sortByFunc[func]):
            entry = sortByFunc[func][dim]

            if dimOfInterest.count(dim) != 0 and tabData.shape[0] == 0:
                # Array tabData has no previous values.
                tabData = generateData(entry, valOfInterests)
                entryList.append(entry)
            elif dimOfInterest.count(dim) != 0 and tabData.shape[0] != 0:
                # Array tabData already contains values for the same function
                tabData = scipy.append(tabData,
                                       generateData(entry, 
                                                    valOfInterests)[:, 1:], 1)
                #set_trace()
                entryList.append(entry)

        #[header,format] = pproc.computevalues(None,None,header=True)
        #set_trace()
        writeTable2(tabData, filename, entryList, fontSize='tiny',
                    header=header, format=format, verbose=verbose)
