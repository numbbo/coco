#! /usr/bin/env python

# Creates tex-file with table entries.

import os
import sys
import scipy

from bbob_pproc import bootstrap
from pdb import set_trace

samplesize = 15 #Bootstrap sample size
header = ['$\Delta f$', '$\ENFEs$', '10\%', '90\%', '$\#$',
          'best', '$2^\mathrm{nd}$', 'med.', '$2^\mathrm{nd}$w.', 'worst']
format = ['%1.1e', '%1.1e', '%1.1e', '%1.1e', '%d',
          '%1.1e', '%1.1e', '%1.1e', '%1.1e', '%1.1e']
#This has to be synchronized with what's computed in generateData.

ranksOfInterest = (1, 2)

idxNbSuccRuns = (4, 13) #TODO: global variable?
#TODO: this global variables depend on ranksOfInterest

maxEvalsFactor = 1e6

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

def writeTable2(entries, filename, fontSize='scriptsize',
                fontSize2='scriptstyle', keep_open=0,
                verbose=True):
    """Writes data of array in a *.tex file. This file then can be used
       in LaTex as input for tables. The file must be placed inside a
       \begin{table} ... \end{table} environment.

       Mandatory inputs:
       entries - Sequence of IndexEntry to be displayed.
       filename - name of output file (string)

       Optional inputs:
       fontSize - size of fonts as they appear in the LaTex table (string)
       fontSize2 - size of the fonts in the math environment (string)
       keep_open - if nonzero the file will not be closed
       width -  number of dimension (from the same function) which are
                displayed in the table. The first column (Ft) is only
                displayed once.

    """

    # Assemble header for whole table (contains more than 1 dimension)
    width = len(entries)

    fullheader = header + (width-1)*header[1:]
    fullformat = format + (width-1)*format[1:]

    data = []
    it = iter(entries)
    data.append(it.next().tabData)
    while True:
        try:
            data.append(it.next().tabData[:, 1:])
        except StopIteration:
            break

    data = scipy.transpose(scipy.concatenate(data, 1))

    # Input checking and reshape data
    #if len(fullformat) != data.shape[0] and len(fullformat) != data.shape[1]:
        #raise WrongInputSizeError('data',data.shape,len(format))
    #elif len(fullformat) != len(fullheader) :
        #raise WrongInputSizeError('header',len(fullheader),len(format))
    #elif len(fullformat) != data.shape[0]:
        #data = scipy.transpose(data)

    # Generate LaTex commands for vertical lines and aligment of the entries.
    tabColumns ='@{$\,$}c@{$\,$}'
    tabColumns += ('|' + (len(header) - 1) * '@{$\,$}c@{$\,$}') * width

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
        caption = ('\\textbf{\\textit{f}\\raisebox{-0.35ex}{' + str(entries[i].funcId) +
                   '} in ' + str(entries[i].dim)) + '-D}'
        caption = caption + ', Nruns = ' + str(entries[i].nbRuns)
        maxEvals = min(entries[i].mMaxEvals, entries[i].dim * maxEvalsFactor)
        caption = caption + ', max.\,FEvals = ' + str(int(maxEvals))
        if i != width - 1:
            f.write('& \multicolumn{' + str(len(format)-1) + '}{@{$\,$}c|@{$\,$}}{' + caption + '}')
        else:
            f.write('& \multicolumn{' + str(len(format)-1) + '}{@{$\,$}c@{$\,$}}{' + caption + '}')
    f.write('\\\\ \n')
    # f.write('\hline \n')
    f.write(' & '.join(fullheader) + '\\\\ \n \hline \n')

    # Write data
    for i in range(0,data.shape[1]):
        suppr = None
        # suppress the same f-value entries from being written more than once
        if len(data[:,i]) > 9:
            suppr = [data[j, i] < 0 for j in xrange(data.shape[0])] # suppress f-values by default
            ftarget = data[0,i]
            fbest = -data[5,i]
            if ((fbest > ftarget and (i == 0 or fbest <= data[0,i-1])) or  # first line without any #FEvals entry
                (i + 1 == data.shape[1] and fbest < data[0,i-1])):         # or last line, if not printed before
                for j in scipy.r_[5:10]:
                    suppr[j] = False
        if len(data[:,i]) > 18:
            fbest = -data[14,i]
            if ((fbest > ftarget and (i == 0 or fbest <= data[0,i-1])) or  # first line without any #FEvals entry
                (i + 1 == data.shape[1] and fbest < data[0,i-1])):         # or last line, if not printed before
                for j in scipy.r_[14:19]:
                    suppr[j] = False

        writeArray(f, data[:,i], fullformat, fontSize2,
                   suppress_entry = suppr) # only one line of f-values

    # Finish writing the table and close file.
    f.write('\end{tabular} \n')
    f.write('\end{' + fontSize + '} \n')
    #f.write('\end{table*} \n')

    # Check if file should be closed
    if keep_open == 0:
        f.close()

    if verbose:
        print 'Wrote in %s.' %(filename+'.tex')


def writeArray(file, vector, format, fontSize, sep = ' & ', linesep = '\\\\ \n',
               suppress_entry = None):
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

    # TODO (see CAVE above): I think the written numbers are only correct, if the
    #     input format specifies two numbers of precision. Otherwise the rounding procedure is wrong.

    # handle input arg
    if suppress_entry is None:
        suppress_entry = len(vector) * (False,)

    # Loop through vector
    for id, x in enumerate(vector):
        #print type(x)
        #print len(vector)

        # Filter entries to suppress, nan, inf...
        if suppress_entry[id]:
            tmp2 = '.'
        elif scipy.isinf(x):
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
                    tmp2 = ('\\textit{' + tmp[0][1] + tmp[0][3] + '}' +  # textit is narrower
                            '\\hspace{0.00em}e')
                    # tmp[0][1] + tmp[0][3]: tmp[0][0] is the sign

                    #TODO: hack because we change the number format
                    # tmp2 += '\\mathit{%+d}' % (int(tmp[1]) - 1)
                    tmp2 += ('\\textit{%+d}' % (int(tmp[1]) - 1)).replace('-', '--')
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
    #~ set_trace()
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

        for j in indexEntry.vData:
            if j[0] > maxEvals:
                break
        vals = scipy.sort(j[indexEntry.nbRuns+1:])
        #Get the function values for maxEvals.

        tmp = []
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


def main(indexEntries, valOfInterests, outputdir, info, verbose=True):
    """From a list of IndexEntry and a prefix, returns latex tabulars in files.
    args:
    info --- string suffix for output files.

    """

    filename = os.path.join(outputdir,'ppdata%s' % ('_' + info))
    for i in indexEntries:
        i.tabData = generateData(i, valOfInterests)

    #[header,format] = pproc.computevalues(None,None,header=True)
    #set_trace()

    writeTable2(indexEntries, filename, fontSize='tiny', verbose=verbose)
