#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""BBOB 2010 tables ERT vs target function values.
One algorithm, one table per function for k dimensions."""

from __future__ import absolute_import

import os
import numpy
from bbob_pproc import bestalg, bootstrap
#from bbob_pproc.pptex import 

from pdb import set_trace

#GLOBAL VARIABLES DEFINITION
percentiles = [10, 90]
header = ['$\Delta f$', '$\#$', '$\ERT$', '%d\%%' % percentiles[0],
          '%d\%%' % percentiles[1], '$\\text{RT}_{\\text{succ}}$']
format = ['%1.1e', '%d', '%1.1e', '%1.1e', '%1.1e', '%1.1e']
#These have to be synchronized with what's computed in generateData.

#METHODS
def writeTable(entries, filename, fontSize='scriptsize',
                fontSize2='scriptstyle', verbose=True):
    """Writes data of array in a *.tex file. This file then can be used
       in LaTex as input for tables. The input must be placed inside a
       table environment.

       Mandatory inputs:
       entries - Sequence of IndexEntry to be displayed.
       filename - name of output file (string)

       Optional inputs:
       fontSize - size of fonts as they appear in the LaTex table (string)
       fontSize2 - size of the fonts in the math environment (string)

    """

    #TODO cleanup tex ouput.
    # Assemble header for whole table (contains more than 1 dimension)
    width = len(entries)

    fullheader = header + (width-1)*header[1:]
    fullformat = format + (width-1)*format[1:]

    data = []
    suppr = []
    it = iter(entries)
    tmp = it.next().tabData
    data.append(tmp)
    tmp2 = [len(tmp[0]) * [False]]
    for i in range(1, len(tmp)):
        if tmp[i-1][1] == tmp[i][1] == 0: # and all(tmp[i][2:] == tmp[i+1][2:])
            tmp2.append([False] + (len(tmp[i])-1) * [True])
        else:
            tmp2.append([False] * len(tmp[i]))
    suppr.append(numpy.vstack(tmp2))

    while True:
        try:
            tmp = it.next().tabData[:, 1:]
            data.append(tmp)
            tmp2 = [len(tmp[0]) * [False]]
            for i in range(1, len(tmp)):
                tmp2.append(len(tmp[i]) * [tmp[i-1][0] == tmp[i][0] == 0])
            #set_trace()
            suppr.append(numpy.vstack(tmp2))

        except StopIteration:
            break

    data = numpy.transpose(numpy.concatenate(data, 1))
    suppr = numpy.transpose(numpy.concatenate(suppr, 1))
    #Transposing, otherwise the concatenation might not behave as expected

    # Input checking and reshape data
    #if len(fullformat) != data.shape[0] and len(fullformat) != data.shape[1]:
        #raise WrongInputSizeError('data',data.shape,len(format))
    #elif len(fullformat) != len(fullheader) :
        #raise WrongInputSizeError('header',len(fullheader),len(format))
    #elif len(fullformat) != data.shape[0]:
        #data = numpy.transpose(data)

    # Generate LaTex commands for vertical lines and aligment of the entries.
    tabColumns = '@{$\;$}c@{$\;$}'
    tabColumns += ('|' + (len(header) - 1) * '@{$\;$}c@{$\;$}') * width

    # Create output file
    if verbose:
        if os.path.exists(filename):
            print 'Overwrite old file %s!' %(filename + '.tex')
        else:
            print 'Write in %s.' %(filename+'.tex')

    try:
        f = open(filename + '.tex','w')
    except ValueError:
        print 'Error opening ' + filename + '.tex'

    # Write tabular environment
    f.write('\\begin{' + fontSize + '} \n')
    f.write('\\begin{tabular}{' + tabColumns + '} \n')
    #f.write('\\begin{tabularx}{0.47\\textwidth}{' + tabColumns + '} \n')

    # Write first two rows containing the info of the table columns
    for i in range(0, width):
        caption = ('\\textbf{\\textit{f}\\raisebox{-0.35ex}{' + str(entries[i].funcId) +
                   '} in ' + str(entries[i].dim)) + '-D}'
        caption = caption + ', N=' + str(entries[i].nbRuns())
        maxEvals = entries[i].mMaxEvals()
        caption = caption + ', mFE='
        if maxEvals >= 1e6:
            strMaxEvals = '%.2e' % (maxEvals)
            # the numbers of significant figures was set so that even if maxEvals
            # is greater than 10 (and lesser than 99), it should still fit.
            (mantissa, exponent) = strMaxEvals.split('e')
            exponent = exponent.lstrip('+0')
            #Remove leading sign and zeros
            caption += ('$%s\\mathrm{\\hspace{0.10em}e}%s$'
                        % (mantissa, exponent))
        else:
            caption += str(int(maxEvals))
        if i != width - 1:
            f.write('& \multicolumn{' + str(len(format)-1) + '}{@{$\;$}c|@{$\;$}}{' + caption + '}')
        else:
            f.write('& \multicolumn{' + str(len(format)-1) + '}{@{$\;$}c@{$\;$}}{' + caption + '}')
    f.write('\\\\ \n')
    f.write(' & '.join(fullheader) + '\\\\ \n \hline \n')

    # Write data
    for i in range(0,data.shape[1]):
        writeArray(f, data[:,i], fullformat, fontSize2,
                   suppress_entry = suppr[:,i]) # only one line of f-values

    # Finish writing the table and close file.
    f.write('\end{tabular} \n')
    #f.write('\end{tabularx} \n')
    f.write('\end{' + fontSize + '} \n')
    #f.write('\end{table*} \n')

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
        if i % 5 == 0:
            isFunctionValues = False
        elif i % 5 == 1: # Find the entries for the number of successes
            if x == 0:
                isFunctionValues = True

        #print type(x)
        #print len(vector)

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

            if i == 0:  # Delta f value
                if x >= 1 and x <= 100 and x == round(x):
                    tmp2 = str(int(round(x)))  # tmp[0][0]
                else:
                    sgn = '+'  # don't assume that a + sign is present
                    if x < 1:
                        sgn = '-'
                    tmp2 = (tmp[0][0] + '\\!\\mathrm{\\hspace{0.10em}e}' +
                            sgn + tmp[1][-1])
            else:
                if isFunctionValues:
                    # tmp[0][0] + tmp[0][2]: format change...
                    tmp2 = ('\\textit{' + tmp[0][0] + tmp[0][2] + '}' +  # textit is narrower
                            '\\hspace{0.00em}e')

                    #TODO: hack because we change the number format
                    tmp2 += ('\\textit{%+d}' % (int(tmp[1]) - 1)).replace('-', '--')
                else:
                    if x < 0:
                        x *= -1
                        tmp = str('%1.e'%x).split('e')
                        tmp2 = '>' + tmp[0][0] + '\\mathrm{\\hspace{0.10em}e}' + tmp[1][-1]
                    else:
                        tmp2 = tmp[0] + '\\mathrm{\\hspace{0.10em}e}' + tmp[1][-1]
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

def generateData(dataSet, targetFuncValues, samplesize=1000):
    """Returns data for the tables.

    Data are supposed to be function-value aligned on some specific function
    values, information on the number of function evaluations to reach the
    target function value complete a row in the resulting array. If some runs
    from dataSet did not reach the target function value, information on the
    final function value reached by these runs will be stored instead.

    Keyword arguments:
    dataSet -- input DataSet instance.
    targetFuncValues -- function values to be displayed
    samplesize -- sample size used for the bootstrapping. The larger this value
    is the longer it takes.

    Outputs:
    Array of data to be displayed in the tables.

    """

    #TODO loop over header to know what to append to curLine.
    res = []
    it = iter(dataSet.evals)
    i = it.next()

    for targetF in reversed(sorted(targetFuncValues)):
        while i[0] > targetF:
            try:
                i = it.next()
            except(StopIteration):
                break

        if i[0] <= targetF: # if at least one success
            N = i.copy()[1:]
            success = numpy.isfinite(N)
            N[numpy.isnan(N)] = dataSet.maxevals[numpy.isnan(N)]

            ertvec = bootstrap.sp(N, issuccessful=success, allowinf=False)

            sumfevals = numpy.sum(N)
            if any(success):
                dispersion = bootstrap.drawSP(N[success], N[success==False],
                                              percentiles, samplesize)[0]
            else:
                #Should never occur...
                dispersion = [-sumfevals]*percentiles
                # we put a negative number to have a different output in
                # the tables.

            curLine = [targetF, ertvec[2], ertvec[0],
                       dispersion[0], dispersion[1],
                       numpy.mean(N[success])]

        else: # 0 success.
            vals = numpy.sort(dataSet.finalfunvals)
            #Get the function values for maxEvals.

            curLine = [targetF, 0] # 0 success
            tmp = [50]
            tmp.extend(percentiles)
            # it's the same prctiles for the funvals as for the bootstrapping.
            curLine.extend(bootstrap.prctile(vals, tmp))

            # unsucc is an array of the largest number of function evaluations
            # for which f is STRICTLY larger to fbest
            unsucc = []
            for j in range(1, dataSet.nbRuns()+1):
                # Get the alignment number of function evaluations
                # corresponding to the 1st occurrence of the best function
                # value obtained.
                k = -1
                while (dataSet.funvals[k - 1, j] == dataSet.finalfunvals[j-1]
                       and k > -len(dataSet.funvals) + 1):
                    k -= 1
                unsucc.append(dataSet.funvals[k, 0])
            curLine.append(bootstrap.prctile(unsucc, [50], issorted=False)[0])

        res.append(curLine)
    return numpy.vstack(res)

def main(dsList, valOfInterests, filename, isDraft=False, verbose=True):
    """Generates latex tabular from a DataSetList.

    Keyword arguments:
    dsList -- list of DataSet instances to put together in a tabular.
    valOfInterests -- list of function values to be displayed.
    filename -- output file name.
    isDraft -- if true, quickens the bootstrapping step.
    verbose -- controls verbosity.
    """

    samplesize = 3000
    if isDraft:
        samplesize = 100

    for i in dsList:
        curValOfInterests = list(j[i.funcId] for j in valOfInterests)
        i.tabData = generateData(i, curValOfInterests, samplesize)

    writeTable(dsList, filename, fontSize='tiny', verbose=verbose)

