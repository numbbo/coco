#! /usr/bin/env python

# Script to read the data from the files within the list indexFiles
# and store it in the list indexEntries.

from __future__ import absolute_import

import sys
import os
import scipy

from bbob_pproc import findindexfiles
from bbob_pproc import pproc

from pdb import set_trace


# GLOBAL VARIABLES
idxEvals = 0
idxF = 1

nbPtsEvals = 20;
nbPtsF = 5;

# FUNCTION DEFINITIONS
def headerInfo(header,entry):
    """Extracts data from the header in the index files and stores them
     in entry, an instance of the class IndexEntry. It returns the parameter entry
     with the correct attributes.

     """

    # Split input line in and create new list.
    headerList = header.split(',')
    # TODO: is it possible that other separators will be used?

    # Loop over all elements in the list and extract the relevant data.
    for elem in headerList:
        elemList = elem.split('=')
        # Process elements
        elemFirst = elemList[0].strip()
        elemSecond = elemList[1].strip()
        # Assign values to the attributes of entry
        if elemFirst == 'funcId':
            entry.funcId = int(elemSecond)
        elif elemFirst == 'DIM':
            entry.dim = int(elemSecond)
        elif elemFirst == 'maxEvals':
            # If maxEvals is 'Inf' it is stored as a string,
            # else as an integer.
            if elemSecond == 'Inf':
                entry.maxEvals = elemSecond
            else:
                entry.maxEvals = int(elemSecond)
        elif elemFirst == 'targetFuncValue':
            entry.targetFuncValue = float(elemSecond)
        elif elemFirst == 'Fopt':
            entry.fopt = float(elemSecond)
        elif elemFirst == 'Precision':
            entry.targetFuncValue = float(elemSecond) + entry.fopt
            entry.precision = float(elemSecond)
        elif elemFirst == 'algId':
            entry.algId = elemSecond
        else:
            print 'Warning: Reading the header data from %s!' % f.name
            print '         Can not handle quantity %s!' % elem
    return entry


# CLASS DEFINITIONS
class IndexEntry:
    """Unit element for the post-processing with given funcId, algId and
    dimension.

    """

    def __init__(self):
        self.funcId = 0    # function Id (integer)
        self.dim = 0        # dimension (integer)
        self.dataFiles = list()    # associated data files (list)
        self.comment = 'no comment'    # comment for the setting (string)
        self.maxEvals = 0    # max. number of function evaluations
        self.mMaxEvals = 0    # measured max. number of function evaluations
        self.targetFuncValue = 0.0     # target function value (float)
        self.algId = 'no name'    # algorithm name (string)
        self.hData = None    # collected data aligned by function values
        self.vData = None    # collected data aligned by function evaluations
        self.nbRuns = 0    # collected data aligned by function evaluations

    def __eq__(self,other):
        return (self.__class__ is other.__class__ and
                self.funcId == other.funcId and
                self.dim == other.dim and
                self.algId == other.algId)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __repr__(self):
        return ('alg: "%s", F%d, dim: %d, dataFiles: %s, '
                % (self.algId, self.funcId, self.dim, str(self.dataFiles)) +
                'comment: "%s", maxEvals: %d,' %(self.comment, self.maxEvals) +
                ' f_t: %g' % (self.targetFuncValue))

    def obtainData(self):
        """Gets aligned data and stores the result in hData and vData."""

        dataSets = pproc.split(self.dataFiles)
        self.nbRuns = len(dataSets)
        self.hData = pproc.alignData(dataSets, 'horizontal')

        dataFiles = []
        for fil in self.dataFiles:
            dataFiles.append(fil.replace('.dat', '.tdat'))

        dataSets = pproc.split(dataFiles)
        self.vData = pproc.alignData(dataSets, 'vertical')

        return


    def obtainData_old(self):
        """Deprecated: fgeneric outputs aligned data now.
        Aligns all data and stores the result in hData and vData.

        """

        dataSets = pproc.split(self.dataFiles)

        dataFiles = []
        for fil in self.dataFiles:
            dataFiles.append(fil.replace('.dat', '.tdat'))

        dataSets2 = pproc.split(dataFiles)
        self.nbRuns = len(dataSets)

        hData = []
        vData = []

        #for horizontal alignment
        evals = len(dataSets) * [0] # updated list of function evaluations.
        f = len(dataSets) * [0] # updated list of function values.
        isRead = len(dataSets) * [True] # read status of dataSets.

        #for vertical alignment
        evals2 = len(dataSets2) * [0] # updated list of function evaluations.
        f2 = len(dataSets2) * [0] # updated list of function values.
        isRead2 = len(dataSets2) * [True] # read status of dataSets.

        idxCurrentF = scipy.inf # Minimization
        currentF = scipy.power(10, idxCurrentF / 5)
        idxCurrentEvals = 0
        currentEvals = scipy.power(10, idxCurrentEvals / 20.)

        # Parallel construction of the post-processed arrays:
        #set_trace()
        while any(isRead):
            for i in range(len(dataSets)):
                curDataSet = dataSets[i]
                if isRead[i]:
                    evals[i] = curDataSet.set[curDataSet.currentPos, idxEvals]
                    f[i] = curDataSet.set[curDataSet.currentPos, idxF]
                    while (curDataSet.currentPos < len(curDataSet.set) - 1 and
                           f[i] > currentF): # minimization
                        curDataSet.currentPos += 1
                        evals[i] = curDataSet.set[curDataSet.currentPos,
                                                  idxEvals]
                        f[i] = curDataSet.set[curDataSet.currentPos, idxF]
                    if not (curDataSet.currentPos < len(curDataSet.set) - 1):
                        isRead[i] = False

            if max(f) <= 0: #TODO: Issue if max(f) < 0
                idxCurrentF = -scipy.inf
                currentF = 0.
            else:
                # TODO modify this line so that idxCurrentF is the max of the f
                # where isRead still is True.
                idxCurrentF = min(idxCurrentF, 
                                  int(scipy.ceil(scipy.log10(max(f)) * 
                                                 nbPtsF)))
                currentF = scipy.power(10, float(idxCurrentF) / nbPtsF)
            #set_trace()
            tmp = [currentF]
            tmp.extend(evals)
            tmp.extend(f)
            hData.append(tmp)
            idxCurrentF -= 1
            currentF = scipy.power(10, float(idxCurrentF) / nbPtsF)

        #set_trace()

        while any(isRead2):
            for i in range(len(dataSets2)):
                curDataSet = dataSets2[i]

                #set_trace()
                if isRead2[i]:
                    evals2[i] = curDataSet.set[curDataSet.currentPos2, idxEvals]
                    f2[i] = curDataSet.set[curDataSet.currentPos2, idxF]
                    while (curDataSet.currentPos2 < len(curDataSet.set) - 1 and
                           evals2[i] < scipy.floor(currentEvals)):
                        curDataSet.currentPos2 += 1
                        evals2[i] = curDataSet.set[curDataSet.currentPos2,
                                                   idxEvals]
                        f2[i] = curDataSet.set[curDataSet.currentPos2, idxF]
                    if not (curDataSet.currentPos2 < len(curDataSet.set) - 1):
                        isRead2[i] = False
            tmp = [scipy.floor(currentEvals)]
            tmp.extend(evals2)
            tmp.extend(f2)
            vData.append(tmp)
            if currentEvals < 10:
                currentEvals += 1
            else:
                while (scipy.floor(currentEvals) >=
                       scipy.floor(scipy.power(10,
                                   float(idxCurrentEvals) / nbPtsEvals))):
                    idxCurrentEvals += 1
                currentEvals = scipy.power(10, float(idxCurrentEvals) / nbPtsEvals)

        self.mMaxEvals = max(vData[-1][1:self.nbRuns+1])
        #set_trace()
        self.hData = scipy.vstack(hData)
        self.vData = scipy.vstack(vData)
        #set_trace()
        return


class Error(Exception):
    """ Base class for errors. """
    pass


class MissingValueError(Error):
    """ Error if a mandatory value is not found within a file.
        Returns a message with the missing value and the respective
        file.

    """

    def __init__(self,value, filename):
        self.value = value
        self.filename = filename

    def __str__(self):
        message = 'The value %s was not found in file %s!' % \
                  (self.value, self.filename)
        return repr(message)


##############################################################################

def main(indexFiles, indexEntries = [], verbose = True):
    # main code

    # Read all index files within indexFiles.
    for elem in indexFiles:
        f = open(os.path.join(elem.path,elem.name))
        if verbose:
            print 'Processing file %s ....' % os.path.join(elem.path,elem.name)
        # Read all data sets within one index file.
        while True:
            try:
                # Create new instance of indexEntry.
                entry = IndexEntry()
                entry.dataFiles = list()

                # Read in first line of new entry (header).
                line = f.next()
                # Extract information from the line.
                #set_trace()
                entry = headerInfo(line, entry)
                # If entry.funcId and entry.dim contain the default values for
                # creating an instance of the class IndexEntry, the information
                # in the (supposed) header is not sufficient.
                if entry.funcId == 0: raise MissingValueError('funcId',f.name)
                if entry.dim == 0: raise MissingValueError('DIM',f.name)
                # Read in second line of entry (comment line). The information
                # is only stored if the line starts with "%", else it is ignored.
                line = f.next()
                if line.startswith('%'):
                    entry.comment = line.strip()
                else:
                    print 'Warning: Comment line is skipped in %s,' % f.name
                    print '         since it is not starting with a %s!' % '%'

                # Read next line of index file (data file and run informations).
                line = f.next()
                # Split line in data file name(s) and run time information. Then
                # store the data file name(s) in entry.dataFiles.
                parts = line.split(', ')
                for elem2 in parts:
                    if elem2.endswith('.dat'):
                        #Windows data to Linux processing
                        elem2 = elem2.replace('\\',os.sep)
                        #Linux data to Windows processing
                        elem2 = elem2.replace('/',os.sep)
                        entry.dataFiles.append(os.path.join(elem.path, elem2))
                    else:
                        pass
                        # TODO: to store the run time information for each entry of
                        # entry.dataFiles an attribute which contains another list
                        # is needed

                # Only create new entry in indexEntries if no entry exists yet.
                if len(indexEntries) == 0:
                    indexEntries.append(entry)
                else:
                    # First check if an entry already exists with the same
                    # combination of the attributes funcId, dim, and algId
                    # already exists. If so, append all elements of
                    # entry.dataFiles to this entry. Else, create a new entry
                    # in indexEntries.
                    added = False
                    for elem2 in indexEntries:
                        if entry == elem2:
                            # Append all data files to existing entry
                            elem2.dataFiles.extend(entry.dataFiles)
                            added = True
                            break
                    if not added:
                        # Create a new entry
                        indexEntries.append(entry)
            except StopIteration:
                break
        # Close index file
        f.close()

    for i in indexEntries:
        i.obtainData_old()
        #set_trace()

    return indexEntries

if __name__ == "__main__":
    sys.exit(main())

# TODO: *somehow it works not on my home laptop
#         to execute 'python findIndexFiles.py'
#       (ImportError: No module named numpy)
