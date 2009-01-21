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
idxF = 2

nbPtsEvals = 20;
nbPtsF = 5;

indexmainsep = ', '

# CLASS DEFINITIONS
class IndexEntry:
    """Unit element for the post-processing with given funcId, algId and
    dimension.
    Class attributes:
        funcId -- function Id (integer)
        dim -- dimension (integer)
        dataFiles -- associated data files (list)
        comment -- comment for the setting (string)
        targetFuncValue -- target function value (float)
        algId -- algorithm name (string)

        The next attributes are set after obtainData was called.
        hData -- collected data aligned by function values (array)
        vData -- collected data aligned by function evaluations (array)
        nbRuns -- collected data aligned by function evaluations (integer)
        mMaxEvals -- measured max. number of function evaluations (float)

    """

    def __init__(self, header, comment, data):
        """Instantiates an IndexEntry from 3 strings constituting an index 
        entry in an index file.

        """

        # Extract information from the line.
        self.__parseHeader(header)

        # Read in second line of entry (comment line). The information
        # is only stored if the line starts with "%", else it is ignored.
        if comment.startswith('%'):
            self.comment = comment.strip()
        else:
            print 'Warning: Comment line is skipped in %s,' % f.name
            print '         since it is not starting with a %s!' % '%'

        # Read next line of index file (data file and run informations).
        # Split line in data file name(s) and run time information. Then
        # store the data file name(s) in entry.dataFiles.
        self.dataFiles = []
        parts = data.split(', ')
        for elem in parts:
            if elem.endswith('dat'):
                #Windows data to Linux processing
                elem = elem.replace('\\', os.sep)
                #Linux data to Windows processing
                elem = elem.replace('/', os.sep)

                self.dataFiles.append(elem)
            else:
                pass
                # TODO: to store the run time information for each entry of
                # entry.dataFiles an attribute which contains another list
                # is needed

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
                'comment: "%s", f_t: %g' %(self.comment, self.targetFuncValue))


    def __parseHeader(self, header):
        """Extracts data from a header in the index files."""

        # Split input line in and create new list.
        headerList = header.split(indexmainsep)
        #split header into a list of key-value based on indexmainsep


        # Loop over all elements in the list and extract the relevant data.
        it = iter(headerList)
        while True:
            try:
                elem = it.next()

                #We need to catch the case where a value contains indexmainsep
                #This could happen when the key algId and the value is a string
                #and is caught by counting quotes.
                while elem.count("'")%2 != 0: 
                    elem += it.next()

                elemList = elem.split('=')
                elemFirst = elemList[0].strip()
                #Do not expect a '=' in the key name
                elemSecond = ''.join(elemList[1:]).strip()

                # Process elements
                # Assign values to the attributes of entry
                if elemFirst == 'funcId':
                    self.funcId = int(elemSecond)
                elif elemFirst == 'DIM':
                    self.dim = int(elemSecond)
                #elif elemFirst == 'maxEvals':
                    ## If maxEvals is 'Inf' it is stored as a string,
                    ## else as an integer.
                    #if elemSecond == 'Inf':
                        #self.maxEvals = elemSecond
                    #else:
                        #self.maxEvals = int(elemSecond)
                elif elemFirst == 'targetFuncValue':
                    self.targetFuncValue = float(elemSecond)
                elif elemFirst == 'Fopt':
                    self.fopt = float(elemSecond)
                elif elemFirst == 'Precision':
                    self.targetFuncValue = float(elemSecond) + self.fopt
                    #TODO: implies fopt has already been defined.
                    self.precision = float(elemSecond)
                elif elemFirst == 'algId':
                    self.algId = elemSecond
                else:
                    print 'Warning: Reading some header data'
                    print '         Can not handle quantity %s!' % elem

            except StopIteration:
                break
        return

    def obtainData_new(self):
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


    def obtainData(self):
        """Aligns all data and stores the result in hData and vData."""

        #TODO: rewrite this: it is huge and ugly.

        dataFiles = self.dataFiles[:]
        for i in range(len(dataFiles)):
            dataFiles[i] = '.'.join(dataFiles[i].split('.')[:-1]) + '.dat'
            #removes the extension and add '.dat' instead.
        dataSets = pproc.split(dataFiles)

        self.nbRuns = len(dataSets)

        hData = []

        #for horizontal alignment
        evals = len(dataSets) * [0] # updated list of function evaluations.
        f = len(dataSets) * [0] # updated list of function values.
        isRead = len(dataSets) * [True] # read status of dataSets.

        idxCurrentF = scipy.inf # Minimization
        currentF = scipy.power(10, float(idxCurrentF) / nbPtsF)

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
        try:
            self.hData = scipy.vstack(hData)
        except ValueError:
            #TODO: empty data!
            self.hData = []

        #set_trace()
        dataFiles = self.dataFiles[:]
        for i in range(len(dataFiles)):
            dataFiles[i] = '.'.join(dataFiles[i].split('.')[:-1]) + '.tdat'
            #removes the extension and add '.tdat' instead.

        dataSets2 = pproc.split(dataFiles)
        vData = []
        #for vertical alignment
        evals2 = len(dataSets2) * [0] # updated list of function evaluations.
        f2 = len(dataSets2) * [0] # updated list of function values.
        isRead2 = len(dataSets2) * [True] # read status of dataSets.

        idxCurrentEvals = 0
        idxDIMCurrentEvals = 0
        currentEvals = scipy.power(10, float(idxCurrentEvals) / nbPtsEvals)

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
            while (scipy.floor(currentEvals) >=
                   scipy.floor(scipy.power(10, 
                               float(idxCurrentEvals) / nbPtsEvals))):
                idxCurrentEvals += 1
            while (scipy.floor(currentEvals) >=
                   self.dim * scipy.power(10, idxDIMCurrentEvals)):
                idxDIMCurrentEvals += 1
            #set_trace()
            currentEvals = min(scipy.power(10, float(idxCurrentEvals) / nbPtsEvals),
                               self.dim * scipy.power(10, idxDIMCurrentEvals))

        try:
            self.vData = scipy.vstack(vData)
            self.mMaxEvals = max(vData[-1][1:self.nbRuns+1])
        except ValueError:
            #TODO: empty data!
            self.vData = []
            self.mMaxEvals = 0

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


class IndexEntries(list):
    """Set of instances of IndexEntry."""

    #Do not inherit from set because IndexEntry instances are mutable.
    def __init__(self, indexFiles=[], verbose=True):
        """Instantiate an IndexEntry from a list of index files."""

        if not indexFiles:
            self = list()
            return

        for elem in indexFiles:
            if isinstance(elem, str):
                try:
                    if not elem.endswith('info'):
                        print 'Warning: %s might not be an index file.' % elem
                    f = open(elem)
                    if verbose:
                        print 'Processing %s.' % elem
                except IOError:
                    print 'Could not find %s.' % elem
                    continue
            else:
                raise Exception('Expect a file name.')

            # Read all data sets within one index file.
            while True:
                try:
                    header = f.next()
                    comment = f.next()
                    tmpline = f.next()
                    # Add the path to the index file to the data files in 
                    # tmpline.
                    data = []
                    for i in tmpline.split(indexmainsep):
                        if i.endswith('dat'):
                            data.append(os.path.join(os.path.split(elem)[0],
                                                     i))
                        else:
                            data.append(i)
                    data = indexmainsep.join(data)

                    self.append(IndexEntry(header, comment, data))
                except StopIteration:
                    break

            # Close index file
            f.close()

        # TODO: While an indexEntry cannot be created by incrementally
        # appending the data files, we need this loop.
        for i in self:
            i.obtainData()
            if verbose:
                print 'Obtaining data for %s' % i.__repr__()

    def append(self, o):
        """Redefines the append method to check for unicity."""

        if not isinstance(o, IndexEntry):
            raise Exception()
        isFound = False
        for i in self:
            if i == o:
                isFound = True
                i.dataFiles.extend(o.dataFiles)
                break
        if not isFound:
            list.append(self, o)

    def sortByDim(self):
        """Returns a dictionary sorted based on indexEntry.dim."""
        sorted = {}
        for i in self:
            sorted.setdefault(i.dim, IndexEntries()).append(i)
        return sorted

    def sortByFunc(self):
        """Returns a dictionary sorted based on indexEntry.dim."""
        sorted = {}
        for i in self:
            sorted.setdefault(i.funcId, IndexEntries()).append(i)
        return sorted

    def sortByFuncGroup(self):
        """Returns a dictionary sorted based on the function group."""

        #% SEPARABLE
        #1 Sphere
        #2 Ellipsoid separable with monotone x-transformation, condition 1e6
        #3 Rastrigin separable with asymmetric x-transformation "condition" 10
        #4 Skew Rastrigin-Bueche separable, "condition" 10, skew-"condition" 100
        #5 Linear slope, neutral extension outside the domain (but not flat)

        #% LOW OR MODERATE CONDITION
        #6 Attractive sector
        #7 Step-ellipsoid, condition 100
        #8 Rosenbrock, non-rotated
        #9 Rosenbrock, rotated

        #% HIGH CONDITION
        #10 Ellipsoid with monotone x-transformation, condition 1e6
        #11 Discus with monotone x-transformation, condition 1e6
        #12 Bent cigar with asymmetric x-transformation, condition 1e6
        #13 Sharp ridge, slope 1:100, condition 10
        #14 Sum of different powers

        #% MULTI-MODAL
        #15 Rastrigin with asymmetric x-transformation, "condition" 10
        #16 Weierstrass with monotone x-transformation, condition 100
        #17 Schaffer F7 with asymmetric x-transformation, condition 10
        #18 Schaffer F7 with asymmetric x-transformation, condition 1000
        #19 F8F2 composition of 2-D Griewank-Rosenbrock

        #% MULTI-MODAL WITH WEAK GLOBAL STRUCTURE
        #20 Schwefel x*sin(x) with tridiagonal transformation, condition 10
        #21 Gallagher 99 Gaussian peaks, global rotation, condition up to 1000
        #22 Gallagher 99 Gaussian peaks, local rotations, condition up to 1000
        #23 Katsuura
        #24 Lunacek bi-Rastrigin, condition 100

        sorted = {}
        #The bins correspond to whether the dimension is greater than 10 or not.
        #Still needs to be sorted by functions.
        for i in self:
            if i.funcId in range(0,6):
                sorted.setdefault('separ', []).append(i)
            elif i.funcId in range(6,10):
                sorted.setdefault('lcond', []).append(i)
            elif i.funcId in range(10,15):
                sorted.setdefault('hcond', []).append(i)
            elif i.funcId in range(15,20):
                sorted.setdefault('multi', []).append(i)
            elif i.funcId in range(20,25):
                sorted.setdefault('mult2', []).append(i)
        return sorted

##############################################################################

def main(indexFiles, indexEntries = IndexEntries(), verbose = True):
    """Extends or creates an IndexEntries from a list of indexFiles."""

    indexEntries.extend(IndexEntries(indexFiles, verbose))

    return indexEntries


if __name__ == "__main__":
    sys.exit(main())
