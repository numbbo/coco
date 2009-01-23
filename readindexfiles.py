#! /usr/bin/env python

"""Defines class IndexEntry, unit element in the post-processing and the list
of instances of IndexEntry: IndexEntries.

"""

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

    # Private attribute used for the parsing of info files.
    __attributes = {'funcId': ('funcId', int), 'DIM': ('dim',int),
                    'Precision': ('precision', float), 'Fopt': ('fopt', float),
                    'targetFuncValue': ('targetFuncValue', float),
                    'algId': ('algId', str)}

    def __init__(self, header, comment, data):
        """Instantiate an IndexEntry from 3 strings constituting an index
        entry in an index file.

        """

        # Extract information from the header line.
        self.__parseHeader(header)

        # Read in second line of entry (comment line). The information
        # is only stored if the line starts with "%", else it is ignored.
        if comment.startswith('%'):
            self.comment = comment.strip()
        else:
            raise Exception()
            print 'Warning: Comment line: "%s" is skipped,' % comment
            print '         since it is not starting with a \%!'
            self.comment = ''

        # Split line in data file name(s) and run time information.
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

    def __eq__(self, other):
        """Compare indexEntry instances."""
        return (self.__class__ is other.__class__ and
                self.funcId == other.funcId and
                self.dim == other.dim and
                self.algId == other.algId)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __repr__(self):
        return ('{alg: %s, F%d, dim: %d}'
                % (self.algId, self.funcId, self.dim))

    def __parseHeader(self, header):
        """Extract data from a header line in an index entry."""

        # Split header into a list of key-value based on indexmainsep
        headerList = header.split(indexmainsep)

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
                #A key name is not expected to contain the string '='
                elemFirst = elemList[0].strip()
                elemSecond = ''.join(elemList[1:]).strip().strip('\'')
                #TODO: use the quotes instead of stripping them to discern
                #between data types?

                try:
                    setattr(self, self.__attributes[elemFirst][0],
                            self.__attributes[elemFirst][1](elemSecond))
                except KeyError:
                    print ('%s is not an expected attribute of IndexEntry.'
                           % elemFirst)
                    #setattr(self, elemFirst, elemSecond)
                    continue

            except StopIteration:
                break

        #TODO: check that no compulsory attributes is missing:
        #dim, funcId, algId and?

        self.targetFuncValue = self.fopt + self.precision
        #TODO: is this needed?
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

        #set_trace()
        # Parallel construction of the post-processed arrays:
        while (any(isRead) or any(f <= currentF)) and currentF != 0.:
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

            tmp = [] #Get the f that are still of interest.
            for i in f:
                if i <= currentF:
                    tmp.append(i)
            if max(tmp) <= 0: #TODO: Issue if max(f) < 0
                idxCurrentF = -scipy.inf
                currentF = 0.
            else:
                idxCurrentF = min(idxCurrentF,
                                  int(scipy.ceil(scipy.log10(max(tmp)) *
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

        #TODO do not align on the chosen evals but the one we get!
        dataSets2 = pproc.split(dataFiles)
        vData = []
        #for vertical alignment:
        evals2 = len(dataSets2) * [0] # updated list of function evaluations.
        f2 = len(dataSets2) * [0] # updated list of function values.
        isRead2 = len(dataSets2) * [True] # read status of dataSets.

        currentEvals = 1

        while any(isRead2):
            nextCurEvals = []
            for i in range(len(dataSets2)):
                curDataSet = dataSets2[i]

                #set_trace()
                if isRead2[i]:
                    while (curDataSet.currentPos2 < len(curDataSet.set) and
                           (curDataSet.set[curDataSet.currentPos2, idxEvals]
                            <= currentEvals)):
                        evals2[i] = curDataSet.set[curDataSet.currentPos2,
                                                   idxEvals]
                        f2[i] = curDataSet.set[curDataSet.currentPos2, idxF]
                        curDataSet.currentPos2 += 1

                    if (curDataSet.currentPos2 < len(curDataSet.set)):
                        nextCurEvals.append(curDataSet.set[curDataSet.currentPos2, idxEvals])
                    else:
                        isRead2[i] = False

            tmp = [currentEvals]
            tmp.extend(evals2)
            tmp.extend(f2)
            vData.append(tmp)
            if nextCurEvals: #Else isRead2 is supposed to be all false.
                currentEvals = min(nextCurEvals)

        try:
            self.vData = scipy.vstack(vData)
            self.mMaxEvals = max(vData[-1][1:self.nbRuns+1])
        except ValueError:
            #TODO: empty data!
            self.vData = []
            self.mMaxEvals = 0

        return


#class Error(Exception):
    #""" Base class for errors. """
    ##TODO: what is this for?
    #pass


#class MissingValueError(Error):
    #""" Error if a mandatory value is not found within a file.
        #Returns a message with the missing value and the respective
        #file.

    #"""

    ##TODO: what is this for?
    #def __init__(self,value, filename):
        #self.value = value
        #self.filename = filename

    #def __str__(self):
        #message = 'The value %s was not found in file %s!' % \
                  #(self.value, self.filename)
        #return repr(message)


class IndexEntries(list):
    """Set of instances of IndexEntry, implement some useful slicing functions.

    This class is not needed elsewhere than in __init__.main. Everywhere else,
    only a sequence of IndexEntry is needed.

    """

    #Do not inherit from set because IndexEntry instances are mutable.

    def __init__(self, indexFiles=[], verbose=True):
        """Instantiate an IndexEntries from a list of index files.

        If provided the list of index files will be splitted into index entries
        which will instantiate IndexEntry and be added to this.

        """

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
            if verbose:
                print 'Obtaining data for %s' % i.__repr__()
            i.obtainData()

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
