#! /usr/bin/env python

"""Defines class IndexEntry, unit element in the post-processing and the list
of instances of IndexEntry: IndexEntries.

"""

from __future__ import absolute_import

import os
import pickle
import warnings
import numpy

from bbob_pproc import findindexfiles
from bbob_pproc.pproc import split, alignData, HMultiReader, VMultiReader
from bbob_pproc.pproc import HArrayMultiReader, VArrayMultiReader

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

        The following attributes/methods are set after obtainData was called:
        hData -- collected data aligned by function values (array)
        vData -- collected data aligned by function evaluations (array)
        nbRuns -- number of runs (integer)
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
            #raise Exception()
            warnings.warn('Comment line: %s is skipped,' % (comment) +
                          'it does not start with \%.')
            self.comment = ''

        # Split line in data file name(s) and run time information.
        self.dataFiles = []
        self.itrials = []
        self.evals = []
        self.finalFminusFtarget = []
        parts = data.split(', ')
        for elem in parts:
            if elem.endswith('dat'):
                #Windows data to Linux processing
                elem = elem.replace('\\', os.sep)
                #Linux data to Windows processing
                elem = elem.replace('/', os.sep)

                self.dataFiles.append(elem)
            else:
                elem = elem.split(':')
                self.itrials.append(int(elem[0]))
                elem = elem[1].split('|')
                self.evals.append(int(elem[0]))
                self.finalFminusFtarget.append(float(elem[1]))
                #pass

        #set_trace()
        ext = {'.dat':(HMultiReader, 'hData'), '.tdat':(VMultiReader, 'vData')}
        for extension, info in ext.iteritems():
            #set_trace()
            dataFiles = list('.'.join(i.split('.')[:-1]) + extension
                             for i in self.dataFiles)
            data = info[0](split(dataFiles))
            setattr(self, info[1], alignData(data))
        #set_trace()

    def __eq__(self, other):
        """Compare indexEntry instances."""
        return (self.__class__ is other.__class__ and
                self.funcId == other.funcId and
                self.dim == other.dim and
                self.algId == other.algId and
                self.comment == other.comment)

    def __ne__(self,other):
        return not self.__eq__(other)

    def __repr__(self):
        return ('{alg: %s, F%d, dim: %d}'
                % (self.algId, self.funcId, self.dim))

    def nbRuns(self):
        return (numpy.shape(self.vData)[1]-1)/2

    def mMaxEvals(self):
        return self.vData[-1,0]

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
                #Here we strip quotes instead of using them to differentiate
                #between data types.

                try:
                    setattr(self, self.__attributes[elemFirst][0],
                            self.__attributes[elemFirst][1](elemSecond))
                except KeyError:
                    warnings.warn('%s is not an expected ' % (elemFirst) +
                                  'attribute of IndexEntry.')
                    continue

            except StopIteration:
                break

        #TODO: check that no compulsory attributes is missing:
        #dim, funcId, algId, precision

        return

    def pickle(self, outputdir, verbose=True):
        if not getattr(self, 'pickleFile', False):
            self.pickleFile = os.path.join(outputdir, 'ppdata_f%d_%d.pickle'
                                           %(self.funcId, self.dim))
            try:
                f = open(self.pickleFile, 'w')
                pickle.dump(self, f)
                f.close()
                if verbose:
                    print 'Saved pickle in %s.' %(self.pickleFile)
            except IOError, (errno, strerror):
                print "I/O error(%s): %s" % (errno, strerror)
            except pickle.PicklingError:
                print "Could not pickle %s" %(self)
        #else: #What?
            #if verbose:
                #print ('Skipped update of pickle file %s: no new data.'
                       #% self.pickleFile)

    def nbSuccess(self, targetFuncValue):
        """Returns the nb of successes for a given target function value."""

        succ = 0
        for j in self.hData:
            if j[0] <= targetFuncValue:
                for k in j[i.nbRuns()+1:]:
                    if k <= targetFuncValue:
                        succ += 1
                break
        return succ


class IndexEntries(list):
    """Set of instances of IndexEntry, implement some useful slicing functions.

    """

    #Do not inherit from set because IndexEntry instances are mutable.

    def __init__(self, args=[], verbose=True):
        """Instantiate IndexEntries from a list of inputs.
        Keyword arguments:
        args -- list of strings being either info file names, folder containing
                info files or pickled data files.
        verbose -- controls verbosity.

        Outputs:
        indexEntries -- list of IndexEntry instances.

        Exception:
        Warning -- Unexpected user input.
        pickle.UnpicklingError

        """

        if not args:
            self = list()
            return

        for i in args:
            if i.endswith('.info'):
                self.processIndexFile(i)
            elif os.path.isdir(i):
                for j in findindexfiles.main(i,verbose):
                    self.processIndexFile(j)
            elif i.endswith('.pickle'):
                try:
                    f = open(i,'r')
                    try:
                        entry = pickle.load(f)
                    except pickle.UnpicklingError:
                        print '%s could not be unpickled.' %(i)
                    f.close()
                    if verbose:
                        print 'Unpickled %s.' % (i)
                    self.append(entry)
                    #set_trace()
                except IOError, (errno, strerror):
                    print "I/O error(%s): %s" % (errno, strerror)

            else:
                warnings.warn('File or folder ' + i + ' not found. ' +
                              'Expecting as input argument either .info ' +
                              'file(s), .pickle file(s) or a folder ' +
                              'containing .info file(s).')


    def processIndexFile(self, indexFile, verbose=True):
        """Reads in an index file information on the different runs."""

        try:
            f = open(indexFile)
            if verbose:
                print 'Processing %s.' % indexFile

            # Read all data sets within one index file.
            indexpath = os.path.split(indexFile)[0]
            while True:
                try:
                    header = f.next()
                    comment = f.next()
                    tmpline = f.next()
                    # Add the path to the index file to the file names
                    data = []
                    for i in tmpline.split(indexmainsep):
                        if i.endswith('dat'): #filenames
                            data.append(os.path.join(indexpath, i))
                        else: #other information
                            data.append(i)
                    data = indexmainsep.join(data)
                    self.append(IndexEntry(header, comment, data))
                except StopIteration:
                    break

        except IOError:
            print 'Could not open %s.' % indexFile

            # Close index file
            f.close()

    def append(self, o):
        """Redefines the append method to check for unicity."""

        if not isinstance(o, IndexEntry):
            raise Exception()
        isFound = False
        for i in self:
            if i == o:
                isFound = True
                if set(i.dataFiles).symmetric_difference(set(o.dataFiles)):
                    #set_trace()
                    i.vData = alignData(VArrayMultiReader([i.vData, o.vData]))
                    #set_trace()
                    i.hData = alignData(HArrayMultiReader([i.hData, o.hData]))
                    if getattr(i, 'pickleFile', False):
                        del i.pickleFile
                break
        if not isFound:
            list.append(self, o)

    def pickle(self, outputdir, verbose=True):
        """Loop over self to pickle each elements."""

        for i in self:
            #set_trace()
            i.pickle(outputdir, verbose)

    def sortByAlg(self):
        """Returns a dictionary sorted based on algId and comment."""

        sorted = {}
        for i in self:
            sorted.setdefault(i.algId + ', ' + i.comment,
                              IndexEntries()).append(i)
        return sorted

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

        sorted = {}
        for i in self:
            if i.funcId in range(1, 6):
                sorted.setdefault('separ', []).append(i)
            elif i.funcId in range(6, 10):
                sorted.setdefault('lcond', []).append(i)
            elif i.funcId in range(10, 15):
                sorted.setdefault('hcond', []).append(i)
            elif i.funcId in range(15, 20):
                sorted.setdefault('multi', []).append(i)
            elif i.funcId in range(20, 25):
                sorted.setdefault('mult2', []).append(i)
            elif i.funcId in range(101, 107):
                sorted.setdefault('nzmod', []).append(i)
            elif i.funcId in range(107, 122):
                sorted.setdefault('nzsev', []).append(i)
            elif i.funcId in range(122, 131):
                sorted.setdefault('nzsmm', []).append(i)
            else:
                warnings.warn('Unknown function id.')

        return sorted

    def successProbability(self, targetFuncValue):
        """Returns overall probability of success given a target function value
        targetFuncValue -- float

        Exception:
        ValueError
        """

        if self:
            succ = 0
            nbRuns = 0
            for i in self:
                nbRuns += i.nbRuns()
                succ += i.nbSuccess()
            return float(succ)/nbRuns
        else:
            raise ValueError('The probability of success is not defined.')
