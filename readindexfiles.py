#! /usr/bin/env python

"""Defines class IndexEntry, unit element in the post-processing and the list
of instances of IndexEntry: IndexEntries.

"""

from __future__ import absolute_import

import os
import re
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
        hData -- collected data aligned by function values (array)
        vData -- collected data aligned by function evaluations (array)
        nbRuns -- number of runs (integer)
        mMaxEvals -- measured max. number of function evaluations (float)

    hData and vData are arrays of data collected from N data sets. Both have
    the same format: zero-th column is the value on which the data of a row is
    aligned, the N subsequent columns are the numbers of function evaluations
    of the aligned data, the N columns after those are the corresponding
    function values. Those 2 times N columns are sorted and go by pairs:
    column 1 and N+1 are related to the first trial, column 2 and N+2...
    """

    # Private attribute used for the parsing of info files.
    __attributes = {'funcId': ('funcId', int), 'DIM': ('dim',int),
                    'Precision': ('precision', float), 'Fopt': ('fopt', float),
                    'targetFuncValue': ('targetFuncValue', float),
                    'algId': ('algId', str)}

    def __init__(self, header, comment, data, verbose=True):
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
        self.instancenumbers = []
        self.evals = []
        self.finalFminusFtarget = []
        self.isFinalized = []
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
                self.instancenumbers.append(int(elem[0]))
                if len(elem) < 2:
                    #Caught a ill-finalized run.
                    self.isFinalized.append(False)
                    warnings.warn('Caught an ill-finalized run in %s'
                                  % self.dataFiles[-1])
                else:
                    self.isFinalized.append(True)
                    elem = elem[1].split('|')
                    self.evals.append(int(elem[0]))
                    self.finalFminusFtarget.append(float(elem[1]))
                #pass

        #set_trace()
        if verbose:
            print "%s" % self.__repr__()

        ext = {'.dat':(HMultiReader, 'hData'), '.tdat':(VMultiReader, 'vData')}
        for extension, info in ext.iteritems():
            #set_trace()
            dataFiles = list('.'.join(i.split('.')[:-1]) + extension
                             for i in self.dataFiles)
            data = info[0](split(dataFiles))
            if verbose:
                print ("Processing %s: %d/%d trials found."
                       % (dataFiles, len(data), len(self.instancenumbers)))
            setattr(self, info[1], alignData(data))
        #set_trace()

    def __eq__(self, other):
        """Compare indexEntry instances."""
        return (self.__class__ is other.__class__ and
                self.funcId == other.funcId and
                self.dim == other.dim and
                #self.precision == other.precision and
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
        # We loop backward to make sure that we did not split inside quotes.
        # It could happen when the key algId and the value is a string.
        p = re.compile('[^,=]+ = .*')
        headerList.reverse()
        it = iter(headerList)
        while True:
            try:
                elem = it.next()
                while not p.match(elem):
                    elem = it.next() + elem

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
                for k in j[self.nbRuns()+1:]:
                    if k <= targetFuncValue:
                        succ += 1
                break
        return succ

    def createDictInstance(self):
        """Returns a dictionary of the instances: the key is the instance id,
        the value is a list of an index corresponding to the instance in the
        hData and vData array.
        """
        dictinstance = {}
        for i in range(len(self.instancenumbers)):
            dictinstance.setdefault(self.instancenumbers[i], []).append(i)

        return dictinstance


    def splitByTrials(self, whichdata=None):
        """Splits the post-processed data arrays by trials.
        Returns a two-element list of dictionaries of arrays, the key of the
        dictionary being the instance id, the value being a smaller
        post-processed data array corresponding to the instance id.
        """

        dictinstance = self.createDictInstance()
        hData = {}
        vData = {}

        for instanceid, idx in iteritems(dictinstance):
            hData[instanceid] = self.hData[:,
                                           numpy.ix_(list(i + 1 for i in idx)),
                                           numpy.ix_(list(i + self.nbRuns()
                                                          for i in idx))]
            vData[instanceid] = self.vData[:,
                                           numpy.ix_(list(i + 1 for i in idx)),
                                           numpy.ix_(list(i + self.nbRuns()
                                                          for i in idx))]

        if whichdata :
            if whichdata == 'hData':
                return hData
            elif whichdata == 'vData':
                return vData

        return (hData, vData)

    def getFuncEvals(self, functionValues):
        """Returns a sequence of the number of function evaluations for
        each run to reach the given function value or the maximum number
        of function evaluations for the given if it did not reached it.
        The second ouput argument is a sequence of the success status for
        each run.
        Keyword arguments:
        functionValues -- 
        """

        try:
            iter(functionValues)
        except TypeError:
            functionValue = (functionValues, )

        sfunctionValues = sorted(functionValues)

        try:
            it = iter(self.hData)
            curline = it.next()
        except StopIteration:
            warnings.warn('Problem here!')

        res = {}
        isSuccessful = {}

        #Isn't this MultiReader.align???
        for fValue in sfunctionValues:
            while curline[0] > fValue:
                try:
                    curline = it.next()
                except StopIteration:
                    break

            tmpres = curline[1:self.nbRuns()+1]
            tmpIsSucc = (curline[self.nbRuns()+1:] <= fValue)

            for i in range(len(tmpIsSucc)):
                if not tmpIsSucc[i]:
                    tmpres[i] = self.vData[-1, 1+i]

            #Append a copy?
            res[fValue] = tmpres
            isSuccessful[fValue] = tmpIsSucc

        res = list(res[fValue] for fValue in functionValues)
        isSuccessful = list(isSuccessful[fValue]
                            for fValue in functionValues)

        return res, isSuccessful

    def getFuncValues(self, functionEvaluations):
        """Returns a sequence of the function values reached right before
        the given number of function evaluations or the last function value
        obtained.
        Keyword arguments:
        functionValues -- 
        isSuccessful -- is not the sequence indicating whether the functionEvaluations
        were actually reached! In this implementation it says whether the 
        """

        try:
            iter(functionEvaluations)
        except TypeError:
            functionEvaluations = (functionEvaluations, )

        sfunctionEvaluations = sorted(functionEvaluations)

        try:
            it = iter(self.vData)
            nextline = it.next()
            curline = nextline
        except StopIteration:
            warnings.warn('Problem here!')

        res = {}
        isSuccessful = {}

        set_trace()
        for fEvals in sfunctionEvaluations:
            while True:
                if nextline[0] > fEvals:
                    break
                try:
                    curline = nextline
                    nextline = it.next()
                except StopIteration:
                    break

            tmpres = curline[self.nbRuns()+1:]
            #set_trace()
            #tmpIsSucc = (curline[1:self.nbRuns()+1] <= fEvals)

            #for i in range(len(tmpIsSucc)):
                #if not tmpIsSucc[i]:
                    #tmpres[i] = self.vData[-1, self.nbRuns()+1+i]

            #Append a copy?
            res[fEvals] = tmpres
            #isSuccessful[fEvals] = tmpIsSucc

        res = list(res[fEvals] for fEvals in functionEvaluations)
        #isSuccessful = list(isSuccessful[fEvals]
                            #for fEvals in functionEvaluations)

        return res #, isSuccessful


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

        if isinstance(args, basestring):
            args = [args]

        for i in args:
            if i.endswith('.info'):
                self.processIndexFile(i, verbose)
            elif os.path.isdir(i):
                for j in findindexfiles.main(i, verbose):
                    self.processIndexFile(j, verbose)
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
            nbLine = 1
            while True:
                try:
                    header = f.next()
                    while not header.strip(): # remove blank lines
                        header = f.next()
                        nbLine += 1
                    comment = f.next()
                    if not comment.startswith('%'):
                        warnings.warn('Entry in file %s at line %d is faulty: '
                                      % (indexFile, nbLine) +
                                      'it will be skipped.')
                        nbLine += 2
                        continue
                    tmpline = f.next()
                    nbLine += 3
                    #TODO: check that something is not wrong with the 3 lines.
                    # Add the path to the index file to the file names
                    data = []
                    for i in tmpline.split(indexmainsep):
                        if i.endswith('dat'): #filenames
                            data.append(os.path.join(indexpath, i))
                        else: #other information
                            data.append(i)
                    data = indexmainsep.join(data)
                    self.append(IndexEntry(header, comment, data, verbose))
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
                    for j in dir(i):
                        if isinstance(getattr(i, j), list):
                            getattr(i, j).extend(getattr(o, j))
                break
        if not isFound:
            list.append(self, o)

    def pickle(self, outputdir, verbose=True):
        """Loop over self to pickle each elements."""

        for i in self:
            #set_trace()
            i.pickle(outputdir, verbose)

    def dictByAlg(self):
        """Returns a dictionary with algId and comment as keys and
        the corresponding slices of IndexEntries as values.
        """

        d = {}
        for i in self:
            d.setdefault(i.algId + ', ' + i.comment, IndexEntries()).append(i)
        return d

    def dictByDim(self):
        """Returns a dictionary with dimension as keys and the corresponding
        slices of IndexEntries as values.
        """

        d = {}
        for i in self:
            d.setdefault(i.dim, IndexEntries()).append(i)
        return d

    def dictByFunc(self):
        """Returns a dictionary with the function id as keys and the
        corresponding slices of IndexEntries as values.
        """

        d = {}
        for i in self:
            d.setdefault(i.funcId, IndexEntries()).append(i)
        return d

    def dictByNoise(self):
        """Returns a dictionary splitting noisy and non-noisy entries.
        """

        sorted = {}
        for i in self:
            if i.funcId in range(1, 25):
                sorted.setdefault('noiselessall', IndexEntries()).append(i)
            elif i.funcId in range(101, 131):
                sorted.setdefault('nzall', IndexEntries()).append(i)
            else:
                warnings.warn('Unknown function id.')

        return sorted

    def dictByFuncGroup(self):
        """Returns a dictionary with function group names as keys and the
        corresponding slices of IndexEntries as values.
        """

        sorted = {}
        for i in self:
            if i.funcId in range(1, 6):
                sorted.setdefault('separ', IndexEntries()).append(i)
            elif i.funcId in range(6, 10):
                sorted.setdefault('lcond', IndexEntries()).append(i)
            elif i.funcId in range(10, 15):
                sorted.setdefault('hcond', IndexEntries()).append(i)
            elif i.funcId in range(15, 20):
                sorted.setdefault('multi', IndexEntries()).append(i)
            elif i.funcId in range(20, 25):
                sorted.setdefault('mult2', IndexEntries()).append(i)
            elif i.funcId in range(101, 107):
                sorted.setdefault('nzmod', IndexEntries()).append(i)
            elif i.funcId in range(107, 122):
                sorted.setdefault('nzsev', IndexEntries()).append(i)
            elif i.funcId in range(122, 131):
                sorted.setdefault('nzsmm', IndexEntries()).append(i)
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
                succ += i.nbSuccess(targetFuncValue)
            return float(succ)/nbRuns
        else:
            raise ValueError('The probability of success is not defined. TODO: obvioulsy, but why?')
