#! /usr/bin/env python
# coding: utf-8

"""Helper routines for read index files.
   Defines class IndexEntry, unit element in the post-processing and the list
of instances of IndexEntry: DataSetList.

"""

from __future__ import absolute_import

import os
import re
import pickle
import warnings
from pdb import set_trace
import numpy
from bbob_pproc import findindexfiles, readalign, bootstrap
from bbob_pproc.readalign import split, alignData, HMultiReader, VMultiReader
from bbob_pproc.readalign import HArrayMultiReader, VArrayMultiReader

#GLOBAL VARIABLES
idxEvals = 0
idxF = 2
nbPtsF = 5;
indexmainsep = ', '

# CLASS DEFINITIONS
class DataSet:
    """Unit element for the post-processing with given funcId, algId and
    dimension.
    Class attributes:
        funcId -- function Id (integer)
        dim -- dimension (integer)
        dataFiles -- associated data files (list)
        comment -- comment for the setting (string)
        targetFuncValue -- target function value (float)
        algId -- algorithm name (string)
        evals -- collected data aligned by function values (array)
        funvals -- collected data aligned by function evaluations (array)
        nbRuns -- number of runs (integer)
        maxevals -- maximum number of function evaluations (array)
        finalfunvals -- final function values (array)

    evals and funvals are arrays of data collected from N data sets. Both have
    the same format: zero-th column is the value on which the data of a row is
    aligned, the N subsequent columns are either the numbers of function
    evaluations for evals or function values for funvals.
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
        self.itrials = []
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
                self.itrials.append(int(elem[0]))
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

        ext = {'.dat':(HMultiReader, 'evals'), '.tdat':(VMultiReader, 'funvals')}
        for extension, info in ext.iteritems():
            #set_trace()
            dataFiles = list('.'.join(i.split('.')[:-1]) + extension
                             for i in self.dataFiles)
            data = info[0](split(dataFiles))
            if verbose:
                print ("Processing %s: %d/%d trials found."
                       % (dataFiles, len(data), len(self.itrials)))
            (adata, maxevals, finalfunvals) = alignData(data)
            setattr(self, info[1], adata)
            try:
                if all(maxevals > self.maxevals):
                    self.maxevals = maxevals
                    self.finalfunvals = finalfunvals
            except AttributeError:
                self.maxevals = maxevals
                self.finalfunvals = finalfunvals

        # Compute ERT
        self.computeERTfromEvals()
        #set_trace()

    def computeERTfromEvals(self):
        self.ert = []
        self.target = []
        for i in self.evals:
            data = i.copy()
            data = data[1:]
            succ = (numpy.isnan(data)==False)
            if any(numpy.isnan(data)):
                data[numpy.isnan(data)] = self.maxevals[numpy.isnan(data)]
            self.ert.append(bootstrap.sp(data, issuccessful=succ)[0])
            self.target.append(i[0])

        #set_trace()
        self.ert = numpy.array(self.ert)
        self.target = numpy.array(self.target)



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

    def mMaxEvals(self):
        return max(self.maxevals)

    def nbRuns(self):
        return numpy.shape(self.funvals)[1]-1

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
                                  'attribute.')
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

    def createDictInstance(self):
        """Returns a dictionary of the instances: the key is the instance id,
        the value is a list of index.
        """
        dictinstance = {}
        for i in range(len(self.itrials)):
            dictinstance.setdefault(self.itrials[i], []).append(i)

        return dictinstance


    def splitByTrials(self, whichdata=None):
        """Splits the post-processed data arrays by trials.
        Returns a two-element list of dictionaries of arrays, the key of the
        dictionary being the instance id, the value being a smaller
        post-processed data array corresponding to the instance id.
        """

        dictinstance = self.createDictInstance()
        evals = {}
        funvals = {}

        for instanceid, idx in iteritems(dictinstance):
            evals[instanceid] = self.evals[:,
                                           numpy.ix_(list(i + 1 for i in idx))]
            funvals[instanceid] = self.funvals[:,
                                           numpy.ix_(list(i + 1 for i in idx))]

        if whichdata :
            if whichdata == 'evals':
                return evals
            elif whichdata == 'funvals':
                return funvals

        return (evals, funvals)

class DataSetList(list):
    """Set of instances of DataSet objects, implement some useful slicing
    functions.

    """

    #Do not inherit from set because DataSet instances are mutable.

    def __init__(self, args=[], verbose=True):
        """Instantiate self from a list of inputs.
        Keyword arguments:
        args -- list of strings being either info file names, folder containing
                info files or pickled data files.
        verbose -- controls verbosity.

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
                    self.append(DataSet(header, comment, data, verbose))
                except StopIteration:
                    break

        except IOError:
            print 'Could not open %s.' % indexFile

            # Close index file
            f.close()

    def append(self, o):
        """Redefines the append method to check for unicity."""
        #TODO: Watchout! extend does not call append.

        #set_trace()
        if not isinstance(o, DataSet):
            raise Exception()
        isFound = False
        for i in self:
            if i == o:
                isFound = True
                if set(i.dataFiles).symmetric_difference(set(o.dataFiles)):
                    #set_trace()
                    i.funvals = alignData(VArrayMultiReader([i.funvals, o.funvals]))[0]
                    #set_trace()
                    i.finalfunvals.extend(o.finalfunvals)
                    i.evals = alignData(HArrayMultiReader([i.evals, o.evals]))[0]
                    i.maxevals.extend(o.maxevals)
                    i.computeERTfromEvals()
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
        the corresponding slices of DataSetList as values.
        """

        d = {}
        for i in self:
            d.setdefault(i.algId + ', ' + i.comment, DataSetList()).append(i)
        return d

    def dictByDim(self):
        """Returns a dictionary with dimension as keys and the corresponding
        slices of DataSetList as values.
        """

        d = {}
        for i in self:
            d.setdefault(i.dim, DataSetList()).append(i)
        return d

    def dictByFunc(self):
        """Returns a dictionary with the function id as keys and the
        corresponding slices of DataSetList as values.
        """

        d = {}
        for i in self:
            d.setdefault(i.funcId, DataSetList()).append(i)
        return d

    def dictByNoise(self):
        """Returns a dictionary splitting noisy and non-noisy entries.
        """

        sorted = {}
        for i in self:
            if i.funcId in range(1, 25):
                sorted.setdefault('noiselessall', DataSetList()).append(i)
            elif i.funcId in range(101, 131):
                sorted.setdefault('nzall', DataSetList()).append(i)
            else:
                warnings.warn('Unknown function id.')

        return sorted

    def dictByFuncGroup(self):
        """Returns a dictionary with function group names as keys and the
        corresponding slices of DataSetList as values.
        """

        sorted = {}
        for i in self:
            if i.funcId in range(1, 6):
                sorted.setdefault('separ', DataSetList()).append(i)
            elif i.funcId in range(6, 10):
                sorted.setdefault('lcond', DataSetList()).append(i)
            elif i.funcId in range(10, 15):
                sorted.setdefault('hcond', DataSetList()).append(i)
            elif i.funcId in range(15, 20):
                sorted.setdefault('multi', DataSetList()).append(i)
            elif i.funcId in range(20, 25):
                sorted.setdefault('mult2', DataSetList()).append(i)
            elif i.funcId in range(101, 107):
                sorted.setdefault('nzmod', DataSetList()).append(i)
            elif i.funcId in range(107, 122):
                sorted.setdefault('nzsev', DataSetList()).append(i)
            elif i.funcId in range(122, 131):
                sorted.setdefault('nzsmm', DataSetList()).append(i)
            else:
                warnings.warn('Unknown function id.')

        return sorted

