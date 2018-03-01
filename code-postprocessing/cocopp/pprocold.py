#! /usr/bin/env python
# coding: utf-8

"""Helper routines for read index files."""

from __future__ import absolute_import

import numpy
import warnings

from pdb import set_trace

__version__ = "$Revision$"
# $URL$
# $Date$

#GLOBAL VARIABLES
idxEvals = 0
idxF = 2
nbPtsF = 5;

#CLASS DEFINITIONS
#class DataSet(numpy.ndarray):
    #"""Aggregation of data from multiple runs. Implements methods to access
    #data in the array.
    #"""

    #def __init__(self, *args, **kwargs):
        #super(DataSet, self).__init__(*args, **kwargs)

    #def getFuncEvals(self, funcValues):
        #"""Provided with function values, return the number of function
        #evaluations to reach.
        #"""

    #def getFuncValues(self, FuncEvals):
        #"""Provided with function values, return the number of function
        #evaluations to reach.
        #"""


class MultiReader(list):
    """Wrapper class of data arrays to be aligned.

    This class is part abstract: some methods have to be defined by inheriting
    classes depending on wanted alignment: isFinished, getInitialValue,
    newCurrentValue, align, idx.

    """

    def __init__(self, data):
        for i in data:
            if len(i) > 0:
                self.append(self.SingleReader(i))

    def currentLine(self):
        """Aggregates currentLines information."""
        res = []
        res.extend(list(i.currentLine[idxEvals] for i in self))
        res.extend(list(i.currentLine[idxF] for i in self))
        return numpy.array(res)

    def currentValues(self):
        return list(i.currentLine[self.idx] for i in self)

    def nextValues(self):
        return list(i.nextLine[self.idx] for i in self if not i.isFinished)

    #def isFinished(self):
        #pass

    #def getInitialValue(self):
        #pass

    #def newCurrentValue(self):
        #pass

    #def align(self, currentValue):
        #pass

    class SingleReader:
        """Single data array reader class."""
        def __init__(self, data):
            if len(data) == 0:
                raise ValueError('Empty data array.')
            self.data = numpy.array(data)
            self.it = self.data.__iter__()
            self.isFinished = False
            self.currentLine = None
            self.nextLine = self.it.next()

        def next(self):
            """Returns the next (last if undefined) line of the array data."""
            self.currentLine = self.nextLine
            try:
                self.nextLine = self.it.next()
            except StopIteration:
                self.isFinished = True
            return self.currentLine


class VMultiReader(MultiReader):
    """Wrapper class of data arrays to be aligned vertically.
    Aligned vertically means, all number of function evaluations are the
    closest from below to the alignment number of function evaluations.
    """

    idx = idxEvals

    def __init__(self, data):
        MultiReader.__init__(self, data)

    def isFinished(self):
        return all(i.isFinished for i in self)

    #def currentLine(self):
        #"""Aggregates currentLines information."""
        #res = []
        #res.extend(list(i.currentLine[idxEvals] for i in self))
        #res.extend(list(i.currentLine[idxF] for i in self))
        #return numpy.array(res)

    def getInitialValue(self):
        for i in self:
            advance_iterator(i)
        res = self.currentValues()
        return min(res)

    def newCurrentValue(self):
        res = self.nextValues()
        if res:
            return min(self.nextValues())
        else:
            return None

    def align(self, currentValue):
        for i in self:
            while not i.isFinished:
                if i.nextLine[self.idx] > currentValue:
                    break
                advance_iterator(i)
        return numpy.insert(self.currentLine(), 0, currentValue)


class HMultiReader(MultiReader):
    """Wrapper class of data arrays to be aligned vertically."""
    idx = idxF

    def __init__(self, data):
        MultiReader.__init__(self, data)
        self.idxCurrentF = numpy.inf #Minimization
        #idxCurrentF is a float for the extreme case where it is infinite.

    def isFinished(self):
        currentValue = numpy.power(10, self.idxCurrentF / nbPtsF)
        if currentValue == 0:
            return True

        return not any(i.nextLine[self.idx] <= currentValue for i in self)

    def getInitialValue(self):
        for i in self:
            advance_iterator(i)
        fvalues = self.currentValues()
        self.idxCurrentF = numpy.ceil(numpy.log10(max(fvalues)) * nbPtsF)
        # Returns the smallest 10^i/nbPtsF value larger than max(Fvalues)
        return numpy.power(10, self.idxCurrentF / nbPtsF)

    def newCurrentValue(self):
        self.idxCurrentF -= 1
        return numpy.power(10, self.idxCurrentF / nbPtsF)

    def align(self, currentValue):
        fvalues = []
        for i in self:
            while not i.isFinished:
                if i.currentLine[self.idx] <= currentValue:
                    break
                advance_iterator(i)
            if i.currentLine[self.idx] <= currentValue:
                fvalues.append(i.currentLine[self.idx])

        #This should not happen
        if not fvalues:
            raise ValueError('Value %g is not reached.')

        self.idxCurrentF = min(self.idxCurrentF,
                               numpy.ceil(numpy.log10(max(fvalues)) * nbPtsF))
        #The update of idxCurrentF is done so all the intermediate function
        #value trigger reached are not written, only the smallest.
        currentValue = numpy.power(10, self.idxCurrentF / nbPtsF)
        return numpy.insert(self.currentLine(), 0, currentValue)


class ArrayMultiReader(MultiReader):
    """Wrapper class of ALIGNED data arrays to be aligned together."""

    idx = 0

    def __init__(self, data):
        MultiReader.__init__(self, data)
        for i in self:
            i.nbRuns = (numpy.shape(i.data)[1] - 1)/2

    def currentLine(self):
        """Aggregates currentLines information."""
        res = []
        res.extend(list(i.currentLine[1:i.nbRuns+1] for i in self))
        res.extend(list(i.currentLine[1+i.nbRuns:] for i in self))
        return numpy.hstack(res)

class VArrayMultiReader(ArrayMultiReader, VMultiReader):
    """Wrapper class of ALIGNED data arrays to be aligned vertically."""

    def __init__(self, data):
        ArrayMultiReader.__init__(self, data)


class HArrayMultiReader(ArrayMultiReader, HMultiReader):
    """Wrapper class of ALIGNED data arrays to be aligned vertically."""

    def __init__(self, data):
        ArrayMultiReader.__init__(self, data)
        self.idxCurrentF = numpy.inf #Minimization


#FUNCTION DEFINITIONS

def alignData(data):
    """Returns an array of aligned data from a list of data arrays.
    Data is the concatenation of rows, the zero-th column being the alignment
    value (or index), the subsequent ones the aligned data. The size of the
    resulting array will be of size 2*N + 1 where N is the number of data sets
    to be aligned. Rows are sorted: the first data set will be in column
    1 and N+1, the second in 2 and N+2. If still valid or if none are available
    at some point, data obtained in previous rows are repeated in the current
    one.

    """

    res = []
    currentValue = data.getInitialValue()
    #set_trace()
    if data.isFinished():
        res.append(data.align(currentValue))

    while not data.isFinished():
        res.append(data.align(currentValue))
        currentValue = data.newCurrentValue()

    return numpy.vstack(res)


def split(dataFiles, dim=None):
    """Split a list of data files into arrays corresponding to data sets."""

    dataSets = []
    for fil in dataFiles:
        try:
            # This doesnt work with windows.
            # content = numpy.loadtxt(fil, comments='%')

            file = open(fil,'r')               # read in the file
            lines = file.readlines()
        except IOError:
            print('Could not find %s.' % fil)
            continue

        content = []

        # Save values in array content. Check for nan and inf.
        for line in lines:
            # skip if comment
            if line.startswith('%'):
                if content:
                    dataSets.append(numpy.vstack(content))
                    content = []
                continue

            # else remove end-of-line sign
            # and split into single strings
            data = line.strip('\n').split()
            if dim and len(data) != dim + 5:
                warnings.warn('Incomplete line %s in  ' % (line) +
                              'data file %s: ' % (dataFiles))
                continue
            for id in range(len(data)):
                if data[id] in ('Inf', 'inf'):
                    data[id] = numpy.inf
                elif data[id] in ('-Inf', '-inf'):
                    data[id] = -numpy.inf
                elif data[id] in ('NaN', 'nan'):
                    data[id] = numpy.nan
                else:
                    data[id] = float(data[id])
            #set_trace()
            content.append(numpy.array(data))
            #Check that it always have the same length?
        if content:
            dataSets.append(numpy.vstack(content))

    return dataSets
