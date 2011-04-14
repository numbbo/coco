#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Helper routines to read in data files.

The terms horizontal and vertical below refer to the horizontal
(fixed-target) and vertical (fixed-budget) views. When considering
convergence graphs of function values over times, we can view it as:

  * costs for different fixed targets represented by horizontal cuts.
  * function values for different fixed budgets represented by vertical
    cuts.

COCO collects experimental data with respect to these two complementary
views. This module provides data structures and methods for dealing with
the experimental data.

"""

from __future__ import absolute_import

import numpy
import warnings

from pdb import set_trace


#GLOBAL VARIABLES
idxEvals = 0 # index of the column where to find the evaluations
idxF = 2 # index of the column where to find the function values
nbPtsF = 5 # nb of target function values for each decade.

#CLASS DEFINITIONS
class MultiReader(list):
    """List of data arrays to be aligned.

    The main purpose of this class is to be used as a single container
    of the data arrays to be aligned by :py:meth:`alignData()` in the
    parent module.
    A data array is defined as an array where rows correspond to
    recordings at different moments of an experiment. Elements of these
    rows correspond to different measures.
    These data arrays can be aligned along the time or the function
    values for instance.

    This class is part abstract. Some methods have to be defined by
    inheriting classes depending on wanted alignment:

      * :py:meth:`isFinished()`, True when all the data is read.
      * :py:meth:`getInitialValue()`, returns the initial alignment
        value.
      * :py:meth:`newCurrentValue()`, returns the next alignment value.
      * :py:meth:`align()`, process all the elements of self to make
        them aligned.

    Some attributes have to be defined as well :py:attr:`idx`,
    the index of the column with alignment values in the data array,
    :py:attr:`idxData`, the index of the column with the actual data.

    """

    # TODO: this class and all inheriting class may have to be redesigned for
    # other kind of problems to work.

    # idx: index of the column in the data array of the alignment value.
    # idxData: index of the column in the data array for the data of concern.

    def __init__(self, data, isHArray=False):
        for i in data:
            if len(i) > 0: # ie. if the data array is not empty.
                self.append(self.SingleReader(i, isHArray))

    def currentLine(self):
        """Aggregates currentLines information."""
        return numpy.array(list(i.currentLine[self.idxData] for i in self))

    def currentValues(self):
        """Gets the list of the current alignment values."""
        return list(i.currentLine[self.idx] for i in self)

    def nextValues(self):
        """Gets the list of the next alignment values."""
        return list(i.nextLine[self.idx] for i in self if not i.isFinished)

    #def isFinished(self):
        """When all the data is read."""
        #pass

    #def getInitialValue(self):
        """Returns the initial alignment value."""
        #pass

    #def newCurrentValue(self):
        """Returns the next alignment value."""
        #pass

    #def align(self, currentValue):
        """Process all the elements of self to make them aligned."""
        #pass

    class SingleReader:
        """Single data array reader class."""

        def __init__(self, data, isHArray=False):
            if len(data) == 0:
                raise ValueError, 'Empty data array.'
            self.data = numpy.array(data)
            self.it = self.data.__iter__()
            self.isNearlyFinished = False
            self.isFinished = False
            self.currentLine = None
            self.nextLine = self.it.next()
            if isHArray:
                self.idxEvals = range(1, numpy.shape(data)[1])
            else:
                self.idxEvals = idxEvals

        def next(self):
            """Returns the next (last if undefined) line of the array data."""

            if not self.isFinished:
                if not self.isNearlyFinished: # the next line is still defined
                    self.currentLine = self.nextLine.copy()
                    # Update nextLine
                    try:
                        self.nextLine = self.it.next()
                    except StopIteration:
                        self.isNearlyFinished = True
                else:
                    self.isFinished = True
                    self.currentLine[self.idxEvals] = numpy.nan
                    #TODO: the line above was not valid for the MultiArrayReader

            return self.currentLine


class VMultiReader(MultiReader):
    """List of data arrays to be aligned vertically.

    Aligned vertically means, all number of function evaluations are the
    closest from below or equal to the alignment number of function
    evaluations.

    """

    idx = idxEvals # the alignment value is the number of function evaluations.
    idxData = idxF # the data of concern are the function values.

    def __init__(self, data):
        super(VMultiReader, self).__init__(data)

    def isFinished(self):
        return all(i.isFinished for i in self)

    def getInitialValue(self):
        for i in self:
            i.next()
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
                i.next()
        return numpy.insert(self.currentLine(), 0, currentValue)


class HMultiReader(MultiReader):
    """List of data arrays to be aligned horizontally.

    Aligned horizontally means all the function values are lesser than
    (or equal to) the current alignment function value.

    """

    idx = idxF # the alignment value is the function value.
    idxData = idxEvals # the data of concern are the number of function evals.

    def __init__(self, data):
        super(HMultiReader, self).__init__(data)
        self.idxCurrentF = numpy.inf # Minimization
        # idxCurrentF is a float for the extreme case where it is infinite.
        # else it is an integer and then is the 'i' in 10**(i/nbPtsF)

    def isFinished(self):
        """Is finished when we found the last alignment value reached."""

        currentValue = numpy.power(10, self.idxCurrentF / nbPtsF)
        if currentValue == 0:
            return True

        return not any(i.nextLine[self.idx] <= currentValue for i in self)

    def getInitialValue(self):
        for i in self:
            i.next()
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
                i.next()
            if i.currentLine[self.idx] <= currentValue:
                fvalues.append(i.currentLine[self.idx])

        #This should not happen
        if not fvalues:
            raise ValueError, 'Value %g is not reached.'

        if max(fvalues) <= 0.:
            self.idxCurrentF = -numpy.inf
            currentValue = 0.
        else:
            self.idxCurrentF = min(self.idxCurrentF,
                               numpy.ceil(numpy.log10(max(fvalues)) * nbPtsF))
            # Above line may return: Warning: divide by zero encountered in
            # log10 in the case of negative fvalues.
            # In the case of negative values for fvalues, self.idxCurrentF
            # should be -numpy.inf at the condition that
            # numpy.power(10, -inf) == 0 is true

            # The update of idxCurrentF is done so all the intermediate
            # function value trigger reached are not written, only the smallest
            currentValue = numpy.power(10, self.idxCurrentF / nbPtsF)

        return numpy.insert(self.currentLine(), 0, currentValue)


class ArrayMultiReader(MultiReader):
    """Class of *aligned* data arrays to be aligned together.

    This class is used for dealing with the output of
    :py:class:`MultiReader`:
    
    * From *raw* data arrays, :py:class:`MultiReader` generates aligned
      data arrays (first column is the alignment value, subsequent
      columns are aligned data).
    * This class also generates aligned data arrays but from other
      aligned data arrays.

    """

    idx = 0 # We expect the alignment value to be the 1st column.

    def __init__(self, data, isHArray=False):
        #super(ArrayMultiReader, self).__init__(data, True)
        MultiReader.__init__(self, data, isHArray)
        #for i in self:
            #i.nbRuns = (numpy.shape(i.data)[1] - 1)

    def currentLine(self):
        """Aggregates currentLines information."""
        res = []
        res.extend(list(i.currentLine[1:] for i in self))
        return numpy.hstack(res)

class VArrayMultiReader(ArrayMultiReader, VMultiReader):
    """Wrapper class of *aligned* data arrays to be aligned vertically."""

    def __init__(self, data):
        ArrayMultiReader.__init__(self, data)
        #TODO: Should this use super?

class HArrayMultiReader(ArrayMultiReader, HMultiReader):
    """Wrapper class of *aligned* data arrays to be aligned horizontally."""

    def __init__(self, data):
        ArrayMultiReader.__init__(self, data, isHArray=True)
        #TODO: Should this use super?
        self.idxCurrentF = numpy.inf #Minimization


#FUNCTION DEFINITIONS

def alignData(data):
    """Aligns the data from a list of data arrays.

    This method returns an array for which the alignment value is the
    first column and the aligned values are in subsequent columns.

    """

    #TODO: is template dependent.

    res = []
    currentValue = data.getInitialValue()
    #set_trace()
    if data.isFinished():
        res.append(data.align(currentValue))

    while not data.isFinished():
        res.append(data.align(currentValue))
        currentValue = data.newCurrentValue()

    return (numpy.vstack(res), numpy.array(list(i.nextLine[idxEvals] for i in data)),
            numpy.array(list(i.nextLine[idxF] for i in data)))
    # Hack: at this point nextLine contains all information on the last line
    # of the data.


def alignArrayData(data):
    """Aligns the data from a list of aligned arrays.

    This method returns an array for which the alignment value is the first
    column and the aligned values are in subsequent columns.

    """

    #TODO: is template dependent.

    res = []
    currentValue = data.getInitialValue()
    #set_trace()
    if data.isFinished():
        res.append(data.align(currentValue))

    while not data.isFinished():
        res.append(data.align(currentValue))
        currentValue = data.newCurrentValue()

    return numpy.vstack(res)
    # Hack: at this point nextLine contains all information on the last line
    # of the data.


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
            print 'Could not find %s.' % fil
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
            for id in xrange(len(data)):
                if data[id] in ('Inf', 'inf'):
                    data[id] = numpy.inf
                elif data[id] in ('-Inf', '-inf'):
                    data[id] = -numpy.inf
                elif data[id] in ('NaN', 'nan'):
                    data[id] = numpy.nan
                else:
                    data[id] = float(data[id])

            content.append(numpy.array(data))
            #Check that it always have the same length?
        if content:
            dataSets.append(numpy.vstack(content))

    return dataSets
