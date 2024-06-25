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

from __future__ import absolute_import, print_function

import os, sys
import numpy
import warnings

from . import genericsettings, testbedsettings, dataformatsettings

from pdb import set_trace
from six import string_types, advance_iterator

# CLASS DEFINITIONS
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

    # TODO: this class and the inheriting classes may have to be
    # redesigned for other kind of problems to work. They seem too have
    # quite a complicated inheritance structure for what they are
    # suppposed to do, that is, just read in a few info and data files.

    # idx: index of the column in the data array of the alignment value.
    # idxData: index of the column in the data array for the data of concern.

    def __init__(self, data, isHArray=False):
        """accepts a list of arrays or a `MultiReader` (i.e. a list of
        `SingleReader`) as input `data` type """
        self.isHArray = isHArray
        try:  # we act like data is a MultiReader
            #  this is meant to make a reset-copy of MultiReader
            for i, reader in enumerate(data):
                self.append(self.SingleReader(reader.data, data.isHArray))
        except AttributeError:  # we assume that data is a simple list
            for ar in data:
                if len(ar) > 0:  # ie. if the data array is not empty.
                    self.append(self.SingleReader(ar, isHArray))

    def currentLine(self):
        """Aggregates currentLines information."""
        return numpy.array(list(i.currentLine[self.idxData] for i in self))

    def currentValues(self):
        """Gets the list of the current alignment values."""
        return list(i.currentLine[self.idx] for i in self)

    def nextValues(self):
        """Gets the list of the next alignment values."""
        return list(i.nextLine[self.idx] for i in self if not i.isFinished)

        # def isFinished(self):
        """When all the data is read."""
        # pass

        # def getInitialValue(self):
        """Returns the initial alignment value."""
        # pass

        # def newCurrentValue(self):
        """Returns the next alignment value."""
        # pass

        # def align(self, currentValue):
        """Process all the elements of self to make them aligned."""
        # pass

    class SingleReader:
        """Single data array reader class."""

        def __init__(self, data, isHArray=False):
            if len(data) == 0:
                raise ValueError('Empty data array.')
            self.data = numpy.asarray(data)
            self.it = self.data.__iter__()
            self.isNearlyFinished = False
            self.isFinished = False
            self.currentLine = None
            self.nextLine = advance_iterator(self.it)
            if isHArray:
                self.idxEvals = range(1, numpy.shape(data)[1])
            else:
                # TODO: this looks like a very bad use-case for global variables (as most are)
                # similar are scattered over the classes below
                self.idxEvals = dataformatsettings.current_data_format.evaluation_idx

        def next(self):
            """Returns the next (last if undefined) line of the array data."""

            if not self.isFinished:
                if not self.isNearlyFinished:  # the next line is still defined
                    self.currentLine = self.nextLine.copy()
                    # Update nextLine
                    try:
                        self.nextLine = advance_iterator(self.it)
                    except StopIteration:
                        self.isNearlyFinished = True
                else:
                    self.isFinished = True
                    self.currentLine[self.idxEvals] = numpy.nan
                    # TODO: the line above was not valid for the MultiArrayReader

            return self.currentLine


class VMultiReader(MultiReader):
    """List of data arrays to be aligned vertically.

    Aligned vertically means, all number of function evaluations are the
    closest from below or equal to the alignment number of function
    evaluations.

    """

    def __init__(self, data):
        super(VMultiReader, self).__init__(data)
        # the alignment value is the number of function evaluations.
        self.idx = dataformatsettings.current_data_format.evaluation_idx
        # the data of concern are the function values.
        self.idxData = dataformatsettings.current_data_format.function_value_idx

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
                if i.nextLine[self.idx] > currentValue and not is_close(i.nextLine[self.idx], currentValue):
                    break
                i.next()
        return numpy.insert(self.currentLine(), 0, currentValue)


class HMultiReader(MultiReader):
    """List of data arrays to be aligned horizontally.

    Aligned horizontally means all the function values are lesser than
    (or equal to) the current alignment function value.

    """

    def __init__(self, data):
        super(HMultiReader, self).__init__(data)
        # the data of concern are the number of function evals.
        self.idxData = dataformatsettings.current_data_format.evaluation_idx
        # the alignment value is the function value.
        self.idx = dataformatsettings.current_data_format.function_value_idx
        self.nbPtsF = testbedsettings.current_testbed.number_of_points
        self.idxCurrentF = numpy.inf  # Minimization
        # idxCurrentF is a float for the extreme case where it is infinite.
        # else it is an integer and then is the 'i' in 10**(i/nbPtsF)
        self.isNegative = False
        self.idxCurrentFOld = 0.

    def calculateCurrentValue(self):
        factor = -1. if self.isNegative else 1.
        return factor * numpy.power(10, self.idxCurrentF / self.nbPtsF)

    def isFinished(self):
        """Is finished when we found the last alignment value reached."""

        currentValue = self.calculateCurrentValue()

        # It can be more than one line for the previous alignment value.
        # We iterate until we find a better value or to the end of the lines.
        for i in self:
            while i.nextLine[self.idx] > currentValue and not i.isFinished:
                i.next()

        return not any(i.nextLine[self.idx] <= currentValue for i in self)

    def getInitialValue(self):
        for i in self:
            i.next()
        fvalues = self.currentValues()
        self.idxCurrentF = numpy.ceil(numpy.log10(max(fvalues) if max(fvalues) > 0 else 1e-19) * self.nbPtsF)
        # Returns the smallest 10^i/nbPtsF value larger than max(Fvalues)
        return self.calculateCurrentValue()

    def newCurrentValue(self):
        if self.idxCurrentF == -numpy.inf:
            self.idxCurrentF = self.idxCurrentFOld
            self.isNegative = True
        elif self.isNegative:
            self.idxCurrentF += 1
        else:
            self.idxCurrentF -= 1
        return self.calculateCurrentValue()

    def align(self, currentValue):
        fvalues = []
        for i in self:
            while not i.isFinished:
                if i.currentLine[self.idx] <= currentValue or is_close(i.currentLine[self.idx], currentValue):
                    break
                i.next()
            if i.currentLine[self.idx] <= currentValue or is_close(i.currentLine[self.idx], currentValue):
                fvalues.append(i.currentLine[self.idx])

        # This should not happen
        if not fvalues:
            raise ValueError('Value %g is not reached.')

        maxf = max(fvalues)
        if maxf <= 0.:
            if currentValue > 0.:
                self.idxCurrentFOld = self.idxCurrentF
                self.idxCurrentF = -numpy.inf
                currentValue = 0.
            else:
                self.idxCurrentF = max((self.idxCurrentF,
                        numpy.floor(numpy.log10(-maxf + 1e-12) * self.nbPtsF)))
                currentValue = self.calculateCurrentValue()
        else:  # maxf > 0
            # it is not quite clear what this correction is for
            # to make sure that we do not get stuck at the same level?
            if maxf >= 2e-12:
                maxf -= 1e-12  # this gave warnings for too small maxf-values
            else:
                maxf /= 2
            self.idxCurrentF = min(self.idxCurrentF,
                                   numpy.ceil(numpy.log10(maxf) * self.nbPtsF))
            # Above line may return: Warning: divide by zero encountered in
            # log10 in the case of negative fvalues.
            # In the case of negative values for fvalues, self.idxCurrentF
            # should be -numpy.inf at the condition that
            # numpy.power(10, -inf) == 0 is true

            # The update of idxCurrentF is done so all the intermediate
            # function value trigger reached are not written, only the smallest
            currentValue = self.calculateCurrentValue()

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

    idx = 0  # We expect the alignment value to be the 1st column.

    def __init__(self, data, isHArray=False):
        # super(ArrayMultiReader, self).__init__(data, True)
        MultiReader.__init__(self, data, isHArray)
        # for i in self:
        # i.nbRuns = (numpy.shape(i.data)[1] - 1)

    def currentLine(self):
        """Aggregates currentLines information."""
        res = []
        res.extend(list(i.currentLine[1:] for i in self))
        return numpy.hstack(res)


class VArrayMultiReader(ArrayMultiReader, VMultiReader):
    """Wrapper class of *aligned* data arrays to be aligned vertically."""

    def __init__(self, data):
        ArrayMultiReader.__init__(self, data)
        # TODO: Should this use super? It probably shouldn't have multiple inheritance in the first place.


class VArrayMultiReaderNew(ArrayMultiReader, VMultiReader):
    """Wrapper class of *aligned* data arrays to be aligned vertically."""

    def __init__(self, data):
        ArrayMultiReader.__init__(self, data)
        # TODO: Should this use super?


class HArrayMultiReader(ArrayMultiReader, HMultiReader):
    """Wrapper class of *aligned* data arrays to be aligned horizontally."""

    def __init__(self, data):
        ArrayMultiReader.__init__(self, data, isHArray=True)
        # TODO: Should this use super?
        self.nbPtsF = testbedsettings.current_testbed.number_of_points
        self.idxCurrentF = numpy.inf  # Minimization
        self.isNegative = False
        self.idxCurrentFOld = 0.


# FUNCTION DEFINITIONS
def align_data(data, idx_evals, idx_funvals, rewind_reader=False):
    """Aligns the data from a list of data arrays.

    This method returns an array for which the alignment value is the
    first column and the aligned values are in subsequent columns.

    """

    if rewind_reader:
        if isinstance(data, HMultiReader):
            data = HMultiReader(data)
        elif isinstance(data, VMultiReader):
            data = VMultiReader(data)
        else:
            raise TypeError("reset class %s not implemented"
                            % type(data))

    keep_idxData, keep_idx = data.idxData, data.idx
    # This is terrible but needed, because several columns idxData need
    # to be read from the very same `data` in the constrained case and a
    # single call to `align_data` only allows to read in one column
    if isinstance(data, HMultiReader):
        data.idxData = idx_evals
        data.idx = idx_funvals
    elif isinstance(data, VMultiReader):  # so far not necessary
        data.idxData = idx_funvals
        data.idx = idx_evals
    else:
        raise TypeError("reset class %s not implemented"
                        % type(data))
    if set((data.idxData, data.idx)) != set((idx_evals, idx_funvals)):
        raise ValueError("indices are inconsistent " +
                         str((idx_evals, idx_funvals, data.idx, data.idxData)))

    res = []
    current_value = data.getInitialValue()
    # set_trace()
    if data.isFinished():
        res.append(data.align(current_value))

    while not data.isFinished() and current_value is not None:
        res.append(data.align(current_value))
        current_value = data.newCurrentValue()

    res = (numpy.vstack(res), numpy.asarray([i.nextLine[idx_evals] for i in data]),
            numpy.asarray([i.nextLine[idx_funvals] for i in data]))
    # Hack: at this point nextLine contains all information on the last line
    # of the data.

    data.idxData, data.idx = keep_idxData, keep_idx

    return res


def alignArrayData(data):
    """Aligns the data from a list of aligned arrays.

    This method returns an array for which the alignment value is the first
    column and the aligned values are in subsequent columns.

    """

    # TODO: is template dependent.

    res = []
    currentValue = data.getInitialValue()
    # set_trace()
    if data.isFinished():
        res.append(data.align(currentValue))

    while not data.isFinished():
        res.append(data.align(currentValue))
        currentValue = data.newCurrentValue()

    return numpy.vstack(res)
    # Hack: at this point nextLine contains all information on the last line
    # of the data.


def openfile(filePath, **kwargs):
    """`kwargs` are passed to `open`"""
    if not os.path.isfile(filePath):
        if ('win32' in sys.platform) and len(filePath) > 259:
            raise IOError(2, 'The path is too long for the file "%s".' % filePath)
        else:
            raise IOError(2, 'The file "%s" does not exist.' % filePath)
    try:
        return open(filePath, 'r', **kwargs)
    except TypeError:  # prevent TypeError: 'errors' is an invalid keyword argument for this function in Jenkins test
        if 'errors' in kwargs:
            kwargs.pop('errors')
        return open(filePath, 'r', **kwargs)


def split(dataFiles, idx_to_load=None, dim=None):
    """Split a list of data files into arrays corresponding to data sets.
       The Boolean list idx_to_load is thereby indicating whether a
       given part of the split is to be considered or not if None, all
       instances are considered.
    """

    data_sets = []
    algorithms = []
    success_ratio = []
    reference_values = {}
    for fil in dataFiles:
        with openfile(fil) as f:
            # This doesnt work with windows.
            # content = numpy.loadtxt(fil, comments='%')
            lines = f.readlines()

        content = []
        idx = 0  # instance index for checking in idx_to_load
        current_instance = 0
        current_reference_value = 0
        is_best_algorithm_data = False

        # Save values in array content. Check for nan and inf.
        for line in lines:
            if line.startswith('%'):
                if content:
                    if (idx_to_load is None) or (idx_to_load and len(idx_to_load) > idx and idx_to_load[idx]):
                        data_sets.append(numpy.vstack(content))
                    elif genericsettings.verbose:
                            print('skipped instance...')
                    # Use only the reference values from instances 1 to 5.
                    if current_instance in (1, 2, 3, 4, 5):
                        reference_values[current_instance] = current_reference_value

                    content = []
                    current_instance = 0
                    current_reference_value = 0
                    is_best_algorithm_data = False
                    idx += 1

                # Get the current instance and reference value.
                parts = line.strip('\n').strip(r'%').split(', ')
                for elem in parts:
                    if '=' in elem:
                        key, value = elem.split('=', 1)
                        if key.strip() == 'instance':
                            current_instance = int(value.strip())
                        elif key.strip() == 'reference value':
                            current_reference_value = float(value.strip())
                        elif key.strip() == 'algorithm type':
                            is_best_algorithm_data = 'best' == value.strip()

                continue

            # else remove end-of-line sign
            # and split into single strings
            data = line.strip('\n').split()

            # remove additional data for best algorithm
            if is_best_algorithm_data:
                index = len(data) - 3
                if index <= 0:
                    warnings.warn('Invalid best algorithm data!')
                else:
                    algorithms.append(data[index])
                    successful_runs = int(data[index + 1])
                    all_runs = int(data[index + 2])
                    success_ratio.append([successful_runs, all_runs])
                    data = data[:-3]  # remove the three processed items from data

            if dim and len(data) != dim + 5:
                warnings.warn('Incomplete line %s in  ' % line +
                              'data file %s: ' % fil)
                continue
            for index in range(len(data)):
                if data[index] in ('Inf', 'inf'):
                    data[index] = numpy.inf
                elif data[index] in ('-Inf', '-inf'):
                    data[index] = -numpy.inf
                elif data[index] in ('NaN', 'nan'):
                    data[index] = numpy.nan
                else:
                    try:
                        data[index] = float(data[index])
                    except ValueError:
                        warnings.warn('%s is not a valid number!' % data[index])
                        data[index] = numpy.nan

            if data:
                content.append(numpy.array(data))
            # Check that it always have the same length?

        if content:
            if (idx_to_load is None) or (idx_to_load and len(idx_to_load) > idx and idx_to_load[idx]):
                data_sets.append(numpy.vstack(content))
            elif genericsettings.verbose:
                    print('skipped instance...')

            # Use only the reference values from instances 1 to 5.
            if current_instance in (1, 2, 3, 4, 5):
                reference_values[current_instance] = current_reference_value

    if len(algorithms) < len(data_sets):
        algorithms = []

    return data_sets, algorithms, reference_values, success_ratio


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
