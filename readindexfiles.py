#! /usr/bin/env python

# Script to read the data from the files within the list indexFiles
# and store it in the list indexEntries.

from __future__ import absolute_import

import sys
import os

from . import findindexfiles

from pdb import set_trace

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
        elif elemFirst == 'algId':
            entry.algId = elemSecond
        else:
            print 'Warning: Reading the header data from %s!' % f.name
            print '         Can not handle quantity %s!' % elem
    return entry


# CLASS DEFINITIONS
class IndexEntry:
    """Creates an instance with atributes funcId, dim, dataFiles,
       comment, maxEvals, targetFuncValue, and algId. A new instance
       contains the default values.
    """

    def __init__(self):
        self.funcId = 0    # function Id (integer)
        self.dim = 0        # dimension (integer)
        self.dataFiles = list()    # associated data files (list)
        self.comment = 'no comment'    # comment for the setting (string)
        self.maxEvals = 0    # max. number of function evaluations (integer or string)
        self.targetFuncValue = 0.0     # target function value (float)
        self.algId = 'no name'    # algorithm name (string)

    def __eq__(self,other):
        return (self.__class__ is other.__class__ and
                self.funcId == other.funcId and
                self.dim == other.dim and
                self.algId == other.algId)

    def __ne__(self,other):
        return not self.__eq__(other)


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

def main(indexFiles, verbose = True):
    # main code

    # Initialization
    indexEntries = list()

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
                parts = line.split(',')
                for elem2 in parts:
                    if elem2.endswith('.dat'):
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

    return indexEntries

if __name__ == "__main__":
    sys.exit(main())

# TODO: *somehow it works not on my home laptop
#         to execute 'python findIndexFiles.py'
#       (ImportError: No module named numpy)
