#! /usr/bin/env python
# coding: utf-8

# Returns arrays of post-processed data from a list of data files.

from __future__ import absolute_import

import scipy
import scipy.io
from pdb import set_trace

from bbob_pproc import bootstrap

__version__ = "$Revision$"
# $URL$
# $Date$

__all__ = ['main', 'postprocess']

# Define global variables.
prctilesTab = (0., 0.1, 0.5, 0.9, 1.)
# These depends on the layout of the data files.
funcEvalsIndex = 0
fitValIndex = 2
maxEvalsFactor = 1e4
valuesOfInterest = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8) #has to be sorted
valuesForDataProf = (1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8) #has to be sorted


class Error(Exception):
    """ Base class for errors. """
    #TODO: what the?
    pass


class MissingValueError(Error):
    """ Error if a mandatory value is not found within a file.
        Returns a message with the missing value and the respective
        file."""

    def __init__(self,value, filename):
        self.value = value
        self.filename = filename

    def __str__(self):
        message = 'The value %s was not found in file %s!' % \
                  (self.value, self.filename)
        return repr(message)


# Remarks:
# Need for the optimum of the function!
# Let's store everything into memory.
# TODO: aligned with function values or function evaluations should be a choice.

def postprocess(dataFiles, fvalueToReach, maxEvals):
    """Post processes raw data files, returns post-processed data in arrays."""
    #TODO this function is long and ugly.
    #set_trace()
    dataSets = split(dataFiles)
    # Is it ok to store all the data in memory?

    measuredMaxEvals = []
    for i in dataSets:
        measuredMaxEvals.append(i.userDefMaxEvals)

    measuredMaxEvals = max(measuredMaxEvals)
    maxEvals = min((measuredMaxEvals, maxEvals))

    # Parallel construction of the post-processed arrays:
    currentFuncEvals = 1.
    currentFitValue = scipy.inf # Minimization

    isFinished = len(dataSets) * [False]
    res = len(dataSets) * [0]
    vals = len(dataSets) * [0]
    previousVals = len(dataSets) * [0]
    arrayFullTab = [] #aligned by function values.
    arrayTab = {} #aligned by function values.
    rlDistr = {} #aligned by function values.
    iValuesOfInterest = 0
    iValuesForDataProf = 0
    #arrayFig = [] #aligned by function evaluations.
    #currentLines = []
    #dataSetsToRead = dataSets
    #dataSetsToRemove = []
    #res = []

    while not all(isFinished) and iValuesOfInterest < len(valuesOfInterest):
        #CurrentLines is a list of arrays with lines from the dataSets
        #set_trace()
        # Align data
        for i in range(len(dataSets)):
            res[i] = dataSets[i].set[dataSets[i].currentPos, funcEvalsIndex]
            vals[i] = (dataSets[i].set[dataSets[i].currentPos, fitValIndex] -
                       fvalueToReach + 1.0e-8)
            if not isFinished[i]:
                while (dataSets[i].currentPos < len(dataSets[i].set)-1 and
                       dataSets[i].set[dataSets[i].currentPos, 0] < maxEvals and
                       vals[i] > currentFitValue):
                    #Get the function values from the last valid position
                    previousVals[i] = vals[i]
                    dataSets[i].currentPos += 1
                    res[i] = dataSets[i].set[dataSets[i].currentPos,
                                             funcEvalsIndex]
                    vals[i] = (dataSets[i].set[dataSets[i].currentPos,
                                               fitValIndex] - fvalueToReach + 
                               1.0e-8)
                if (dataSets[i].currentPos == len(dataSets[i].set) - 1 or 
                    dataSets[i].set[dataSets[i].currentPos, 0] >= maxEvals):
                    isFinished[i] = True

        # Process currentLines:
        # 1) Check that the current lines correspond to the current position
        # either for function values or function evaluations.
        # 2) compute what needs to be computed.
        #if isFuncEvalsAligned:
        #arrayTab = scipy.append(scipy.floor(currentFunctionEvals),
                               #bootstrap.prctile(res))
        #if currentFunctionEvals < 10.:
            #currentFunctionEvals += 1.
        #else:
            #currentFunctionEvals *= 10.**0.05
        #if isFitValAligned:
        if scipy.isinf(currentFitValue):
            currentFitValue = max(valuesOfInterest[0],
                                  10**(scipy.ceil(scipy.log10(max(vals))*5)/5))
            #What does this mean?
        tmp = scipy.append(currentFitValue, computevalues(res,maxEvals))
        for j in range(2,7):
            if tmp[j] == maxEvals:
                tmp[j] = scipy.nan
        arrayFullTab.append(tmp[:])

        #arrayFullTab is displayed in the figures. As opposed to the tables,
        #we are not constrained to use the default maxEvals for comparison
        #sake. We can then use the user-defined maximum number of function
        #evaluations.

        # Tests if we have covered all of the valuesOfInterest (using the
        # iterator iValuesOfInterest and then if 
        # currentFitValue == valuesOfInterest[iValuesOfInterest].
        if (iValuesOfInterest < len(valuesOfInterest) and
            (currentFitValue-valuesOfInterest[iValuesOfInterest] <
             max([currentFitValue * (1.0 - 10.**-0.1),
                  scipy.finfo('float64').resolution]))):
            arrayTab[valuesOfInterest[iValuesOfInterest]] = computevalues(res,
                                                     maxEvals, dispersion=True)
            previousVals.sort()
            ranksOfInterest = [previousVals[0], previousVals[2],
                               bootstrap.prctile(previousVals, 50)[0],
                               previousVals[-3], previousVals[-1]]
            #Gets the last valid function values. Might not be corresponding
            #to maxEvals.
            # minimization
            #set_trace()
            for j in range(4,9): #index of the percentiles
                if arrayTab[valuesOfInterest[iValuesOfInterest]][j] >= maxEvals:
                    arrayTab[valuesOfInterest[iValuesOfInterest]][j] = ranksOfInterest[j-4]
                    #TODO rewrite
                    #use vals either here or in computevalues.
            iValuesOfInterest += 1

        if (iValuesForDataProf < len(valuesForDataProf) and
            (currentFitValue-valuesForDataProf[iValuesForDataProf] <
             max([currentFitValue*(1.0-10.**-0.1),
                  scipy.finfo('float64').resolution]))):
            rlDistr[valuesForDataProf[iValuesForDataProf]]= res[:]
            iValuesForDataProf += 1

        currentFitValue *= 10.**(-0.2)
    arrayFullTab = scipy.vstack(arrayFullTab)
    tmp = []
    for i in valuesOfInterest:
        if i in arrayTab:
            #set_trace()
            tmp.append(scipy.append(scipy.array([i]), arrayTab[i])) #Weird.
        else:
            vals.sort()
            ranksOfInterest = []
            tmp2 = scipy.array((i, scipy.inf, scipy.inf, scipy.inf, 0.0,
                                vals[0], vals[2],
                                bootstrap.prctile(vals, 50)[0], vals[-3],
                                vals[-1])) # minimization
            tmp.append(tmp2)
            #tmp.append(scipy.array((i,scipy.inf,scipy.inf,scipy.inf,len(dataSets),
            #                        maxEvals,maxEvals,maxEvals,maxEvals,
            #                        maxEvals)))
            #Raymond: Why this line? It makes the computation of the probability
            #of success wrong.

    #set_trace()
    arrayTab = scipy.vstack(tmp)
    #set_trace()
    return (arrayFullTab, arrayTab, rlDistr, len(dataSets), maxEvals)


def testpostprocess():
    files = ['/users/dsa/ros/Desktop/coco/BBOB/demo/matlab/datafminunc_worestart/data_f1/datafminunc0_f1_DIM2.dat']
    #files = glob.glob('/users/dsa/ros/Desktop/coco/BBOB/demo/postProcessing/data_f2/*10.dat')
    #res = postprocess(files,1.e-8,scipy.in)
    res = postprocess(files,1.e-8,1.0e5)
    #print res
    return res


def computevalues(N, maxEvals, header=False, dispersion=False):
    """
    Return different measures of a distribution of couple [fevals, fvalues].

    @parameter N - is the array of the fevals.
    @parameter fvalueToReach - is the target function value to reach.
    @parameter maxEvals - is the maximum number of function evaluations.

    @return - [SP1,disp(SP1),success probability,quantiles]
    """
    #set_trace()
    if not header:
        N.sort() # Works because N is supposed to be a 1d array.
        sp1m = bootstrap.sp1(N,maxvalue=maxEvals)
        if dispersion:
            dispersionSP1 = bootstrap.draw(N, [10,90], 15, bootstrap.sp1,
                                           [maxEvals])
            res = [sp1m[0], dispersionSP1[0][0], dispersionSP1[0][1], sp1m[2]]
        else:
            res = [sp1m[0], sp1m[1]] #SP1 and success rate
        if len(N) < 3: #catch error.
            raise Exception, 'Probleme here' #Deal with this.
        res.extend((N[0], N[2], bootstrap.prctile(N,50)[0], N[-3], N[-1]))
        return res

    else:
        # Returns header and format of the entries for one function
        # and one dimension.
        header = ['$\Delta f$', '$\ENFEs$', '10\%', '90\%', '$\#$',
                  'best', '$3^\mathrm{rd}$', 'med.', '$3^\mathrm{rd}w.$', 'worst']
        format = ['%1.1e', '%1.1e', '%1.1e', '%1.1e', '%d',
                  '%1.1e', '%1.1e', '%1.1e', '%1.1e', '%1.1e']
        return header, format


def split(dataFiles):
    """Split the data files into arrays corresponding to the data sets."""
    #TODO: optimize by splitting using %
    dataSets = []
    for fil in dataFiles:
        content = scipy.io.read_array(fil,comment='%')
        dataSetFinalIndex = scipy.where(scipy.diff(content[:,0])<0)[0]
        #splitting is done by comparing the number of function evaluations
        #which should be monotonous.
        if len(dataSetFinalIndex)> 0:
            dataSetFinalIndex += 1
            dataSetFinalIndex = scipy.append(scipy.append(0,dataSetFinalIndex),
                                             None)
            #dataSetFinalIndex = scipy.insert(dataSetFinalIndex,0,-2)
            for i in range(len(dataSetFinalIndex)-1):
                dataSet = DataSet(content[dataSetFinalIndex[i]:
                                          dataSetFinalIndex[i+1],:])
                dataSets.append(dataSet)
        else:
            dataSet = DataSet(content)
            dataSets.append(dataSet)

    return dataSets


def main(indexEntry, verbose=True):
    """Updates an indexEntry with attributes containing post processed data."""
    # TODO: PLEASE describe in short input and output arguments of the function
    #       it is probably more time efficient to do it in advance. 
    #This is clumsy
    maxEvals = scipy.floor(10**(scipy.floor(scipy.log10(maxEvalsFactor * indexEntry.dim)*20.)/20.))
    #maxEvals = maxEvalsFactor * indexEntry.dim

    res = postprocess(indexEntry.dataFiles,
                      indexEntry.targetFuncValue, maxEvals)

    indexEntry.arrayFullTab = res[0]
    indexEntry.arrayTab = res[1]
    indexEntry.rlDistr = res[2]
    indexEntry.nbRuns = res[3]
    #set_trace()
    indexEntry.maxEvals = int(min((maxEvals, res[4])))
    #Problem here: maxEvals is the empirical value and not the one set for the
    #experiment!
    #The maxEvals to appear in the table should be this one.

    if verbose:
        print 'Processed IndexEntry: func %d, dim %d' % (indexEntry.funcId,
                                                         indexEntry.dim)
    #print res[0]
    #print res[1]
    #print indexEntry
    return None

class DataSet:
    def __init__ (self, set):
        self.currentPos = 0
        self.set = set
        self.userDefMaxEvals = set[-1, 0]
