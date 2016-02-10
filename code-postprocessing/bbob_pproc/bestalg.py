#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Best algorithm dataset module 

    This module implements :py:class:`BestAlgSet` class which is used as
    data structure for the data set of the virtual best algorithm.
    Therefore this module will be imported by other modules which need
    to access best algorithm data set.

    The best algorithm data set can be accessed by the
    :py:data:`bestalgentries2009` variable. This variable needs to be
    initialized by executing functions :py:func:`loadBBOB2009()`

    This module can also be used generate the best algorithm data set
    with its generate method.

"""

from __future__ import absolute_import

import os
import sys
import glob
import getopt
import pickle
import gzip
from pdb import set_trace
import warnings
import numpy as np

from . import genericsettings, readalign, pproc
from .toolsdivers import print_done
from . import toolsstats

bestalgentries2009 = {}
bestalgentries2010 = {}
bestalgentries2012 = {}
bestalgentriesever = {}
bestbiobjalgentries2016 = {}

algs2009 = ("ALPS", "AMALGAM", "BAYEDA", "BFGS", "Cauchy-EDA",
"BIPOP-CMA-ES", "CMA-ESPLUSSEL", "DASA", "DE-PSO", "DIRECT", "EDA-PSO",
"FULLNEWUOA", "G3PCX", "GA", "GLOBAL", "iAMALGAM", "IPOP-SEP-CMA-ES",
"LSfminbnd", "LSstep", "MA-LS-CHAIN", "MCS", "NELDER", "NELDERDOERR", "NEWUOA",
"ONEFIFTH", "POEMS", "PSO", "PSO_Bounds", "RANDOMSEARCH", "Rosenbrock",
"SNOBFIT", "VNS")

# Warning: NEWUOA is there twice: NEWUOA noiseless is a 2009 entry, NEWUOA
# noisy is a 2010 entry
algs2010 = ("1komma2", "1komma2mir", "1komma2mirser", "1komma2ser", "1komma4",
"1komma4mir", "1komma4mirser", "1komma4ser", "1plus1", "1plus2mirser", "ABC",
"AVGNEWUOA", "CMAEGS", "DE-F-AUC", "DEuniform", "IPOP-ACTCMA-ES",
"BIPOP-CMA-ES", "MOS", "NBC-CMA", "NEWUOA", "PM-AdapSS-DE", "RCGA", "SPSA",
"oPOEMS", "pPOEMS")

algs2012 = ("ACOR", "BIPOPaCMA", "BIPOPsaACM", "aCMA", "CMAES", "aCMAa", "aCMAm", "aCMAma", "aCMAmah", "aCMAmh", "DBRCGA", "DE", "DEAE", "DEb", "DEctpb", "IPOPsaACM", "JADE", "JADEb", "JADEctpb", "NBIPOPaCMA", "NIPOPaCMA", "DE-AUTO", "DE-BFGS", "DE-ROLL", "DE-SIMPLEX", "MVDE", "PSO-BFGS", "xNES", "xNESas", "SNES")

# TODO: this should be reimplemented: 
#  o a best algorithm should derive from the DataSet class
#  o a best algorithm and an algorithm portfolio are almost the same, 
#    they should derive from a CombinedAlgorithmDataSet? 

#CLASS DEFINITIONS


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

class BestAlgSet():
    """Unit element of best algorithm data set.

    Here unit element means for one function and one dimension.
    This class is derived from :py:class:`DataSet` but it does not
    inherit from it.

    Class attributes:
        - funcId -- function Id (integer)
        - dim -- dimension (integer)
        - comment -- comment for the setting (string)
        - algId -- algorithm name (string)
        - evals -- collected data aligned by function values (array)
        - maxevals -- maximum number of function evaluations (array)

    evals and funvals are arrays of data collected from N data sets.
    Both have the same format: zero-th column is the value on which the
    data of a row is aligned, the N subsequent columns are either the
    numbers of function evaluations for evals or function values for
    funvals.

    Known bug: algorithms where the ERT is NaN or Inf are not taken into
    account!?
    
    """

    def __init__(self, dictAlg):
        """Instantiate one best algorithm data set.
        
        :keyword dictAlg: dictionary of datasets, keys are algorithm
                          names, values are 1-element
                          :py:class:`DataSetList`.
        
        """

        # values of dict dictAlg are DataSetList which should have only one
        # element which will be assigned as values in the following lines.
        d = set()
        f = set()
        for i in dictAlg.values():
            d |= set(j.dim for j in i)
            f |= set(j.funcId for j in i)

        if len(f) > 1 or len(d) > 1:
            Usage('Expect the data of algorithms for only one function and '
                  'one dimension.')

        f = f.pop()
        d = d.pop()

        dictMaxEvals = {}
        dictFinalFunVals = {}
        tmpdictAlg = {}
        for alg, i in dictAlg.iteritems():
            if len(i) == 0:
                warnings.warn('Algorithm %s was not tested on f%d %d-D.'
                              % (alg, f, d))
                continue
            elif len(i) > 1:
                warnings.warn('Algorithm %s has a problem on f%d %d-D.'
                              % (alg, f, d))
                continue

            tmpdictAlg[alg] = i[0] # Assign ONLY the first element as value
            dictMaxEvals[alg] = i[0].maxevals
            dictFinalFunVals[alg] = i[0].finalfunvals

        dictAlg = tmpdictAlg

        sortedAlgs = dictAlg.keys()
        # algorithms will be sorted along sortedAlgs which is now a fixed list

        # Align ERT
        erts = list(np.transpose(np.vstack([dictAlg[i].target, dictAlg[i].ert]))
                    for i in sortedAlgs)
        res = readalign.alignArrayData(readalign.HArrayMultiReader(erts, False))

        resalgs = []
        reserts = []
        # For each function value
        for i in res:
            # Find best algorithm
            curerts = i[1:]
            assert len((np.isnan(curerts) == False)) > 0
            currentbestert = np.inf
            currentbestalg = ''
            for j, tmpert in enumerate(curerts):
                if np.isnan(tmpert):
                    continue # TODO: don't disregard these entries
                if tmpert == currentbestert:
                    # TODO: what do we do in case of ties?
                    # look at function values corresponding to the ERT?
                    # Look at the function evaluations? the success ratio?
                    pass
                elif tmpert < currentbestert:
                    currentbestert = tmpert
                    currentbestalg = sortedAlgs[j]
            reserts.append(currentbestert)
            resalgs.append(currentbestalg)

        dictiter = {}
        dictcurLine = {}
        resDataSet = []

        # write down the #fevals to reach the function value.
        for funval, alg in zip(res[:, 0], resalgs):
            it = dictiter.setdefault(alg, iter(dictAlg[alg].evals))
            curLine = dictcurLine.setdefault(alg, np.array([np.inf, 0]))
            while curLine[0] > funval:
                try:
                    curLine = it.next()
                except StopIteration:
                    break
            dictcurLine[alg] = curLine.copy()
            tmp = curLine.copy()
            tmp[0] = funval
            resDataSet.append(tmp)

        setalgs = set(resalgs)
        dictFunValsNoFail = {}
        for alg in setalgs:
            for curline in dictAlg[alg].funvals:
                if (curline[1:] == dictAlg[alg].finalfunvals).any():
                    # only works because the funvals are monotonous
                    break
            dictFunValsNoFail[alg] = curline.copy()

        self.evals = resDataSet
        # evals is not a np array but a list of arrays because they may not
        # all be of the same size.
        self.maxevals = dict((i, dictMaxEvals[i]) for i in setalgs)
        self.finalfunvals = dict((i, dictFinalFunVals[i]) for i in setalgs)
        self.funvalsnofail = dictFunValsNoFail
        self.dim = d
        self.funcId = f
        self.algs = resalgs
        self.algId = 'Virtual Best Algorithm'
        self.comment = 'Combination of ' + ', '.join(sortedAlgs)
        self.ert = np.array(reserts)
        self.target = res[:, 0]

        bestfinalfunvals = np.array([np.inf])
        for alg in sortedAlgs:
            if np.median(dictAlg[alg].finalfunvals) < np.median(bestfinalfunvals):
                bestfinalfunvals = dictAlg[alg].finalfunvals
                algbestfinalfunvals = alg
        self.bestfinalfunvals = bestfinalfunvals
        self.algbestfinalfunvals = algbestfinalfunvals

    def __eq__(self, other):
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

    def pickle(self, outputdir=None, verbose=True):
        """Save instance to a pickle file.

        Saves the instance to a pickle file. If not specified
        by argument outputdir, the location of the pickle is given by
        the location of the first index file associated.

        """

        # the associated pickle file does not exist
        if not getattr(self, 'pickleFile', False):
            if outputdir is None:
                outputdir = os.path.split(self.indexFiles[0])[0] + '-pickle'
                if not os.path.isdir(outputdir):
                    try:
                        os.mkdir(outputdir)
                    except OSError:
                        print ('Could not create output directory % for pickle files'
                               % outputdir)
                        raise

            self.pickleFile = os.path.join(outputdir,
                                           'bestalg_f%03d_%02d.pickle'
                                            %(self.funcId, self.dim))

        if getattr(self, 'modsFromPickleVersion', True):
            try:
                f = open(self.pickleFile, 'w') # TODO: what if file already exist?
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
        """Returns a dictionary of the instances

        The key is the instance id, the value is a list of index.

        """

        dictinstance = {}
        for i in range(len(self.instancenumbers)):
            dictinstance.setdefault(self.instancenumbers[i], []).append(i)

        return dictinstance

    def detERT(self, targets):
        """Determine the expected running time to reach target values.

        :keyword list targets: target function values of interest

        :returns: list of expected running times corresponding to the
                  targets.

        """
        res = []
        for f in targets:
            idx = (self.target<=f)
            try:
                res.append(self.ert[idx][0])
            except IndexError:
                res.append(np.inf)
        return res
    # TODO: return the algorithm here as well.

    def detEvals(self, targets):
        """Determine the number of evaluations to reach target values.

        :keyword seq targets: target precisions
        :returns: list of arrays each corresponding to one value in
                  targets and the list of the corresponding algorithms

        """
        res = []
        res2 = []
        for f in targets:
            tmp = np.array([np.nan] * len(self.bestfinalfunvals))
            tmp2 = None
            for i, line in enumerate(self.evals):
                if line[0] <= f:
                    tmp = line[1:]
                    tmp2 = self.algs[i]
                    break
            res.append(tmp)
            res2.append(tmp2)
        return res, res2

#FUNCTION DEFINITIONS

def loadBBOB2009(force=False):
    """Assigns :py:data:`bestalgentries2009`.

    This function is needed to set the global variable
    :py:data:`bestalgentries2009`. It unpickles file 
    :file:`bestalgentries2009.pickle.gz`

    :py:data:`bestalgentries2009` is a dictionary accessed by providing
    a tuple :py:data:`(dimension, function)`. This returns an instance
    of :py:class:`BestAlgSet`.
    The data is that of algorithms submitted to BBOB 2009, the list of
    which can be found in variable :py:data:`algs2009`.

    """
    global bestalgentries2009
    # global statement necessary to change the variable bestalg.bestalgentries2009

    if not force and bestalgentries2009:
        return
    
    print "Loading best algorithm data from BBOB-2009...",
    sys.stdout.flush()
 
    bestalgfilepath = os.path.split(__file__)[0]
    #    picklefilename = os.path.join(bestalgfilepath, 'bestalgentries2009.pickle')
    #    cocofy(picklefilename)
    #    fid = open(picklefilename, 'r')

    picklefilename = os.path.join(bestalgfilepath, 'bestalgentries2009.pickle.gz')
    fid = gzip.open(picklefilename, 'r')
    try:
        bestalgentries2009 = pickle.load(fid)
    except:
        warnings.warn("no best algorithm loaded")
        # raise  # outcomment to diagnose
        bestalgentries2009 = None
    fid.close()
    print_done()

def loadBBOB2010():
    """Assigns :py:data:`bestalgentries2010`.

    This function is needed to set the global variable
    :py:data:`bestalgentries2010`. It unpickles file 
    :file:`bestalgentries2010.pickle.gz`

    :py:data:`bestalgentries2010` is a dictionary accessed by providing
    a tuple :py:data:`(dimension, function)`. This returns an instance
    of :py:class:`BestAlgSet`.
    The data is that of algorithms submitted to BBOB 20&0, the list of
    which can be found in variable :py:data:`algs2010`.

    """
    global bestalgentries2010
    # global statement necessary to change the variable bestalg.bestalgentries2010

    print "Loading best algorithm data from BBOB-2010...",  
    bestalgfilepath = os.path.split(__file__)[0]
    picklefilename = os.path.join(bestalgfilepath, 'bestalgentries2010.pickle.gz')
    #    cocofy(picklefilename)
    fid = gzip.open(picklefilename, 'r')
    bestalgentries2010 = pickle.load(fid)
    fid.close()
    print " done."

def loadBBOB2012():
    """Assigns :py:data:`bestalgentries2012`.

    This function is needed to set the global variable
    :py:data:`bestalgentries2012`. It unpickles file 
    :file:`bestalgentries2012.pickle.gz`

    :py:data:`bestalgentries2012` is a dictionary accessed by providing
    a tuple :py:data:`(dimension, function)`. This returns an instance
    of :py:class:`BestAlgSet`.
    The data is that of algorithms submitted to BBOB 20&0, the list of
    which can be found in variable :py:data:`algs2012`.

    """
    global bestalgentries2012
    # global statement necessary to change the variable bestalg.bestalgentries2012

    print "Loading best algorithm data from BBOB-2012...",  
    bestalgfilepath = os.path.split(__file__)[0]
    picklefilename = os.path.join(bestalgfilepath, 'bestalgentries2012.pickle.gz')
    #    cocofy(picklefilename)
    fid = gzip.open(picklefilename, 'r')
    bestalgentries2012 = pickle.load(fid)
    fid.close()
    print " done."

def loadBBOBever():
    """Assigns :py:data:`bestalgentriesever`.

    This function is needed to set the global variable
    :py:data:`bestalgentriesever`. It unpickles file 
    :file:`bestalgentriesever.pickle.gz`

    :py:data:`bestalgentriesever` is a dictionary accessed by providing
    a tuple :py:data:`(dimension, function)`. This returns an instance
    of :py:class:`BestAlgSet`.
    The data is that of algorithms submitted to BBOB 2009 and 2010, the
    list of which is the union in variables :py:data:`algs2009`
    and :py:data:`algs2010`.

    """
    global bestalgentriesever
    # global statement necessary to change the variable bestalg.bestalgentriesever

    print "Loading best algorithm data from BBOB...",  
    bestalgfilepath = os.path.split(__file__)[0]
    picklefilename = os.path.join(bestalgfilepath, 'bestalgentriesever.pickle.gz')
    #    cocofy(picklefilename)
    fid = gzip.open(picklefilename, 'r')
    bestalgentriesever = pickle.load(fid)
    fid.close()
    print " done."

def loadBestBiobj2016():
    """Assigns :py:data:`bestbiobjalgentries2016`.

    This function is needed to set the global variable
    :py:data:`bestbiobjalgentries2016`. It unpickles file 
    :file:`bestbiobjalgentries2016.pickle.gz`

    :py:data:`bestbiobjalgentries2016` is a dictionary accessed by providing
    a tuple :py:data:`(dimension, function)`. This returns an instance
    of :py:class:`BestAlgSet`.

    """
    global bestbiobjalgentries2016
    # global statement necessary to change the variable bestalg.bestbiobjalgentries2016

    if bestbiobjalgentries2016:
        return

    print "Loading best bi-objective algorithm data from BBOB-2016...",  
    sys.stdout.flush()

    bestalgfilepath = os.path.split(__file__)[0]
    #picklefilename = os.path.join(bestalgfilepath, 'bestbiobjalgentries2016.pickle.gz')
    picklefilename = os.path.join(bestalgfilepath, 'bestbiobjalgentries2016.pickle')
    #fid = gzip.open(picklefilename, 'r')
    fid = open(picklefilename, 'r')
    bestbiobjalgentries2016 = pickle.load(fid)
    fid.close()
    print_done()

def loadBestAlgorithm(isBioobjective):
    """Loads the best single or bi objective algorithm. """
    
    if isBioobjective:
        # Currently we do not have a good best algorithm for the bi-objective case.
        return None
#        if not bestbiobjalgentries2016:
#            loadBestBiobj2016()
#        return bestbiobjalgentries2016
    else:
        if not bestalgentries2009:
            loadBBOB2009()
        return bestalgentries2009

def usage():
    print __doc__  # same as: sys.modules[__name__].__doc__, was: main.__doc__

def generate(dictalg):
    """Generates dictionary of best algorithm data set.
    """

    # dsList, sortedAlgs, dictAlg = processInputArgs(args, verbose=verbose)
    res = {}
    for f, i in pproc.dictAlgByFun(dictalg).iteritems():
        for d, j in pproc.dictAlgByDim(i).iteritems():
            tmp = BestAlgSet(j)
            res[(d, f)] = tmp
    return res

def customgenerate(args = algs2009):
    """Generates best algorithm data set.

    It will create a folder bestAlg in the current working directory
    with a pickle file corresponding to the bestalg dataSet of the
    algorithms listed in variable args.

    This method is called from the python command line from a directory
    containing all necessary data folders::

    >>> from bbob_pproc import bestalg
    >>> import os
    >>> path = os.path.abspath(os.path.dirname(os.path.dirname('__file__')))
    >>> os.chdir(os.path.join(path, 'data'))
    >>> infoFile = 'ALPS/bbobexp_f2.info'
    >>> if not os.path.exists(infoFile):
    ...     import urllib
    ...     import tarfile
    ...     dataurl = 'http://coco.gforge.inria.fr/data-archive/2009/ALPS_hornby_noiseless.tgz'
    ...     filename, headers = urllib.urlretrieve(dataurl)
    ...     archivefile = tarfile.open(filename)
    ...     archivefile.extractall()
    >>> os.chdir(os.path.join(path, 'data'))
    >>> bestalg.customgenerate(('ALPS', '')) # doctest: +ELLIPSIS
    Searching in...
    >>> os.chdir(path)

    """

    outputdir = 'bestCustomAlg'

    verbose = True
    dsList, sortedAlgs, dictAlg = pproc.processInputArgs(args, verbose=verbose)

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        if verbose:
            print 'Folder %s was created.' % (outputdir)

    res = generate(dictAlg)
    picklefilename = os.path.join(outputdir, 'bestalg.pickle')
    fid = open(picklefilename, 'w')
    pickle.dump(res, fid)
    fid.close()

    print 'done with writing pickle...'

def getAllContributingAlgorithmsToBest(algnamelist, target_lb=1e-8, 
                                       target_ub=1e2):
    """Computes first the artificial best algorithm from given algorithm list
       algnamelist, constructed by extracting for each target/function pair
       the algorithm with best ERT among the given ones. Returns then the list
       of algorithms that are contributing to the definition of the best
       algorithm, separated by dimension, and sorted by importance (i.e. with
       respect to the number of target/function pairs where each algorithm is
       best). Only target/function pairs are taken into account where the target
       is in between target_lb and target_ub.
       This method should be called from the python command line from a directory
       containing all necessary data folders::

        >>> from bbob_pproc import bestalg
        >>> import os
        >>> path = os.path.abspath(os.path.dirname(os.path.dirname('__file__')))
        >>> os.chdir(path)
        >>> os.chdir(os.path.join(path, "data"))
        >>> bestalg.getAllContributingAlgorithmsToBest(('IPOP-CMA-ES', 'RANDOMSEARCH')) # doctest:+ELLIPSIS
        Generating best algorithm data...
        >>> os.chdir(path)

    """

    print "Generating best algorithm data from given algorithm list...\n",  
    customgenerate(algnamelist)
    
    bestalgfilepath = 'bestCustomAlg'
    picklefilename = os.path.join(bestalgfilepath, 'bestalg.pickle')
    fid = open(picklefilename, 'r')
    bestalgentries = pickle.load(fid)
    fid.close()
    print 'loading of best algorithm data done.'
    
    countsperalgorithm = {}
    for (d, f) in bestalgentries:
        print 'dimension:', d, ', function:', f
        print f
        setofalgs = set(bestalgentries[d,f].algs)
        # pre-processing data to only look at targets >= target_lb:
        correctedbestalgentries = []
        for i in range(0,len(bestalgentries[d,f].target)):
            if ((bestalgentries[d,f].target[i] >= target_lb) and
                (bestalgentries[d,f].target[i] <= target_ub)):
                
                correctedbestalgentries.append(bestalgentries[d,f].algs[i])
        print len(correctedbestalgentries)
        # now count how often algorithm a is best for the extracted targets
        for a in setofalgs:
            # use setdefault to initialize with zero if a entry not existant:
            countsperalgorithm.setdefault((d, a), 0) 
            countsperalgorithm[(d,a)] += correctedbestalgentries.count(a)
            
    selectedalgsperdimension = {}
    for (d,a) in sorted(countsperalgorithm):
        if not selectedalgsperdimension.has_key(d):
            selectedalgsperdimension[d] = []
        selectedalgsperdimension[d].append((countsperalgorithm[(d,a)], a))
    
    for d in sorted(selectedalgsperdimension):
        print d, 'D:'
        for (count, alg) in sorted(selectedalgsperdimension[d], reverse=True):
            print count, alg
        print '\n'
    
    
    print " done."

    
def extractBestAlgorithms(args = algs2009, f_factor=2,
                          target_lb=1e-8, target_ub=1e22):
    """Returns (and prints) per dimension a list of algorithms within
    algorithm list args that contains an algorithm if for any
        dimension/target/function pair this algorithm:
        - is the best algorithm wrt ERT
        - its own ERT lies within a factor f_factor of the best ERT
        - there is no algorithm within a factor of f_factor of the best ERT
          and the current algorithm is the second best.

    """

    # TODO: use pproc.TargetValues class as input target values
    # default target values:
    targets = pproc.TargetValues(
        10**np.arange(np.log10(max((1e-8, target_lb))),
                      np.log10(target_ub) + 1e-9, 0.2))
    # there should be a simpler way to express this to become the
    # interface of this function

    print 'Loading algorithm data from given algorithm list...\n'  

    verbose = True
    dsList, sortedAlgs, dictAlg = pproc.processInputArgs(args, verbose=verbose)

    print 'This may take a while (depending on the number of algorithms)'

    selectedAlgsPerProblem = {}
    for f, i in pproc.dictAlgByFun(dictAlg).iteritems():
        for d, j in pproc.dictAlgByDim(i).iteritems():
            selectedAlgsPerProblemDF = []
            best = BestAlgSet(j)
            
            for i in range(0, len(best.target)):
                t = best.target[i]
                # if ((t <= target_ub) and (t >= target_lb)):
                if toolsstats.in_approximately(t,
                                    targets((f, d), discretize=True)):
                    # add best for this target:
                    selectedAlgsPerProblemDF.append(best.algs[i])
                
                    # add second best or all algorithms that have an ERT
                    # within a factor of f_factor of the best:
                    secondbest_ERT = np.infty
                    secondbest_str = ''
                    secondbest_included = False        
                    for astring in j:
                        currdictalg = dictAlg[astring].dictByDim()
                        if currdictalg.has_key(d):
                            curralgdata = currdictalg[d][f-1]                        
                            currERT = curralgdata.detERT([t])[0]
                            if (astring != best.algs[i]):
                                if (currERT < secondbest_ERT):
                                    secondbest_ERT = currERT
                                    secondbest_str = astring
                                if (currERT <= best.detERT([t])[0] * f_factor):
                                    selectedAlgsPerProblemDF.append(astring)
                                    secondbest_included = True
                    if not (secondbest_included) and (secondbest_str != ''):
                        selectedAlgsPerProblemDF.append(secondbest_str)
            
            if len(selectedAlgsPerProblemDF) > 0:
                selectedAlgsPerProblem[(d, f)] = selectedAlgsPerProblemDF
        
        print 'pre-processing of function', f, 'done.'                    
                                  
    print 'loading of best algorithm(s) data done.'
    
    countsperalgorithm = {}
    for (d, f) in selectedAlgsPerProblem:
        print 'dimension:', d, ', function:', f
        setofalgs = set(selectedAlgsPerProblem[d,f])
        
        # now count how often algorithm a is best for the extracted targets
        for a in setofalgs:
            # use setdefault to initialize with zero if a entry not existant:
            countsperalgorithm.setdefault((d, a), 0) 
            countsperalgorithm[(d,a)] += selectedAlgsPerProblem[d,f].count(a)
            
    selectedalgsperdimension = {}
    for (d,a) in sorted(countsperalgorithm):
        if not selectedalgsperdimension.has_key(d):
            selectedalgsperdimension[d] = []
        selectedalgsperdimension[d].append((countsperalgorithm[(d,a)], a))
    
    for d in sorted(selectedalgsperdimension):
        print d, 'D:'
        for (count, alg) in sorted(selectedalgsperdimension[d], reverse=True):
            print count, alg
        print '\n'
    
    
    print " done."
    
    return selectedalgsperdimension
