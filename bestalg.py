#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate the best algorithm dataset."""

from __future__ import absolute_import

import os
import sys
import glob
import getopt
import pickle
import gzip
from pdb import set_trace
import warnings
import numpy

if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))

from bbob_pproc import readalign
from bbob_pproc.pproc import DataSetList, processInputArgs, dictAlgByFun, dictAlgByDim
from bbob_pproc.dataoutput import algPlotInfos

bestalgentries = {}

algs = ("ALPS", "AMALGAM", "BAYEDA", "BFGS", "Cauchy-EDA",
"CMA-ES", "CMA-ESPLUSSEL", "DASA", "DE-PSO", "DIRECT", "EDA-PSO",
"FULLNEWUOA", "G3PCX", "GA", "GLOBAL", "iAMALGAM", "LSfminbnd", "LSstep",
"MA-LS-CHAIN", "MCS", "NELDER", "NELDERDOERR", "NEWUOA", "ONEFIFTH", "POEMS",
"PSO", "PSO_Bounds", "RANDOMSEARCH", "Rosenbrock", "SEP-CMA-ES", "SNOBFIT",
"VNS")
#Shorter list for test purpose
#algs = ("ALPS", "AMALGAM", "AVGNEWUOA")

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

class BestAlgSet:
    """Unit element for the post-processing with given funcId, algId and
    dimension.
    Class attributes:
        funcId -- function Id (integer)
        dim -- dimension (integer)
        comment -- comment for the setting (string)
        algId -- algorithm name (string)
        evals -- collected data aligned by function values (array)
        maxevals -- maximum number of function evaluations (array)

    evals and funvals are arrays of data collected from N data sets. Both have
    the same format: zero-th column is the value on which the data of a row is
    aligned, the N subsequent columns are either the numbers of function
    evaluations for evals or function values for funvals.
    """

    def __init__(self, dictAlg):

        # values of dict dictAlg are DataSetList which should have only one
        # element which will be assigned as values in the following lines.
        d = set()
        f = set()
        for i in dictAlg.values():
            d |= set(j.dim for j in i)
            f |= set(j.funcId for j in i)

        if len(f) > 1 or len(d) > 1:
            #set_trace()
            Usage('Expect the data of algorithms for only one function and one dimension.')

        f = f.pop()
        d = d.pop()
        rmkeys = []
        dictMaxEvals = {}
        dictFinalFunVals = {}
        for alg, i in dictAlg.iteritems():
            if len(i) != 1:
                # Special case could occur?
                txt = ('Algorithm %s has problem in this case: f%d %d-D.'
                       % (alg, f, d))
                warnings.warn(txt)
                #set_trace()
                rmkeys.append(alg)
            else:
                dictAlg[alg] = i[0] # Assign the first element as value for alg
                #TODO: check it was not assigned already.
                dictMaxEvals[alg] = i[0].maxevals
                dictFinalFunVals[alg] = i[0].finalfunvals
        for i in rmkeys:
            del dictAlg[i]

        sortedAlgs = dictAlg.keys()

        #Align ERT
        erts = list(numpy.transpose(numpy.vstack([dictAlg[i].target, dictAlg[i].ert]))
                    for i in sortedAlgs)
        res = readalign.alignArrayData(readalign.HArrayMultiReader(erts))
        #TODO: check

        resalgs = []
        reserts = []
        #Foreach function value
        for i in res:
            #find best algorithm
            tmp = i[1:]
            idxfinite = (numpy.isnan(tmp) == False)
            tmp = tmp[idxfinite]
            # Check that it is not empty.
            tmpsortedAlgs = []
            for j, k in enumerate(idxfinite):
                if k:
                    tmpsortedAlgs.append(sortedAlgs[j])
            idx = numpy.argmin(tmp) # TODO: What in case of ties?
            reserts.append(tmp[idx])
            resalgs.append(tmpsortedAlgs[idx])

        dictiter = {}
        dictcurLine = {}
        resDataSet = []
        #write down the #fevals to reach the function value.
        for funval, alg in zip(res[:, 0], resalgs):
            it = dictiter.setdefault(alg, iter(dictAlg[alg].evals))
            curLine = dictcurLine.setdefault(alg, numpy.array([numpy.inf, 0]))

            while curLine[0] > funval:
               try:
                   curLine = it.next()
               except StopIteration:
                   break
            dictcurLine[alg] = curLine.copy()
            tmp = curLine.copy()
            tmp[0] = funval
            resDataSet.append(tmp)

        #set_trace()
        #TODO: do some testings.
        #TODO: resds should be an instance of DataSet
        self.evals = resDataSet
        # evals is not a numpy array but a list of arrays because they may not
        # all be of the same size.
        self.maxevals = dictMaxEvals
        self.finalfunvals = dictFinalFunVals
        self.dim = d
        self.funcId = f
        # What if some algorithms don't have the same number of runs
        # How do we save maxfunevals (to compute the ERT...?)
        self.algs = resalgs
        self.algId = 'Virtual Best Algorithm of BBOB 2009'
        self.comment = 'Combination of ' + ', '.join(algs)
        self.ert = numpy.array(reserts)
        self.target = res[:, 0]
        #return resds

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

    def pickle(self, outputdir=None, verbose=True):
        """Save DataSet instance to a pickle file.
        Saves the instance of DataSet to a pickle file. If not specified by
        argument outputdir, the location of the pickle is given by the location
        of the first index file associated to the DataSet.
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
        """Returns a dictionary of the instances: the key is the instance id,
        the value is a list of index.
        """
        dictinstance = {}
        for i in range(len(self.itrials)):
            dictinstance.setdefault(self.itrials[i], []).append(i)

        return dictinstance

    def detERT(self, targets):
        res = []
        for f in targets:
            idx = (self.target<=f)
            try:
                res.append(self.ert[idx][0])
            except IndexError:
                res.append(numpy.inf)
        return res

    def detEvals(self, targets):
        res = []
        res2 = []
        for f in targets:
            tmp = numpy.array([numpy.nan])
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

def loadBBOB2009():
    global bestalgentries
    # global statement necessary to change the variable bestalg.bestalgentries

    #bestalgfilepath = os.path.join(os.path.split(__file__)[0], 'bestalg')
    #for fun in range(1, 25)+range(101, 131):
        #for D in [2, 3, 5, 10, 20, 40]:
            #picklefilename = os.path.join(bestalgfilepath,
                                          #'bestalg_f%03d_%02d.pickle.gz' % (fun, D))
            ##TODO: what if file is not found?
            #fid = gzip.open(picklefilename, 'r')
            #bestalgentries[(D, fun)] = pickle.load(fid)
            #fid.close()

    print "Loading best algorithm data from BBOB-2009."
    bestalgfilepath = os.path.split(__file__)[0]
    picklefilename = os.path.join(bestalgfilepath, 'bestalgentries2009.pickle.gz')
    #TODO: what if file is not found?
    fid = gzip.open(picklefilename, 'r')
    #set_trace()
    bestalgentries = pickle.load(fid)
    fid.close()
    print "Done."


def usage():
    print main.__doc__

def main(argv=None):
    """
    """

    if argv is None:
        argv = sys.argv[1:]

    try:
        try:
            opts, args = getopt.getopt(argv, "h",
                                       ["help"])
        except getopt.error, msg:
             raise Usage(msg)

        args = algs
        # if not (args):
        #     usage()
        #     sys.exit()
        outputdir = 'bestAlg'

        verbose = True

        #Process options
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                sys.exit()
            else:
                assert False, "unhandled option"

        dsList, sortedAlgs, dictAlg = processInputArgs(args, verbose=verbose)

        #set_trace()
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
            if verbose:
                print 'Folder %s was created.' % (outputdir)

        res = {}
        for f, i in dictAlgByFun(dictAlg).iteritems():
            for d, j in dictAlgByDim(i).iteritems():
                tmp = BestAlgSet(j)
                picklefilename = os.path.join(outputdir,
                                             'bestalg_f%03d_%02d.pickle' % (f, d))
                fid = open(picklefilename, 'w')
                pickle.dump(tmp, fid, 2)
                fid.close()
                res[(d, f)] = tmp
        picklefilename = os.path.join(outputdir, 'bestalg.pickle')
        fid = open(picklefilename, 'w')
        pickle.dump(res, fid, 2)
        fid.close()

        #dataoutput.outputPickle(resDsList)

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
   from bbob_pproc.bestalg import BestAlgSet
   # Above line is necessary to specify from which module the BestAlgset class
   # is from.
   sys.exit(main())
