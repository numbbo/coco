#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Algorithm portfolio data set module.

The algorithm portfolio consists in running multiple algorithms in
parallel.
Current limitation: the portfolio data set must come from data sets that
are identical (same number of repetitions on the same instances of the
functions.

**Example:**

.. plot::
   :width: 75%

   import glob
   from pylab import *
   import pickle
   import cocopp
   import cocopp.compall.pprldmany
   import cocopp.algportfolio

   # Collect and unarchive data
   dsets = {}
   for alg in ('BIPOP-CMA-ES_hansen_noiseless', 'NEWUOA_ros_noiseless'):
      dsets[alg] = cocopp.load(cocopp.bbob(alg))

   # Generate the algorithm portfolio
   dspf = cocopp.algportfolio.build(dsets)
   dsets['Portfolio'] = dspf # store the portfolio in dsets

   # plot the run lengths distribution functions
   plt.figure()
   for algname, ds in dsets.items():
      dataset = ds.dictByDimFunc()[10][13]  # DataSet dimension 10 on F13
      cocopp.compall.pprldmany.plot(dataset, label=algname)
   cocopp.compall.pprldmany.beautify()
   legend(loc='best') # Display legend
   plt.show()   
   
"""

# TODO: generalize behaviour for data sets that have different instances...

from __future__ import absolute_import, print_function

import sys
from pdb import set_trace
import warnings
import numpy as np

from . import pproc as pp
from . import readalign as ra

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

class DataSet(pp.DataSet):
    """Unit element of algorithm portfolio data set.

    Modified class attributes:
        - comment -- comments for the setting (list)
        - algId -- algorithm name (list)

    """

    def __init__(self, dslist):
        """Instantiate one algorithm portfolio data set.

        :param dict dslist: list of :py:class:`pproc.DataSetList`
                            instances.

        """

        def _conv_evals(evals, algnb, maxevals):
            if evals > maxevals[algnb]:
                return np.nan
            res = 0.
            mevals = np.asarray(maxevals)
            if evals > len(maxevals) or not isinstance(evals, int):
                smevals = np.sort(mevals)
                for i in smevals:
                    res += min(evals - 1, i)
            else:
                for i in range(1, evals):
                    res += np.sum(i <= mevals)
            res += np.sum(evals <= mevals[:algnb+1])
            return res

        # Checking procedure
        d = set()
        f = set()
        trials = []
        for i in dslist:
            d.add(i.dim)
            f.add(i.funcId)
            trials.append(i.createDictInstanceCount())
        if len(f) > 1 or len(d) > 1:
            raise Usage('%s: Expect the data of algorithms for only one '
                        'function and one dimension.' % (dslist))
        elif trials[1:] != trials[:-1]:
            # this check will be superfluous if we find that all instances
            # are equivalent.
            # raise Usage('%s: Expect the data to have the same instances.'
            #             % (dslist))
            warnings.warn('portfolio will be generated from algorithm with different instances')

        self.dim = d.pop()
        self.funcId = f.pop()
        algId = []
        comment = []
        for i in dslist:
            algId.append(i.algId)
            comment.append(i.comment)
        self.algId = tuple(algId)
        self.comment = tuple(comment)

        # Data handling
        nbruns = dslist[0].nbRuns() # all data sets have the same #runs
        corresp = [[]] * len(dslist)
        if False:
            # find correspondence with respect to first element in dslist
            dictref = dslist[0].createDictInstance()
            for i, ds in enumerate(dslist):
                tmpdict = ds.createDictInstance()
                for j in sorted(dictref):
                    corresp[i].extend(tmpdict[j])
        else:
            for i in range(len(dslist)):
                corresp[i] = range(nbruns)
        self.instancenumbers = trials.pop()
        maxevals = []
        finalfunvals = []
        evals = []
        funvals = []
        for i in range(nbruns):
            tmpmaxevals = []
            tmpfinalfunvals = []
            tmpevals = []
            tmpfunvals = []
            for j, ds in enumerate(dslist):
                tmpmaxevals.append(ds.maxevals[corresp[j][i]])
                tmpfinalfunvals.append(ds.finalfunvals[corresp[j][i]])
                tmpevals.append(ds.evals[:, np.r_[0, corresp[j][i]+1]])
                tmpfunvals.append(ds.funvals[:, np.r_[0, corresp[j][i]+1]].copy())
            maxevals.append(np.sum(tmpmaxevals))
            finalfunvals.append(min(tmpfinalfunvals))
            tmpevals = ra.alignArrayData(ra.HArrayMultiReader(tmpevals))
            tmpres = []
            for j in tmpevals:
                tmp = []
                for k, e in enumerate(j[1:]):
                    tmp.append(_conv_evals(e, k, tmpmaxevals))
                tmpres.append(min(tmp))
            evals.append(np.column_stack((tmpevals[:, 0], tmpres)))

            for j, a in enumerate(tmpfunvals):
                for k in range(len(a[:, 0])):
                    a[k, 0] = _conv_evals(a[k, 0], j, tmpmaxevals)
            tmpfunvals = ra.alignArrayData(ra.VArrayMultiReader(tmpfunvals))
            tmpres = []
            for j in tmpfunvals:
                tmpres.append(min(j[1:]))
            funvals.append(np.column_stack((tmpfunvals[:, 0], tmpres)))
        self.maxevals = np.array(maxevals)
        self.finalfunvals = np.array(finalfunvals)
        self.evals = ra.alignArrayData(ra.HArrayMultiReader(evals))
        self.funvals = ra.alignArrayData(ra.VArrayMultiReader(funvals))
        self.computeERTfromEvals()

#FUNCTION DEFINITIONS

def build(dictAlg, sortedAlg=None):
    """Merge datasets in an algorithm portfolio.

    :param dict dictAlg: dictionary of data sets with algorithm name for
                         keys, see ``pproc.DataSetList.dictByAlg``
    :param seq sortedAlgs: sequence for sorting the entries of
                           :py:data:`dictAlg`, if not provided,
                           dictAlg.keys() will be instead
    :returns: an instance of :py:class:`DataSetList` with the porfolio
              data sets

    """
    if not sortedAlg:
        sortedAlg = dictAlg.keys()
    tmpres = []
    for f, i in pp.dictAlgByFun(dictAlg).items():
        for d, j in pp.dictAlgByDim(i).items():
            tmp = []
            if sortedAlg:
                tmplist = list(j[k] for k in sortedAlg)
            else:
                tmplist = j.values()
            for k in tmplist:
                assert len(k) == 1 # one element list
                tmp.append(k[0])
            try:
                tmpres.append(DataSet(tmp))
            except Usage as err:
                print(err.msg, file=sys.stderr)
    res = pp.DataSetList()
    res.extend(tmpres)
    return res
