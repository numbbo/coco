#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bootstrapping and statistics routines."""

import warnings
import numpy as np
from bbob_pproc import genericsettings
from pdb import set_trace


def fix_data_number(data, ndata=15,
                       last_elements_randomized=True, warn=False):
    """return copy of data vector modified to length ``ndata``
    or ``data`` itself.

    Assures ``len(data) == ndata``.

    :param data: is a (row)-vector

    >>> from bbob_pproc.toolsstats import fix_data_number
    >>> data = [1,2,4]
    >>> assert len(fix_data_number(data, 1)) == 1
    >>> assert len(fix_data_number(data, 3)) == 3
    >>> assert len(fix_data_number(data, 4)) == 4
    >>> assert len(fix_data_number(data, 14)) == 14
    >>> assert fix_data_number(data, 14)[2] == data[2]

    """
    if len(data) == ndata:
        return data
    len_ = len(data)
    if warn:
        warnings.warn(str([len_, ndata]) +
                      ' actual and desired number of data disagree')
    if len_ > ndata:
        if last_elements_randomized:
            return np.random.permutation(data)[:ndata]
        else:
            return data[:ndata]
    if ndata >= 2 * len_:
        data = np.hstack((ndata // len_) * [data])
    if len(data) < ndata:
        # append some permuted original data
        if last_elements_randomized:
            few_data = np.random.permutation(data[:len_])[:ndata-len(data)]
        else:
            few_data = data[:ndata-len(data)]
        data = np.hstack([data, few_data])
    assert len(data) == ndata
    return data


def sp1(data, maxvalue=np.Inf, issuccessful=None):
    """sp1(data, maxvalue=Inf, issuccessful=None) computes a
    mean value over successful entries in data divided by
    success rate, the so-called SP1

    Input:
      data -- array contains, e.g., number of function
        evaluations to reach the target value
      maxvalue -- number, if issuccessful is not provided, data[i]
        is defined successful if it is truly smaller than maxvalue
      issuccessful -- None or array of same length as data. Entry
         i in data is defined successful, if issuccessful[i] is
         True or non-zero 

    Returns: (SP1, success_rate, nb_of_successful_entries), where
      SP1 is the mean over successful entries in data divided
      by the success rate. SP1 equals np.Inf when the success
      rate is zero.
    """

    # check input args
    if not getattr(data, '__iter__', False):  # is not iterable
        raise Exception, 'data must be a sequence'
    if issuccessful is not None:
        if not getattr(issuccessful, '__iter__', False):  # is not iterable
            raise Exception, 'issuccessful must be a sequence or None'
        if len(issuccessful) != len(data):
            raise Exception, 'lengths of data and issuccessful disagree'

    # remove NaNs
    if issuccessful is not None:
        issuccessful = [issuccessful[i] for i in xrange(len(issuccessful))
                        if not np.isnan(data[i])]
    dat = [d for d in data if not np.isnan(d)]
    N = len(dat)

    if N == 0:
        return(np.nan, np.nan, np.nan)

    # remove unsuccessful data
    if issuccessful is not None:
        dat = [dat[i] for i in xrange(len(dat)) if issuccessful[i]]
    else:
        dat = [d for d in dat if d < maxvalue]
    succ = float(len(dat)) / N

    # return
    if succ == 0:
        return (np.Inf, 0., 0)
    else:
        return (np.mean(dat) / succ, succ, len(dat))

def sp(data, maxvalue=np.Inf, issuccessful=None, allowinf=True):
    """sp(data, issuccessful=None) computes the sum of the function evaluations
    over all runs divided by the number of success, the so-called SP.

    Input:
      data -- array contains, e.g., number of function
        evaluations to reach the target value
      maxvalue -- number, if issuccessful is not provided, data[i]
        is defined successful if it is truly smaller than maxvalue
      issuccessful -- None or array of same length as data. Entry
         i in data is defined successful, if issuccessful[i] is
         True or non-zero
      allowinf -- If False, replace inf output (in case of no success)
         with the sum of function evaluations.

    Returns: (SP, success_rate, nb_of_successful_entries), where SP is the sum
      of successful entries in data divided by the number of success.
    """

    # TODO allowinf is obsolete

    # check input args
    if not getattr(data, '__iter__', False):  # is not iterable
        raise Exception, 'data must be a sequence'
    if issuccessful is not None:
        if not getattr(issuccessful, '__iter__', False):  # is not iterable
            raise Exception, 'issuccessful must be a sequence or None'
        if len(issuccessful) != len(data):
            raise Exception, 'lengths of data and issuccessful disagree'

    # remove NaNs
    if issuccessful is not None:
        issuccessful = [issuccessful[i] for i in xrange(len(issuccessful))
                        if not np.isnan(data[i])]
    dat = [d for d in data if not np.isnan(d)]
    N = len(dat)

    if N == 0:
        return(np.nan, np.nan, np.nan)

    # remove unsuccessful data
    if issuccessful is not None:
        succdat = [dat[i] for i in xrange(len(dat)) if issuccessful[i]]
    else:
        succdat = [d for d in dat if d < maxvalue]
    succ = float(len(succdat)) / N

    # return
    if succ == 0:
        if not allowinf:
            res = np.sum(dat)  # here it is divided by min(1, succ)
        else:
            res = np.inf
    else:
        res = np.sum(dat) / float(len(succdat))

    return (res, succ, len(succdat))

def drawSP_from_dataset(data_set, ftarget, percentiles, samplesize=genericsettings.simulated_runlength_bootstrap_sample_size):
    """returns ``(percentiles, all_sampled_values_sorted)`` of simulated 
    runlengths to reach ``ftarget`` based on a ``DataSet`` class instance, 
    specifically:: 
     
        evals = data_set.detEvals([ftarget])[0] # likely to be 15 "data points"
        idx_nan = np.isnan(evals)  # nan == did not reach ftarget
        return drawSP(evals[~idx_nan], data_set.maxevals[idx_nan], percentiles, samplesize)
    
    The expected value of ``all_sampled_values_sorted`` is the expected 
    runtime ERT, as obtained by ``data_set.detERT([ftarget])[0]``. 
    
    """
    try:
        evals = data_set.detEvals([ftarget])[0]
    except AttributeError:
        print 'drawSP_from_dataset expects a DataSet instance as first input, was: ' + str(type(data_set))
        raise 
    nanidx = np.isnan(evals)
    return drawSP(evals[~nanidx], data_set.maxevals[nanidx], percentiles, samplesize)
    
def drawSP(runlengths_succ, runlengths_unsucc, percentiles, samplesize=10 + 990 / (1 + 10 * genericsettings.in_a_hurry)):
    """Returns the percentiles of the bootstrapped distribution of
    'simulated' running lengths of successful runs.

    Input:
      - *runlengths_succ* -- array of running lengths of successful runs
      - *runlengths_unsucc* -- array of running lengths of unsuccessful
                               runs

    Return:
       (percentiles, all_sampled_values_sorted)

    Details:
       A single successful running length is computed by adding
       uniformly randomly chosen running lengths until the first time a
       successful one is chosen. In case of no successful run the sum of
       unsuccessful runs is bootstrapped. 

    """
    # TODO: for efficiency reasons a special treatment in the case, 
    #   where all runs are successful and all_sampled_values_sorted is not needed

    Nsucc = len(runlengths_succ)
    Nunsucc = len(runlengths_unsucc)
    
    if Nsucc == 0:
        raise NotImplementedError('this code has been removed as it was not clear whether it makes sense')
        # return (np.Inf*np.array(percentiles), )
        # TODO: the following line does not work because of the use of function sum which interface is different than that of sp or sp1
        return (draw(runlengths_unsucc, percentiles, samplesize=samplesize, 
                     func=sum
                     # func=lambda x: [sum(x)]
                     ), sorted(runlengths_unsucc))
    # if Nunsucc == 0: # Special case: all success, how can we improve efficiency?
    #    return 
    if 11 < 3 and Nunsucc == 0:  # not tested yet: draw each once without replacement and repeat  
        idx = np.random.shuffle(range(Nsucc))
        arrStats = [runlengths_succ[idx[i % Nsucc]] for i in xrange(int(samplesize))]
        arrStats.sort()  # could be avoided
        return (prctile(runlengths_succ, percentiles, issorted=False),
            arrStats)
    if 11 < 3 and Nunsucc == 0:  # not tested yet: bootstraps, but more efficient
        arrStats = [runlengths_succ[np.random.randint(Nsucc)] 
                      for i in xrange(int(samplesize))]
        arrStats.sort()  # could be avoided
        return (prctile(arrStats, percentiles, issorted=True), arrStats)

    # geometric distribution for number of unsuccessful runs
    # The samplesize depends on the number of unsuccessful runs?

    arrStats = []
    sdata = np.array(runlengths_succ)  # more efficient indexing
    udata = np.array(runlengths_unsucc)  # more efficient indexing
    Nu = len(udata)
    Ns = len(sdata)
    # data = np.r_[udata, sdata]
    N = Ns + Nu

    for i in xrange(int(samplesize)):
        # relying that idx<len(data)
        sumdata = 0
        idx = np.random.randint(N)
        # set_trace()
        while idx < Nu:
            sumdata += udata[idx]
            idx = np.random.randint(N)

        sumdata += sdata[idx - Nu]

        arrStats.append(sumdata)  # We know we have one success here.

    arrStats.sort()

    return (prctile(arrStats, percentiles, issorted=True),
            arrStats)

def draw(data, percentiles, samplesize=1e3, func=sp1, args=()):
    """Generates the empirical bootstrap distribution from a sample.

    Input:
      - *data* -- a sequence of data values
      - *percentiles* -- a single scalar value or a sequence of
        percentiles to be computed from the bootstrapped distribution.
      - *func* -- function that computes the statistics as
        func(data,*args) or func(data,*args)[0], by default toolsstats.sp1
      - *args* -- arguments to func, the zero-th element of args is
        expected to be a sequence of boolean giving the success status
        of the associated data value. This specialization of the draw
        procedure is due to the interface of the performance computation
        methods sp1 and sp.
      - *samplesize* -- number of bootstraps drawn, default is 1e3,
        for more reliable values choose rather 1e4. 
        performance is linear in samplesize, 0.2s for samplesize=1000.

    Return:
        (prctiles, all_samplesize_bootstrapped_values_sorted)

    Example:
        >> import toolsstats
        >> data = np.random.randn(22)
        >> res = toolsstats.draw(data, (10,50,90), samplesize=1e4)
        >> print res[0]

    .. note::
       NaN-values are also bootstrapped, but disregarded for the 
       calculation of percentiles which can lead to somewhat
       unexpected results.

    """
    arrStats = []
    N = len(data)
    adata = np.array(data)  # more efficient indexing
    succ = None
    # there is a third argument to func which is the array of success
    if len(args) > 1:
        succ = np.array(args[1])
    # should NaNs also be boostrapped?
    argsv = args
    if 1 < 3:
        for i in xrange(int(samplesize)):
            # relying that idx<len(data)
            idx = np.random.randint(N, size=N)

            # This part is specialized to conform with sp1 and sp.
            if len(args) > 1:
                argsv[1] = succ[np.r_[idx]]

            arrStats.append(func(adata[np.r_[idx]], *(argsv))[0])

            # arrStats = [data[i] for i in idx]  # efficient up to 50 data
    else:  # not more efficient
        arrIdx = np.random.randint(N, size=N * samplesize)
        arrIdx.resize(samplesize, N)
        arrStats = [func(adata[np.r_[idx]], *args) for idx in arrIdx]

    arrStats.sort()

    return (prctile(arrStats, percentiles, issorted=True),
            arrStats)

def prctile(x, arrprctiles, issorted=False, ignore_nan=True):
    """Computes percentile based on data with linear interpolation

    :keyword sequence data: (list, array) of data values
    :keyword prctiles: percentiles to be calculated. Values beyond the 
                       interval [0,100] also return the respective
                       extreme value in data.
    :type prctiles: scalar or sequence
    :keyword issorted: indicate if data is sorted
    :Return:
        sequence of percentile values in data according to argument
        prctiles

    .. note::
        treats np.Inf and -np.Inf and np.NaN, the latter are
        simply disregarded

    """
    if not getattr(arrprctiles, '__iter__', False):  # is not iterable
        arrprctiles = (arrprctiles,)
        # makes a tuple even if the arrprctiles is not iterable
    # remove NaNs, sort

    x = [d for d in x if d is not None and (not np.isnan(d) or not ignore_nan)]
    if not issorted:
        x.sort()

    N = float(len(x))
    if N == 0:
        return [np.NaN for a in arrprctiles]

    res = []
    for p in arrprctiles:
        i = -0.5 + (p / 100.) * N
        ilow = int(np.floor(i))
        ihigh = int(np.ceil(i))
        if i <= 0:
            res += [x[0]]
        elif i >= N - 1:
            res += [x[-1]]
        elif ilow == ihigh:
            res += [x[ilow]]
        # np.bool__.any() works as well as np.array.any()...
        elif np.isinf(x[ihigh]).any() and ihigh - i <= 0.5:
            res += [x[ihigh]]
        elif np.isinf(x[ilow]).any() and i - ilow < 0.5:
            res += [x[ilow]]
        else:
            res += [(ihigh - i) * x[ilow] + (i - ilow) * x[ihigh]]
    return res

def randint(upper, n):
    res = np.floor(upper * np.random.rand(n))
    if any(res >= upper):
        raise Exception, 'np.random.rand returned 1'
    return res

def ranksum_statistic(N1, N2):
    """Returns the U test statistic of the rank-sum (Mann-Whitney-Wilcoxon) test. 

    http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U
    Small sample sizes (direct method).

    """
    # Possible optimization by setting sample 1 to be the one with the smallest
    # rank.

    # TODO: deal with more general type of sorting.
    s1 = sorted(N1)
    s2 = sorted(N2)
    U = 0.
    for i in s1:
        Ui = 0.  # increment of U
        for j in s2:
            if j < i:
                Ui += 1.
            elif j == i:
                Ui += .5
            else:
                break
        # if Ui == 0.:
            # break
        U += Ui
    return U

###############################################################################
# Copyrights from Gary Strangman due to inclusion of his code for the ranksumtest
# method and related.
# Found at: http://www.nmr.mgh.harvard.edu/Neural_Systems_Group/gary/python.html

# Copyright (c) 1999-2007 Gary Strangman; All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

def zprob(z):
    """Returns the area under the normal curve 'to the left of' the given z value.

    http://www.nmr.mgh.harvard.edu/Neural_Systems_Group/gary/python.html

    Thus:

        - for z<0, zprob(z) = 1-tail probability
        - for z>0, 1.0-zprob(z) = 1-tail probability
        - for any z, 2.0*(1.0-zprob(abs(z))) = 2-tail probability

    Adapted from z.c in Gary Perlman's \|Stat.  Can handle multiple dimensions.

    Usage:   azprob(z)    where z is a z-value

    """
    def yfunc(y):
        x = (((((((((((((-0.000045255659 * y
                         + 0.000152529290) * y - 0.000019538132) * y
                       - 0.000676904986) * y + 0.001390604284) * y
                     - 0.000794620820) * y - 0.002034254874) * y
                   + 0.006549791214) * y - 0.010557625006) * y
                 + 0.011630447319) * y - 0.009279453341) * y
               + 0.005353579108) * y - 0.002141268741) * y
             + 0.000535310849) * y + 0.999936657524
        return x

    def wfunc(w):
        x = ((((((((0.000124818987 * w
                    - 0.001075204047) * w + 0.005198775019) * w
                  - 0.019198292004) * w + 0.059054035642) * w
                - 0.151968751364) * w + 0.319152932694) * w
              - 0.531923007300) * w + 0.797884560593) * np.sqrt(w) * 2.0
        return x

    Z_MAX = 6.0  # maximum meaningful z-value
    x = np.zeros(z.shape, np.float_)  # initialize
    y = 0.5 * np.fabs(z)
    x = np.where(np.less(y, 1.0), wfunc(y * y), yfunc(y - 2.0))  # get x's
    x = np.where(np.greater(y, Z_MAX * 0.5), 1.0, x)  # kill those with big Z
    prob = np.where(np.greater(z, 0), (x + 1) * 0.5, (1 - x) * 0.5)
    return prob

def ranksumtest(x, y):
    """Calculates the rank sum statistics for the two input data sets 
    ``x`` and ``y`` and returns z and p. 
    
    This method returns a slight difference compared to scipy.stats.ranksumtest
    in the two-tailed p-value. Should be test drived...

    Returns: z-value for first data set ``x`` and two-tailed p-value
    
    """
    x, y = map(np.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x, y))
    ranked = rankdata(alldata)
    x = ranked[:n1]
    y = ranked[n1:]
    s = np.sum(x, axis=0)
    assert s + np.sum(y, axis=0) == np.sum(range(n1 + n2 + 1))
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    prob = 2 * (1.0 - zprob(abs(z)))
    return z, prob

def rankdata(a):
    """Ranks the data in a, dealing with ties appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    Example:
      In [15]: stats.rankdata([0, 2, 2, 3])
      Out[15]: array([ 1. ,  2.5,  2.5,  4. ])

    Parameters:
      - *a* : array
        This array is first flattened.

    Returns:
      An array of length equal to the size of a, containing rank scores.

    """
    a = np.ravel(a)
    n = len(a)
    svec, ivec = fastsort(a)
    sumranks = 0
    dupcount = 0
    newarray = np.zeros(n, float)
    for i in xrange(n):
        sumranks += i
        dupcount += 1
        if i == n - 1 or svec[i] != svec[i + 1]:
            averank = sumranks / float(dupcount) + 1
            for j in xrange(i - dupcount + 1, i + 1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray

def significancetest(entry0, entry1, targets):
    """Compute the rank-sum test between two data sets.

    For a given target function value, the performances of two
    algorithms are compared. The result of a significance test is
    computed on the number of function evaluations for reaching the
    target or, if not available, the function values for the smallest
    budget in an unsuccessful trial. 
    
    Known bugs: this is not a fair comparison, because the successful 
    trials could be very long.  

    :keyword DataSet entry0: -- data set 0
    :keyword DataSet entry1: -- data set 1
    :keyword list targets: -- list of target function values

    :returns: list of (z, p) for each target function values in
              input argument targets. z and p are values returned by the
              ranksumtest method.

    """

    bootstraps = False  # future extension
    res = []
    evals = []
    bestalgs = []
    isBestAlg = False
    # one of the entry is an instance of BestAlgDataSet
    for entry in (entry0, entry1):
        tmp = entry.detEvals(targets)
        if not entry.__dict__.has_key('funvals'):  # this looks like a terrible hack
            isBestAlg = True
            # for i, j in enumerate(tmp[0]):
                # if np.isnan(j).all():
                    # tmp[0][i] = np.array([np.nan]*len(entry.bestfinalfunvals))
            # Make sure that the length of elements of tmp[0] is the same as
            # that of the associated function values
            evals.append(tmp[0])
            bestalgs.append(tmp[1])
        else:
            evals.append(tmp)
            bestalgs.append(None)
            
    if not isBestAlg:
        erts = [None, None]
        erts[0] = entry0.detERT(targets)
        erts[1] = entry1.detERT(targets)
        averageevals = [None, None]
        averageevals[0] = entry0.detAverageEvals(targets)
        averageevals[1] = entry1.detAverageEvals(targets)
        if bootstraps: 
            psucc0 = 1 - sum(np.isnan(entry0.getEvals(targets)), axis= -1) / entry0.nbRuns()
            psucc1 = None
            if psucc0 == 1 and psucc1 == 1:
                bootstraps = False

    for i in range(len(targets)):
        # 1. Determine FE_umin,  the minimum evals in unsuccessful trials 
        FE_umin = np.inf

        # if there is at least one unsuccessful run
        if (np.isnan(evals[0][i]).any() or np.isnan(evals[1][i]).any()):
            fvalues = []
            if isBestAlg:
                for j, entry in enumerate((entry0, entry1)):
                    # if best alg entry
                    if isinstance(entry.finalfunvals, dict):
                        alg = bestalgs[j][i]
                        if alg is None:
                            tmpfvalues = entry.bestfinalfunvals
                        else:
                            tmpfvalues = entry.finalfunvals[alg]
                    else:
                        unsucc = np.isnan(evals[j][i])
                        if unsucc.any():
                            FE_umin = min(entry.maxevals[unsucc])
                        else:
                            FE_umin = np.inf
                        # Determine the function values for FE_umin
                        tmpfvalues = np.array([np.inf] * entry.nbRuns())
                        for curline in entry.funvals:
                            # only works because the funvals are monotonous
                            if curline[0] > FE_umin:
                                break
                            prevline = curline[1:]
                        tmpfvalues = prevline.copy()
                        # tmpfvalues = entry.finalfunvals
                        # if (tmpfvalues != entry.finalfunvals).any():
                            # set_trace()
                    fvalues.append(tmpfvalues)
            else:
                # 1) find min_{both algorithms}(conducted FEvals in
                # unsuccessful trials) =: FE_umin
                FE_umin = np.inf
                if np.isnan(evals[0][i]).any() or np.isnan(evals[1][i]).any():
                    FE = []
                    for j, entry in enumerate((entry0, entry1)):
                        unsucc = np.isnan(evals[j][i])
                        if unsucc.any():
                            tmpfe = min(entry.maxevals[unsucc])
                        else:
                            tmpfe = np.inf
                        FE.append(tmpfe)
                    FE_umin = min(FE)

                    # Determine the function values for FE_umin
                    fvalues = []
                    for j, entry in enumerate((entry0, entry1)):
                        prevline = np.array([np.inf] * entry.nbRuns())
                        for curline in entry.funvals:
                            # only works because the funvals are monotonous
                            if curline[0] > FE_umin:
                                break
                            prevline = curline[1:]
                        fvalues.append(prevline)

        # 2. 3. 4. Collect data for the significance test:
        curdata = []  # current data 
        for j, entry in enumerate((entry0, entry1)):
            tmp = evals[j][i].copy()
            idx = np.isnan(tmp) + (tmp > FE_umin)
            tmp = np.power(tmp, -1.)
            if idx.any():
                tmp[idx] = -fvalues[j][idx]  # larger data is better
            curdata.append(tmp)

        z_and_p = ranksumtest(curdata[0], curdata[1])
        if isBestAlg:
            z_and_p = list(z_and_p)  # no idea what that is for
            z_and_p[1] /= 2.  # one-tailed p-value instead of two-tailed
        else:  # possibly correct 
            ibetter = 0 if z_and_p[0] > 0 else 1  # larger data is better
            iworse = 1 - ibetter
            if not (erts[ibetter][i] <= erts[iworse][i] and  # inf are equal
                (erts[ibetter][i] is np.inf or  # comparable data: only f-values are compared for significance (they are compared for the same #evals)
                 averageevals[ibetter][i] < averageevals[iworse][i])):  # better algorithm must not have larger effort, should this take into account FE_umin?
                z_and_p = (z_and_p[0], 1.0)                  

        res.append(z_and_p)

    return res

def significance_all_best_vs_other(datasets, targets, best_alg_idx=None):
    """:param datasets: is a list of DataSet from different algorithms, otherwise on the same function and dimension (which is not necessarily checked)
    :param targets: is a list of target values, 
    :param best_alg_idx:  for each target the best algorithm to be tested against the others 
    
    returns a list of ``(z, p)`` tuples, each is the result for the ranksumtest 
    for the respective target value in targets and the index list of best algorithm. 
    
    """ 
    if best_alg_idx is None:
        erts = []
        for ds in datasets:
            erts.append(ds.detERT(targets))
        best_alg_idx = np.array(erts).argsort(0)[0, :]  # indexed by target index
        assert len(best_alg_idx) == len(targets)
    elif 1 < 3:  # only for debugging
        erts = []
        for ds in datasets:
            erts.append(ds.detERT(targets))
        best_alg_idx2 = np.array(erts).argsort(0)[0, :]  # indexed by target index
        assert all(best_alg_idx2 == best_alg_idx)
        
    # significance test of best given algorithm against all others
    significance_versus_others = []  # indexed by target index
    assert len(best_alg_idx) == len(targets)
    if len(datasets) > 1:
        for itarget, target in enumerate(targets):
            z_and_p = (0, 0)
            for jalg in xrange(len(datasets)):
                if jalg == best_alg_idx[itarget]:
                    continue
                z_and_p2 = significancetest(datasets[jalg], datasets[best_alg_idx[itarget]], [target])[0]
                if z_and_p2[1] > z_and_p[1]:  # look for strongest opponent, ie weakest p
                    z_and_p = z_and_p2 
            significance_versus_others.append(z_and_p)
    return significance_versus_others, best_alg_idx

def fastsort(a):
    # fixme: the wording in the docstring is nonsense.
    """Sort an array and provide the argsort.

    Parameters:
      *a* : array

    Returns:
      (sorted array, indices into the original array)

    """
    it = np.argsort(a)
    as_ = a[it]
    return as_, it

def sliding_window_data(data, width=2, operator=np.median,
                        number_of_stats=0, only_finite_data=True):
    """width is an absolute number, the resulting data has
    the same length as the original data and the window width
    is between width/2 at the border and width in the middle.

    """
    if width < 2:
        return (data, [])
    if width >= len(data):
        warnings.warn('sliding window width %d should be smaller than ' +
            'the number of data %d' % int(width), len(data))
    down = width // 2
    up = width // 2 + (width % 2)
    d = np.array(data, copy=False)
    smoothened_data = []
    stats = []
    stats_mod = len(d) // number_of_stats
    i_last_stats = 0
    next = 0.1 + 1.8 * np.random.rand()
    for i in xrange(len(d)):
        current_data = d[max((i - down, 0)) : min((i + up, len(d)))]
        if only_finite_data:
            if np.isfinite(d[i]):
                idx = np.isfinite(current_data)
                smoothened_data.append(operator(current_data[idx]))
            else:
                smoothened_data.append(d[i])
        else:
            smoothened_data.append(operator(current_data))
        if i_last_stats > next * stats_mod:
            stats.append([i, prctile(
                current_data[np.isfinite(current_data)]
                    if only_finite_data else current_data,
                [2, 10, 25, 50, 75, 90, 98])])
            i_last_stats = 0
            next = 0.1 + 1.8 * np.random.rand()
        i_last_stats += 1

    return (np.array(smoothened_data, copy=False)
        if isinstance(data, np.ndarray) else smoothened_data, stats)


###############################################################################
