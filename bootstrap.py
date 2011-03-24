#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Bootstrapping and statistics routines."""

import numpy
from pdb import set_trace

def sp1(data, maxvalue=numpy.Inf, issuccessful=None):
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
      by the success rate. SP1 equals numpy.Inf when the success
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
                        if not numpy.isnan(data[i])]
    dat = [d for d in data if not numpy.isnan(d)]
    N = len(dat)

    if N == 0:
        return(numpy.nan, numpy.nan, numpy.nan)

    # remove unsuccessful data
    if issuccessful is not None:
        dat = [dat[i] for i in xrange(len(dat)) if issuccessful[i]]
    else:
        dat = [d for d in dat if d < maxvalue]
    succ = float(len(dat)) / N

    # return
    if succ == 0:
        return (numpy.Inf, 0., 0)
    else:
        return (numpy.mean(dat) / succ, succ, len(dat))

def sp(data, maxvalue=numpy.Inf, issuccessful=None, allowinf=True):
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

    #TODO allowinf is obsolete

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
                        if not numpy.isnan(data[i])]
    dat = [d for d in data if not numpy.isnan(d)]
    N = len(dat)

    if N == 0:
        return(numpy.nan, numpy.nan, numpy.nan)

    # remove unsuccessful data
    if issuccessful is not None:
        succdat = [dat[i] for i in xrange(len(dat)) if issuccessful[i]]
    else:
        succdat = [d for d in dat if d < maxvalue]
    succ = float(len(succdat)) / N

    # return
    if succ == 0:
        if not allowinf:
            res = numpy.sum(dat) # here it is divided by min(1, succ)
        else:
            res = numpy.inf
    else:
        res = numpy.sum(dat) / float(len(succdat))

    return (res, succ, len(succdat))


def drawSP(runlengths_succ, runlengths_unsucc, percentiles, samplesize=1e3):
    """Returns the percentiles of the bootstrap distribution of 'simulated'  
       running lengths of successful runs.
    Input:
      runlengths_succ--array of running lengths of successful runs
      runlengths_unsucc--array of running lengths of unsuccessful runs
    Return:
       (percentiles, all_sampled_values_sorted)
    Details:
       A single successful running length is computed by adding uniformly
       randomly chosen running lengths until the first time a successful one is
       chosen. In case of no successful run the sum of unsuccessful runs is
       bootstrapped. 
    """

    # TODO: for efficiency reasons a special treatment in the case, 
    #   where all runs are successful and all_sampled_values_sorted is not needed

    Nsucc = len(runlengths_succ)
    Nunsucc = len(runlengths_unsucc)

    if Nsucc == 0:
        # return (numpy.Inf*numpy.array(percentiles), )
        #TODO: the following line does not work because of the use of function sum which interface is different than that of sp or sp1
        return (draw(runlengths_unsucc, percentiles, samplesize=samplesize, func=sum), sorted(runlengths_unsucc))

    #if Nunsucc == 0: # Special case: all success, how can we improve efficiency?
    #    return 
    if 11 < 3 and Nunsucc == 0:  # not tested yet: draw each once without replacement and repeat  
        idx = numpy.random.shuffle(range(Nsucc))
        arrStats = [runlengths_succ[idx[i % Nsucc]] for i in xrange(int(samplesize))]
        arrStats.sort()  # could be avoided
        return (prctile(runlengths_succ, percentiles, issorted=False),
            arrStats)
    if 11 < 3 and Nunsucc == 0:  # not tested yet: bootstrap, but more efficient
        arrStats = [runlengths_succ[numpy.random.randint(Nsucc)] 
                      for i in xrange(int(samplesize))]
        arrStats.sort()  # could be avoided
        return (prctile(arrStats, percentiles, issorted=True), arrStats)

    # geometric distribution for number of unsuccessful runs
    # The samplesize depends on the number of unsuccessful runs?

    arrStats = []
    sdata = numpy.array(runlengths_succ)  # more efficient indexing
    udata = numpy.array(runlengths_unsucc)  # more efficient indexing
    Nu = len(udata)
    Ns = len(sdata)
    #data = numpy.r_[udata, sdata]
    N = Ns + Nu

    for i in xrange(int(samplesize)):
        # relying that idx<len(data)
        sumdata = 0
        idx = numpy.random.randint(N)
        #set_trace()
        while idx < Nu:
            sumdata += udata[idx]
            idx = numpy.random.randint(N)

        sumdata += sdata[idx-Nu]

        arrStats.append(sumdata) # We know we have one success here.

    arrStats.sort()

    return (prctile(arrStats, percentiles, issorted=True),
            arrStats)

def draw(data, percentiles, samplesize=1e3, func=sp1, args=()):
    """Generates the empirical bootstrap distribution from a sample.

    Input:
    data--a sequence of data values
    percentiles--a single scalar value or a sequence of percentiles
        to be computed from the bootstrapped distribution.
    func--function that computes the statistics as func(data,*args)
        or func(data,*args)[0], by default bootstrap.sp1
    args--arguments to func, the zero-th element of args is expected to be a
        sequence of boolean giving the success status of the associated data
        value. This specialization of the draw procedure is due to the
        interface of the performance computation methods sp1 and sp.
    samplesize--number of bootstraps drawn, default is 1e3,
       for more reliable values choose rather 1e4. 
       performance is linear in samplesize, 0.2s for samplesize=1000.
    Return:
        (prctiles, all_samplesize_bootstrapped_values_sorted)
    Example:
        import bootstrap
        data = numpy.random.randn(22)
        res = bootstrap.draw(data, (10,50,90), samplesize=1e4)
        print res[0]
    Remark:
       NaN-values are also bootstrapped, but disregarded for the 
       calculation of percentiles which can lead to somewhat
       unexpected results.
    """

    arrStats = []
    N = len(data)
    adata = numpy.array(data)  # more efficient indexing
    succ = None
    # there is a third argument to func which is the array of success
    if len(args) > 1:
        succ = numpy.array(args[1])
    # should NaNs also be bootstrapped?
    argsv = args
    if 1 < 3:
        for i in xrange(int(samplesize)):
            # relying that idx<len(data)
            idx = numpy.random.randint(N, size=N)

            #This part is specialized to conform with sp1 and sp.
            if len(args) > 1:
                argsv[1] = succ[numpy.r_[idx]]

            arrStats.append(func(adata[numpy.r_[idx]], *(argsv))[0])

            # arrStats = [data[i] for i in idx]  # efficient up to 50 data
    else:  # not more efficient
        arrIdx = numpy.random.randint(N, size=N*samplesize)
        arrIdx.resize(samplesize, N)
        arrStats = [func(adata[numpy.r_[idx]], *args) for idx in arrIdx]

    arrStats.sort()

    return (prctile(arrStats, percentiles, issorted=True),
            arrStats)

# utils not really part of bootstrap module though:
def prctile(x, arrprctiles, issorted=False):
    """prctile -- computes percentile based on data with linear interpolation
    :Calling Sequence:
        prctile(data, prctiles, issorted=False)
    :Arguments:
        data -- a sequence (list, array) of data values
        prctiles -- a scalar or a sequence of pertentiles 
            to be calculated. Values beyond the interval [0,100]
            also return the respective extreme value in data. 
        issorted -- indicate if data is sorted
    :Return:
        sequence of percentile values in data according to argument
        prctiles
    :Remark:
        treats numpy.Inf and -numpy.Inf and numpy.NaN, the latter are
        simply disregarded
    """

    if not getattr(arrprctiles, '__iter__', False):  # is not iterable
        arrprctiles = (arrprctiles,)
        # makes a tuple even if the arrprctiles is not iterable
    # remove NaNs, sort

    x = [d for d in x if not numpy.isnan(d) and d is not None]
    if not issorted:
        x.sort()

    N = float(len(x))
    if N == 0:
        return [numpy.NaN for a in arrprctiles]

    res = []
    for p in arrprctiles:
        i = -0.5 + (p/100.) * N
        ilow = int(numpy.floor(i))
        ihigh = int(numpy.ceil(i))
        if i <= 0:
            res += [x[0]]
        elif i >= N-1:
            res += [x[-1]]
        elif ilow == ihigh:
            res += [x[ilow]]
        #numpy.bool__.any() works as well as numpy.array.any()...
        elif numpy.isinf(x[ihigh]).any() and ihigh - i <= 0.5:
            res += [x[ihigh]]
        elif numpy.isinf(x[ilow]).any() and i - ilow < 0.5:
            res += [x[ilow]]
        else:
            res += [(ihigh-i) * x[ilow] + (i-ilow) * x[ihigh]]
    return res

def randint(upper, n):
    res = numpy.floor(upper*numpy.random.rand(n))
    if any(res>=upper):
        raise Exception, 'numpy.random.rand returned 1'
    return res

def ranksumtest(N1, N2):
    """Custom rank-sum (Mann-Whitney-Wilcoxon) test
    http://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U
    Small sample sizes (direct method).
    Keyword arguments:
    N1    sample 1
    N2    sample 2
    """

    # Possible optimization by setting sample 1 to be the one with the smallest
    # rank.

    #TODO: deal with more general type of sorting.
    s1 = sorted(N1)
    s2 = sorted(N2)
    U = 0.
    for i in s1:
        Ui = 0. # increment of U
        for j in s2:
            if j < i:
                Ui += 1.
            elif j == i:
                Ui += .5
            else:
                break
        #if Ui == 0.:
            #break
        U += Ui
    return U

###############################################################################
# Copyrights from Gary Strangman due to inclusion of his code for the ranksums
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
    Thus, 
        for z<0, zprob(z) = 1-tail probability
        for z>0, 1.0-zprob(z) = 1-tail probability
        for any z, 2.0*(1.0-zprob(abs(z))) = 2-tail probability
    Adapted from z.c in Gary Perlman's |Stat.  Can handle multiple dimensions.

    Usage:   azprob(z)    where z is a z-value
    """
    def yfunc(y):
        x = (((((((((((((-0.000045255659 * y
                         +0.000152529290) * y -0.000019538132) * y
                       -0.000676904986) * y +0.001390604284) * y
                     -0.000794620820) * y -0.002034254874) * y
                   +0.006549791214) * y -0.010557625006) * y
                 +0.011630447319) * y -0.009279453341) * y
               +0.005353579108) * y -0.002141268741) * y
             +0.000535310849) * y +0.999936657524
        return x

    def wfunc(w):
        x = ((((((((0.000124818987 * w
                    -0.001075204047) * w +0.005198775019) * w
                  -0.019198292004) * w +0.059054035642) * w
                -0.151968751364) * w +0.319152932694) * w
              -0.531923007300) * w +0.797884560593) * numpy.sqrt(w) * 2.0
        return x

    Z_MAX = 6.0    # maximum meaningful z-value
    x = numpy.zeros(z.shape, numpy.float_) # initialize
    y = 0.5 * numpy.fabs(z)
    x = numpy.where(numpy.less(y,1.0),wfunc(y*y),yfunc(y-2.0)) # get x's
    x = numpy.where(numpy.greater(y,Z_MAX*0.5),1.0,x)          # kill those with big Z
    prob = numpy.where(numpy.greater(z,0),(x+1)*0.5,(1-x)*0.5)
    return prob

def ranksums(x, y):
    """Calculates the rank sums statistic on the provided scores and
    returns the result.
    This method returns a slight difference compared to scipy.stats.ranksums
    in the two-tailed p-value. Should be test drived...

    Returns: z-statistic, two-tailed p-value
    """
    x,y = map(numpy.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = numpy.concatenate((x,y))
    ranked = rankdata(alldata)
    x = ranked[:n1]
    y = ranked[n1:]
    s = numpy.sum(x,axis=0)
    assert s + numpy.sum(y,axis=0) == numpy.sum(range(n1+n2+1))
    expected = n1*(n1+n2+1) / 2.0
    z = (s - expected) / numpy.sqrt(n1*n2*(n1+n2+1)/12.0)
    prob = 2*(1.0 -zprob(abs(z)))
    return z, prob

def rankdata(a):
    """Ranks the data in a, dealing with ties appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    Example
    -------
    In [15]: stats.rankdata([0, 2, 2, 3])
    Out[15]: array([ 1. ,  2.5,  2.5,  4. ])

    Parameters
    ----------
    a : array
        This array is first flattened.

    Returns
    -------
    An array of length equal to the size of a, containing rank scores.
    """
    a = numpy.ravel(a)
    n = len(a)
    svec, ivec = fastsort(a)
    sumranks = 0
    dupcount = 0
    newarray = numpy.zeros(n, float)
    for i in xrange(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in xrange(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray

def fastsort(a):
    # fixme: the wording in the docstring is nonsense.
    """Sort an array and provide the argsort.

    Parameters
    ----------
    a : array

    Returns
    -------
    (sorted array,
     indices into the original array,
    )
    """
    it = numpy.argsort(a)
    as_ = a[it]
    return as_, it

###############################################################################
