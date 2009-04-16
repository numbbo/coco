# bootstrapping

import numpy

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


def draw(data, percentiles, samplesize=1e3, func=sp1, args=()):
    """Input:
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
    # there is a third argument to func which is the arrayof success
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

