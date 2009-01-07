# bootstrapping 
# this is linked in directory BBOB/code/python/postProcessing and will
# be copied if something has to change to include it in the
# postProcessing package.
# this should move to another directory in future 

# BUG:
# elif ilow == ihigh:
#     res += [x[i]]
# i is not an integer


import numpy

def sp1(data, maxevals=numpy.Inf):
    """sp1(data, maxevals=Inf) computes the mean over
    successful entries in data divided by the success
    rate (SP1). Successful entries in data are truly smaller than
    maxevals. Input data contains, e.g., number of function
    evaluations to reach the target value.
    Returns (SP1, success_rate), where SP1 equals numpy.Inf
        when the success rate is zero. 
    """

    dat = [d for d in data if not numpy.isnan(d)]
    N = len(dat)
    dat = [d for d in dat if d < maxevals]
    succ = float(len(dat)) / N
    if succ == 0:
        return (numpy.Inf, 0., 0)
    else:
        return (numpy.mean(dat) / succ, succ, len(dat))

def draw(data, percentiles, samplesize=1e3, func=sp1, args=()):
    """Input:
    data--a sequence of data values
    percentiles--a single scalar value or a sequence of percentiles
        to be computed from the bootstrapped distribution.
    func--function that computes the statistics as func(data,*args)
        or func(data,*args)[0], by default bootstrap.sp1
    args--arguments to func
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
    # should NaNs also be bootstrapped? 
    if 1 < 3:
        for i in xrange(int(samplesize)):
            # relying that idx<len(data) 
            idx = numpy.random.randint(N, size=N) 
            res = func(adata[numpy.r_[idx]], *args)
            if isinstance(res, list) or isinstance(res, tuple):
                res = res[0]
            arrStats.append(res)
            # arrStats = [data[i] for i in idx]  # efficient up to 50 data
    else:  # not more efficient
        arrIdx = numpy.random.randint(N, size=N*samplesize)
        arrIdx.resize(samplesize, N)
        arrStats = [func(adata[numpy.r_[idx]], *args)[0] for idx in arrIdx]

    arrStats = sorted(arrStats)
    return (prctile(arrStats, percentiles, issorted=True),
            arrStats)


# utils not really part of bootstrap module though:
def prctile(x, arrprctiles, issorted=False):
    """prctile -- computes percentile based on data with linear
    interpolation
    :Calling Sequence:
        prctile(data, prctiles, issorted=False)
    :Arguments:
        data -- a sequence (list, array) of data values
        prctiles -- a scalar or a sequence of pertentiles 
            to be calcuated. Values beyond the interval [0,100]
            also return the respective extreme value in data. 
        issorted -- indicate when data is sorted
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
        x = numpy.sort(x)
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
        elif numpy.isinf(x[ihigh]) and ihigh - i <= 0.5:
            res += [x[ihigh]]
        elif numpy.isinf(x[ilow]) and i - ilow < 0.5:
            res += [x[ilow]]
        else:
            res += [(ihigh-i) * x[ilow] + (i-ilow) * x[ihigh]]
    return res

def randint(upper, n):
    res = numpy.floor(upper*numpy.random.rand(n))
    if any(res>=upper):
        raise Error('numpy.random.rand returned 1')
    return res

