#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Stats including rank-sum test for small sample and correct tied ranks.
Author: Sturla Molden
http://mail.scipy.org/pipermail/scipy-user/2009-February/019759.html
"""

import numpy as np

import scipy
import scipy.special
zprob = scipy.special.ndtr

def ranksumtest(x, y):
     """
     Wilcoxon rank sum test

     Returns:
	W-statistic
	Z-statistic
	one-tailed p-value, asymptotic approximation
	one-tailed p-value, Monte Carlo approximation

     Corrected for ties.
     """

     x,y = map(np.asarray, (x, y))
     n1 = len(x)
     n2 = len(y)
     alldata = np.concatenate((x,y))
     ranked = rankdata(alldata)
     x = ranked[:n1]
     y = ranked[n1:]
     w = np.sum(x,axis=0)

     def montecarlo():
         shuffle = np.random.shuffle
         a = np.zeros(1000)
         shuffle(ranked) # bug in numpy: the first shuffle doesn't work
         for i in xrange(1000):
             shuffle(ranked)
             a[i] = np.sum(ranked[:n1],axis=0)
         return np.sum(a >= w) / 1000.0

     def aymptotic_p():
         expected = n1*(n1+n2+1) / 2.0
         z = (w - expected) / np.sqrt(n1*n2*(n1+n2+1)/12.0)
         return 1.0 - zprob(z), z

     def aymptotic_p_ties():
         t = []
         _t = 0
         for r in ranked:
             if r % 1:
                 _t += 1
             else:
                 if _t:
                     t.append(_t)
                     _t = 0
         if _t: t.append(_t)
         t = np.asarray(t)
         expected = n1*(n1+n2+1) / 2.0
         tcorr = np.sum((t-1)*t*(t+1))/float((n1+n2)*(n1+n2-1))
         z = (w - expected) / np.sqrt(n1*n2*(n1+n2+1-tcorr)/12.0)
         return 1.0 - zprob(z), z

     p_mc = montecarlo()
     if np.any(ranked % 1):
         p, z = aymptotic_p_ties()
     else:
         p, z = aymptotic_p()
     return w, z, p, p_mc



def rankdata(a):
     a = np.ravel(a)
     n = len(a)
     svec, ivec = fastsort(a)
     sumranks = 0
     dupcount = 0
     newarray = np.zeros(n, float)
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
     it = np.argsort(a)
     as_ = a[it]
     return as_, it
