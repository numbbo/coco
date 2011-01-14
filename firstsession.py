#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""First session script."""

# grep '^>>>\|^\.\.\.' firstsession.tex |sed -e 's/^.\{4\}//'

import bbob_pproc as bb

ds = bb.load('exampledata/alg0_rawdata/bbobexp_f2.info')
print ds

ds = bb.load('exampledata/alg0_pickledata/ppdata_f002_20.pickle')
ds = bb.load('exampledata/alg0_rawdata')

import glob
ds = bb.load(glob.glob('exampledata/alg0_pickledata/ppdata_f002_*.pickle'))

ds = bb.load('exampledata/alg0_pickledata/ppdata_f002_20.pickle')
bb.info(ds) # display information on DataSetList ds
d = ds[0] # store the first element of ds in d for convenience
print d.funvals
budgets = d.funvals[:, 0] # stores first column in budgets
funvals = d.funvals[:, 1:] # stores all other columns in funvals

nbrows, nbruns = funvals.shape
import matplotlib.pyplot as plt
plt.ion() # interactive mode is now on
for i in range(0, nbruns):
    plt.loglog(budgets, funvals[:, i])
plt.grid()
plt.xlabel('Budgets')
plt.ylabel('Best Function Values')

import numpy as np
plt.loglog(budgets, np.median(funvals, axis=1), linewidth=3, color='r',
           label='median')
plt.legend() # display legend
ds1 = bb.load('exampledata/alg1_pickledata/ppdata_f002_20.pickle')
print ds1
d1 = ds1[0]
budgets1 = d1.funvals[:, 0]
funvals1 = d1.funvals[:, 1:]
for i in range(0, funvals1.shape[1]):
    plt.loglog(budgets1, funvals1[:, i], linestyle='--')
plt.loglog(budgets1, np.median(funvals1, axis=1), linewidth=3, color='g',
           label='median alg1')
plt.legend() # updates legend
plt.savefig('examplefigure')  # save active figure as image

plt.figure() # open a new figure
from bbob_pproc.bootstrap import prctile
q = np.array(list(prctile(i, [25, 50, 75]) for i in funvals))
ymed = q[:, 1]
ylow = ymed - q[:, 0]
yhig = q[:, 2] - ymed
yerr = np.vstack((ylow, yhig))
plt.errorbar(budgets, ymed, yerr, color='r', label='alg0')
plt.xscale('log')
plt.yscale('log')
plt.grid()
q1 = np.array(list(prctile(i, [25, 50, 75]) for i in funvals1))
ymed1 = q1[:, 1]
yerr1 = np.vstack((ymed1 - q1[:, 0], q1[:, 2] - ymed1))
plt.errorbar(budgets1, ymed1, yerr1, color='g', label='alg1')
plt.legend()

targets = d.evals[:, 0]
evals =  d.evals[:, 1:]
nbrows, nbruns = evals.shape
plt.figure()
for i in range(0, nbruns):
    plt.loglog(targets, evals[:, i])
plt.grid()
plt.xlabel('Targets')
plt.ylabel('Function Evaluations')
plt.loglog(d.target[d.target>=1e-8], d.ert[d.target>=1e-8], lw=3,
           color='r', label='ert')
plt.legend()

plt.figure()
for i in range(0, nbruns):
    plt.loglog(evals[:, i], targets)
plt.grid()
plt.xlabel('Function Evaluations')
plt.ylabel('Targets')
plt.loglog(d.ert[d.target>=1e-8], d.target[d.target>=1e-8], lw=3,
           color='r', label='ert')
plt.legend()

from bbob_pproc import pprldistr
ds = bb.load(glob.glob('exampledata/alg0_pickledata/ppdata_f0*_20.pickle'))
plt.figure()
pprldistr.plot(ds)
pprldistr.beautify() # resize the window to view whole figure

from bbob_pproc.compall import ppperfprof
ds = bb.load(glob.glob('exampledata/alg0_pickledata/ppdata_f0*_20.pickle'))
plt.figure()
ppperfprof.plot(ds)
ppperfprof.beautify()

from bbob_pproc import ppfigdim
ds = bb.load(glob.glob('exampledata/alg0_pickledata/ppdata_f002_*.pickle'))
plt.figure()
ppperfprof.plot(ds)
ppperfprof.beautify()
