#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""First session script.

Folders BIPOP-CMA-ES_hansen_noiseless, BIPOP-CMA-ES, NEWUOA need to be
in the current working directory.

The corresponding archives of these folders can be found at
http://coco.lri.fr/BBOB2009/

"""

# grep '^>>>\|^\.\.\.' firstsession.tex |sed -e 's/^.\{4\}//'

from pylab import *
ion() # may be needed for figures to be shown when executing the script

import bbob_pproc as bb

ds = bb.load('BIPOP-CMA-ES_hansen_noiseless/bbobexp_f2.info')
print ds

ds = bb.load('BIPOP-CMA-ES/ppdata_f002_20.pickle')
ds = bb.load('BIPOP-CMA-ES_hansen_noiseless')

import glob
ds = bb.load(glob.glob('BIPOP-CMA-ES/ppdata_f002_*.pickle'))

ds = bb.load('BIPOP-CMA-ES/ppdata_f002_20.pickle')
bb.info(ds) # display information on DataSetList ds
d = ds[0] # store the first element of ds in d for convenience
print d.funvals
budgets = d.funvals[:, 0] # stores first column in budgets
funvals = d.funvals[:, 1:] # stores all other columns in funvals

nbrows, nbruns = funvals.shape
for i in range(0, nbruns):
    loglog(budgets, funvals[:, i])
grid()
xlabel('Budgets')
ylabel('Best Function Values')

loglog(budgets, median(funvals, axis=1), linewidth=3, color='r',
       label='median')
legend() # display legend
ds1 = bb.load('NEWUOA/ppdata_f002_20.pickle')
print ds1
d1 = ds1[0]
budgets1 = d1.funvals[:, 0]
funvals1 = d1.funvals[:, 1:]
for i in range(0, funvals1.shape[1]):
    loglog(budgets1, funvals1[:, i], linestyle='--')
loglog(budgets1, median(funvals1, axis=1), linewidth=3, color='g',
       label='median NEWUOA')
legend() # updates legend
savefig('examplefigure')  # save active figure as image

targets = d.evals[:, 0]
evals =  d.evals[:, 1:]
nbrows, nbruns = evals.shape
figure()
for i in range(0, nbruns):
    loglog(targets, evals[:, i])
grid()
xlabel('Targets')
ylabel('Function Evaluations')
loglog(d.target[d.target>=1e-8], d.ert[d.target>=1e-8], lw=3,
       color='r', label='ert')
legend()

figure()
for i in range(0, nbruns):
    loglog(evals[:, i], targets)
grid()
xlabel('Function Evaluations')
ylabel('Targets')
loglog(d.ert[d.target>=1e-8], d.target[d.target>=1e-8], lw=3,
       color='r', label='ert')
legend()

figure() # open a new figure
from bbob_pproc.bootstrap import prctile
q = array(list(prctile(i, [25, 50, 75]) for i in evals))
xmed = q[:, 1]
xlow = xmed - q[:, 0]
xhig = q[:, 2] - xmed
xerr = vstack((xlow, xhig))
errorbar(xmed, targets, xerr=xerr, color='r', label='Median')
xscale('log')
yscale('log')
xlabel('Function Evaluations')
ylabel('Targets')
grid()
legend()

from bbob_pproc import pprldistr
ds = bb.load(glob.glob('BIPOP-CMA-ES/ppdata_f0*_20.pickle'))
figure()
pprldistr.plot(ds)
pprldistr.beautify() # resize the window to view whole figure

from bbob_pproc.compall import ppperfprof
ds = bb.load(glob.glob('BIPOP-CMA-ES/ppdata_f0*_20.pickle'))
figure()
ppperfprof.plot(ds)
ppperfprof.beautify()

from bbob_pproc import ppfigdim
ds = bb.load(glob.glob('BIPOP-CMA-ES/ppdata_f002_*.pickle'))
figure()
ppperfprof.plot(ds)
ppperfprof.beautify()

