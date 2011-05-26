#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""First session script.

Folders:

 - BBOB2009rawdata/BIPOP-CMA-ES_hansen_noiseless,
 - BBOB2009pythondata/BIPOP-CMA-ES,
 - BBOB2009pythondata/NEWUOA

need to be in the current working directory. Otherwise, data will be
collected automatically online.

The corresponding archives of these folders can be found at
http://coco.lri.fr/BBOB2009/

"""

# grep '^>>>\|^\.\.\.' firstsession.tex |sed -e 's/^.\{4\}//'

import urllib
import tarfile
from pylab import *
ion() # may be needed for figures to be shown when executing the script

import bbob_pproc as bb

# Collect and unarchive data (~20MB)
dataurl = 'http://coco.lri.fr/BBOB2009/rawdata/BIPOP-CMA-ES_hansen_noiseless.tar.gz'
filename, headers = urllib.urlretrieve(dataurl)
archivefile = tarfile.open(filename)
archivefile.extractall()

# Display some information
ds = bb.load('BBOB2009rawdata/BIPOP-CMA-ES_hansen_noiseless/bbobexp_f2.info')
print ds

# Collect and unarchive data (3.4MB)
dataurl = 'http://coco.lri.fr/BBOB2009/pythondata/BIPOP-CMA-ES.tar.gz'
filename, headers = urllib.urlretrieve(dataurl)
archivefile = tarfile.open(filename)
archivefile.extractall()

# Load a pickle file
ds = bb.load('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f002_20.pickle')
# Load a folder
ds = bb.load('BBOB2009rawdata/BIPOP-CMA-ES_hansen_noiseless')

# Load data using a wildcard
import glob
ds = bb.load(glob.glob('BIPOP-CMA-ES/ppdata_f002_*.pickle'))

# Display function values versus time
ds = bb.load('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f002_20.pickle')
bb.info(ds) # display information on DataSetList ds
d = ds[0] # store the first element of ds in d for convenience
print d.funvals
budgets = d.funvals[:, 0] # stores first column in budgets
funvals = d.funvals[:, 1:] # stores all other columns in funvals

# Plot function values versus time
nbrows, nbruns = funvals.shape
for i in range(0, nbruns):
    loglog(budgets, funvals[:, i])
grid()
xlabel('Budgets')
ylabel('Best Function Values')

# Plot median function values versus time
loglog(budgets, median(funvals, axis=1), linewidth=3, color='r',
       label='median')
legend() # display legend

# Add another data set
dataurl = 'http://coco.lri.fr/BBOB2009/pythondata/NEWUOA.tar.gz'
filename, headers = urllib.urlretrieve(dataurl)
archivefile = tarfile.open(filename)
archivefile.extractall()

ds1 = bb.load('BBOB2009pythondata/NEWUOA/ppdata_f002_20.pickle')
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

# Plot function evaluations versus target precision
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
gca().invert_xaxis() # xaxis from the easiest to the hardest
legend()  # this operation updates the figure with the inverse axis.

# Plot target precision versus function evaluations
# (swap x-y of previous figure)
figure()
for i in range(0, nbruns):
    loglog(evals[:, i], targets)
grid()
xlabel('Function Evaluations')
ylabel('Targets')
loglog(d.ert[d.target>=1e-8], d.target[d.target>=1e-8], lw=3,
       color='r', label='ert')
legend()

# Plot target precision versus function evaluations with error bars
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

# Empirical cumulative distribution function figure
from bbob_pproc import pprldistr
ds = bb.load(glob.glob('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f0*_20.pickle'))
figure()
pprldistr.plot(ds)
pprldistr.beautify() # resize the window to view whole figure

# Empirical cumulative distribution function of bootstrapped ERT figure
from bbob_pproc.compall import pprldmany
ds = bb.load(glob.glob('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f0*_20.pickle'))
figure()
pprldmany.plot(ds)
pprldmany.beautify()

# Scaling figure
from bbob_pproc import ppfigdim
ds = bb.load(glob.glob('BBOB2009pythondata/BIPOP-CMA-ES/ppdata_f002_*.pickle'))
figure()
ppfigdim.plot(ds)
ppfigdim.beautify()

