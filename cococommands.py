#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for using COCO from the (i)Python interpreter.

For all operations in the Python interpreter, it will be assumed that
the package has been imported as bb, just like it is done in the first
line of the examples below.

The main data structures used in COCO are :py:class:`DataSet`, which
corresponds to data of one algorithm on one problem, and
:py:class:`DataSetList`, which is for collections of :py:class:`DataSet`
instances. Both classes are implemented in :py:mod:`bbob_pproc.pproc`.

Examples:

* Start by importing :py:mod:`bbob_pproc`::

    >>> import bbob_pproc as bb # load bbob_pproc
    >>> import os
    >>> os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

* Load a data set, assign to variable :py:data:`ds`::

      >>> ds = bb.load('BIPOP-CMA-ES_hansen_noiseless/bbobexp_f2.info') # doctest:+ELLIPSIS
      Processing BIPOP-CMA-ES_hansen_noiseless/bbobexp_f2.info.
      ...
      Processing ['BIPOP-CMA-ES_hansen_noiseless/data_f2/bbobexp_f2_DIM40.tdat']: 15/15 trials found.

* Get some information on a :py:class:`DataSetList` instance::

      >>> print ds # doctest:+ELLIPSIS
      [DataSet(cmaes V3.30.beta on f2 2-D), ..., DataSet(cmaes V3.30.beta on f2 40-D)]
      >>> bb.info(ds)
      6 data set(s)
      Algorithm(s): cmaes V3.30.beta
      Dimension(s): 2, 3, 5, 10, 20, 40
      Function(s): 2
      Max evals: 75017
      Df      |     min       10      med       90      max
      --------|--------------------------------------------
      1.0e+01 |      55      151     2182    49207    55065
      1.0e+00 |     124      396     2820    56879    59765
      1.0e-01 |     309      466     2972    61036    66182
      1.0e-03 |     386      519     3401    67530    72091
      1.0e-05 |     446      601     3685    70739    73472
      1.0e-08 |     538      688     4052    72540    75010

"""

from __future__ import absolute_import

#from bbob_pproc import ppsingle, ppfigdim, dataoutput
# from bbob_pproc.pproc import DataSetList, DataSet
from bbob_pproc import pproc

#__all__ = ['load', 'info', 'pickle', 'systeminfo', 'DataSetList', 'DataSet']

def load(filename):
    """Create a :py:class:`DataSetList` instance from a file or folder.
    
    Input argument filename can be a single :file:`info` file name, a
    single pickle filename or a folder name. In the latter case, the
    folder is browsed recursively for :file:`info` or :file:`pickle`
    files.

    """
    return pproc.DataSetList(filename)

# info on the DataSetList: algId, function, dim

def info(dsList):
    """Display more info on an instance of DatasetList."""
    dsList.info()

# TODO: method for pickling data in the current folder!
def pickle(dsList):
    """Pickle a DataSetList."""
    dsList.pickle(verbose=True)
    # TODO this will create a folder with suffix -pickle from anywhere:
    # make sure the output folder is created at the right location

def systeminfo():
    """Display information on the system."""
    import sys
    print sys.version
    import numpy
    print 'Numpy %s' % numpy.__version__
    import matplotlib
    print 'Matplotlib %s' % matplotlib.__version__
    import bbob_pproc
    print 'bbob_pproc %s' % bbob_pproc.__version__


#def examples():
#    """Execute example script from examples.py"""
#
#    from bbob_pproc import examples

#def plot(dsList):
#    """Generate some plots given a DataSetList instance."""

#    # if only a single data set
#    if len(dsList) == 1:
#        ppsingle.generatefig(dsList)
#        ppsingle.beautify()
#        plt.show()
#        # table?
#    else:
#        # scaling figure
#        ppfigdim.generatefig(dsList, (10., 1., 1e-1, 1e-2, 1e-3, 1e-5, 1e-8))
#        ppfigdim.beautify()

#    if 
    
    # all data sets are from the same algorithm
    # 

# do something to lead a single DataSet instead?

# TODO: make sure each module have at least one method that deals with DataSetList instances.

# TODO: data structure dictAlg?
# TODO: hide modules that are not necessary

#    bbob2010 (package)
#    bbobies
#    bestalg
#    toolsstats
#    bwsettings
#    changeAlgIdAndComment
#    dataoutput
#    determineFtarget
#    determineFtarget3
#    findfiles
#    genericsettings
#    grayscalesettings
#    minirun
#    minirun_2
#    ppfig
#    ppfigdim -> ERT vs dim 1 alg: 1 DataSetList ok
#    pplogloss -> ERT loss vs ? 1 alg, 1 dim: 1 DataSetList ok
#    pprldistr -> Runevals (or?) vs %) 1 alg 1 dim: 1 DataSet... ?
#    comp2 (package)
#        ppfig2 ->
#        pprldistr2 ->
#        ppscatter ->
#        pptable2 ->
#    compall (package)
#        ppfigs ->
#        ppperfprof ->
#        pprldmany ->
#        pptables ->
#    pproc
#    pprocold
#    pptable -> ERT, etc... vs target 1 alg
#    pptex
#    ranksumtest
#    readalign
#    readindexfiles  # obsolete
#    run
#    run2
#    runcomp2
#    runcompall
#    rungeneric
#    rungeneric1
#    rungeneric2
#    rungenericmany
#    runmarc

