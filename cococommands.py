#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module containing the main commands to use on the (i)python command shell.

The package heavily relies on matplotlib plotting facilities:
http://matplotlib.sourceforge.net/


"""

from __future__ import absolute_import

#from bbob_pproc import ppsingle, ppfigdim, dataoutput
from bbob_pproc.pproc import DataSetList, DataSet

def load(filename):
    """Create a DataSetList instance from a file which filename is provided.
    
    Input argument filename can be a single info filename, a single pickle
    filename or a folder name.

    """

    return DataSetList(filename)

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
# Figure should show up and not close afterwards (=> do something about this: matplotlib.use('Agg') )

# TODO: data structure dictAlg?
# TODO: hide modules that are not necessary

#    bbob2010 (package)
#    bbobies
#    bestalg
#    bootstrap
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
#        pptables ->
#    pproc
#    pprocold
#    pptable -> ERT, etc... vs target 1 alg
#    pptex
#    ranksumtest
#    readalign
#    readindexfiles
#    run
#    run2
#    runcomp2
#    runcompall
#    rungeneric
#    rungeneric1
#    rungeneric2
#    rungenericmany
#    runmarc

