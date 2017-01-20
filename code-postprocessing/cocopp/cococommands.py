#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for using COCO from the (i)Python interpreter.

For all operations in the Python interpreter, it will be assumed that
the package has been imported as bb, just like it is done in the first
line of the examples below.

The main data structures used in COCO are :py:class:`DataSet`, which
corresponds to data of one algorithm on one problem, and
:py:class:`DataSetList`, which is for collections of :py:class:`DataSet`
instances. Both classes are implemented in :py:mod:`cocopp.pproc`.

Examples:

* Start by importing :py:mod:`cocopp`::

    >>> import cocopp as pp # load COCO postprocessing
    >>> import os
    >>> import urllib
    >>> import tarfile
    >>> path = os.path.abspath(os.path.dirname(os.path.dirname('__file__')))
    >>> os.chdir(path)
    >>> pp.genericsettings.verbose = False # ensure to make below doctests work 

* Load a data set, assign to variable :py:data:`ds`::

    >>> infoFile = 'data/BIPOP-CMA-ES/bbobexp_f2.info'
    >>> if not os.path.exists(infoFile):
    ...   os.chdir(os.path.join(path, 'data'))
    ...   dataurl = 'http://coco.gforge.inria.fr/data-archive/2009/BIPOP-CMA-ES_hansen_noiseless.tgz'
    ...   filename, headers = urllib.urlretrieve(dataurl)
    ...   archivefile = tarfile.open(filename)
    ...   archivefile.extractall()
    ...   os.chdir(path)
    >>> ds = pp.load(infoFile)
      Data consistent according to test in consistency_check() in pproc.DataSet

* Get some information on a :py:class:`DataSetList` instance::

    >>> print ds # doctest:+ELLIPSIS
    [DataSet(BIPOP-CMA-ES on f2 2-D), ..., DataSet(BIPOP-CMA-ES on f2 40-D)]
    >>> pp.info(ds)
    6 data set(s)
    Algorithm(s): BIPOP-CMA-ES
    1 Function with ID 2
    Dimension(s): 2, 3, 5, 10, 20, 40
    Max evals: [762, 1537, 2428, 6346, 20678, 75010]

"""

from __future__ import absolute_import

from . import pproc

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
    dsList.pickle()
    # TODO this will create a folder with suffix -pickle from anywhere:
    # make sure the output folder is created at the right location

def systeminfo():
    """Display information on the system."""
    import sys
    print(sys.version)
    import numpy
    print('Numpy %s' % numpy.__version__)
    import matplotlib
    print('Matplotlib %s' % matplotlib.__version__)
    try:
        from . import __version__ as version
    except:
        from cocopp import __version__ as version
    print('cocopp %s' % version)


# do something to lead a single DataSet instead?

# TODO: make sure each module have at least one method that deals with DataSetList instances.

# TODO: data structure dictAlg?
# TODO: hide modules that are not necessary
