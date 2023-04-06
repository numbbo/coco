#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depreciated (`cocopp` itself is to be used from Jupyter or IPython):
Module for using COCO from the (i)Python interpreter.

The main data structures used in COCO are :py:class:`DataSet`, which
corresponds to data of one algorithm on one problem, and
:py:class:`DataSetList`, which is for collections of :py:class:`DataSet`
instances. Both classes are implemented in :py:mod:`cocopp.pproc`.

Examples:

* Start by importing :py:mod:`cocopp`::

    >>> import cocopp
    >>> cocopp.genericsettings.verbose = False # ensure to make below doctests work
    >>> def print_(*args, **kwargs): pass
    >>> cocopp.archives.bbob._print = print_  # avoid download notification

* Load a data set, assign to variable :py:data:`ds`::

    >>> path = cocopp.archives.bbob.get(4)
    >>> print('ESC'); dsl = cocopp.load(path)  # a dataset list  # doctest:+ELLIPSIS
    ESC...
    >>> ds = dsl[0]

* Get some information on a :py:class:`DataSetList` instance::

    >>> print(dsl)  # doctest:+ELLIPSIS
    [DataSet(BIPOP-CMA-ES_hansen on f1 2-D), DataSet(BIPOP-CMA-ES...
    >>> dsl.info()
    144 data set(s)
    Algorithm(s): BIPOP-CMA-ES_hansen
    24 Functions with IDs 1-24
    Dimension(s): 2, 3, 5, 10, 20, 40
    Max evals: [1625595, 2349823, 3114271, 5884514, 12102699, 36849608]

* Get some information on a :py:class:`DataSet` instance::

    >>> print(ds)
    DataSet(BIPOP-CMA-ES_hansen on f1 2-D)
    >>> ds.info()
    Algorithm: BIPOP-CMA-ES_hansen
    Function ID: 1
    Dimension DIM = 2
    Number of trials: 15
    Final target Df: 1e-08
    min / max number of evals per trial: 224 / 333
       evals/DIM:  best     15%     50%     85%     max |  ERT/DIM  nsucc
      ---Df---|-----------------------------------------|----------------
      1.0e+03 |       0       0       0       0       0 |      0.5  15
      1.0e+01 |       0       0       2       8      10 |      2.9  15
      1.0e-01 |       8      13      22      38      52 |     24.2  15
      1.0e-03 |      34      48      56      74      77 |     58.2  15
      1.0e-05 |      64      70      89     100     102 |     86.1  15
      1.0e-08 |     112     116     128     150     166 |    130.9  15

"""

from __future__ import absolute_import

from .pproc import get_DataSetList as _DataSetList
from .toolsdivers import StringList as _StringList
from .archiving import official_archives
from . import config as _config
from . import pproc as _pproc
from . import testbedsettings as _testbedsettings

def load(filename):
    """[currently broken when further used within `cocopp`, see `load2`] Create a :py:class:`DataSetList` instance from a file or folder.

    Input argument filename can be a single :file:`info` file name, a
    single pickle filename or a folder name. In the latter case, the
    folder is browsed recursively for :file:`info` or :file:`pickle`
    files.

    Details: due to newly implemented side effects when data are read in,
    the returned data set list may not work anymore when used with plotting
    functions of the `cocopp` module, see also `load2`.
    """
    return _DataSetList(official_archives.all.get_extended(_StringList(filename)))

def load2(args):
    """[WIP] return a `dict` of `dict` of `DataSetLists` with dimension and pathname as keys.

    `args` is a string or a list of strings passed to
    `cocopp.official_archives.all.get_extended` to determine the desired
    data sets which can also come from a local folder or a zip-file.

    Examples:

    >>> import cocopp
    >>> def load2(s):
    ...     print(s)  # predictable output
    ...     return cocopp.load2(s)
    >>> def pprld(dsl):
    ...     print('_')  # predictable output
    ...     with cocopp.toolsdivers.InfolderGoneWithTheWind():
    ...         cocopp.compall.pprldmany.main(dsl)  # writes pprldmany_default.*
    >>> ddsl = load2('bbob/2009/B*')  # doctest:+ELLIPSIS
    bbob/200...
    >>> assert sorted(ddsl) == [2, 3, 5, 10, 20, 40], ddsl
    >>> assert all([len(ddsl[i]) == 3 for i in ddsl]), ddsl  # 3 algorithms
    >>> pprld(ddsl[2])  # doctest:+ELLIPSIS
    _...

    ::

        >> ddsl31 = load('bbob/2009/*')  # 31 data sets, takes ~3 minutes at first ever loading
        >> assert sorted(ddsl31) == [2, 3, 5, 10, 20, 40], ddsl
        >> assert len(ddsl31[3]) == 31, ddsl

    """
    args2 = official_archives.all.get_extended(args)
    dsList, _sortedAlgs, dictAlg = _pproc.processInputArgs(args2, True)  # takes ~1 minutes per 10 data sets
    dsList2 = _pproc.DataSetList(_testbedsettings.current_testbed.filter(dsList))
    dictAlg = dsList2.dictByAlgName()
    _config.config() # make sure that the filtered settings are taken into account?
    return _pproc.dictAlgByDim(dictAlg)

# info on the DataSetList: algId, function, dim

def info(dsList):
    """Display more info on an instance of DatasetList."""
    dsList.info()

# pproc.get_DataSetList writes and loads the pickled class
def _pickle(dsList):
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
