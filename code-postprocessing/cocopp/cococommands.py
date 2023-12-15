#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A collection of commands for using COCO from Jupyter or IPython.

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

import collections as _collections
import warnings as _warnings
import numpy as np

from .pproc import get_DataSetList as _DataSetList
from .toolsdivers import StringList as _StringList
from .archiving import official_archives
from . import config as _config
from . import pproc as _pproc
from . import genericsettings as _genericsettings
from . import testbedsettings as _testbedsettings

def true_number_of_trials(ds):
    """return number of actually conducted trials for `DataSet` `ds`"""
    if len(ds.instancenumbers) != len(ds._evals[0]) - 1:
        _warnings.warn("DataSet {0}: instancenumbers = {1} != {2} = len(_evals[0]) - 1"
                       .format(ds.algId, ds.instancenumbers, len(ds._evals[0]) - 1))
    return len(ds.instancenumbers)
class DataWithFewSuccesses:
    """The `all` property is an `OrderedDict` with all ``(dimension, funcId)``-

    tuples that have less than ``self.minsuccesses`` successes. The tuples
    are the `dict` keys, the values are the respective numbers for
    ``[evaluations, successes, trials]``, where evaluations are the average
    overall (sum of) evaluations per instance. The minimal number of desired
    successes can be changed at any time by re-assigning the `minsuccesses`
    attribute in which case the return value of `result` may change. When
    the `success_threshold` attribute is change, `compute_successes` has to
    be called to update the results. The `with_budget_left` property gives
    those entries of `all` which have less than `budget_multiplier` x
    dimension evaluations. The `current` property gives `with_budget_left`
    or `all` if the former is empty.

    The first argument can be a folder or filename or any other string
    excepted by `cocopp.load` matching a single dataset or it can be a
    `DataSetList`.
    
    Usage concept example::

        >> run_more = DataWithFewSuccesses('folder_or_file_name_for_cocopp.load').current
        >> for p in cocoex.Suite('bbob', '', ''):
        ..     if (p.id_function, p.dimension) not in run_more:
        ..         continue
        ..     p.observe_with(...)
        ..     # run solver on problem p
        ..     [...]

    Details
    -------
    The success number is calculated with the `raw_value` parameter of
    `DataSet.detSuccesses` thereby bypassing successes from instance
    balancing copies. However it falls back with a warning when this does
    not work.

    In the experimental setup from 2009, the used budget [evaluations per
    instance] is "overestimated" by a factor of three because each instance
    is run three times to begin with.
    
    TODO: When in this case a `DataSetList` is passed instead
    of a folder name, we could check whether it was instantiated with
    ``genericsettings.balance_instances == False`` which is not the default
    setting. Either ``len(self.evals[0]) != len(self._evals[0])`` or
    ``self.instance_multipliers is not None and
    np.any(self.instance_multipliers > 1)`` indicate that a balancing
    action was taken.

    See also ``cocopp.load``.

    """
    @property
    def current(self):
        """`OrderedDict` of ``(function, dimension)`` with budget left or otherwise all """
        return self.with_budget_left or self.all
    @property
    def all(self):
        """depends on attributes `minsuccesses` and `successes` only"""
        return (_collections.OrderedDict(sorted(
                ((ds.funcId, ds.dim),
                 [self.evaluations[i], self.successes[i], len(ds.instancenumbers)])
                    for i, ds in enumerate(self.dsl)
                        if self.successes[i] < self.minsuccesses
                        and (ds.funcId, ds.dim) not in self.exclude)))
    @property
    def with_budget_left(self):
        """depends on attributes `minsuccesses`, `successes`, `budget_multiplier` and `evaluations`"""
        return _collections.OrderedDict(sorted(
            ((ds.funcId, ds.dim),
             [self.evaluations[i], self.successes[i], len(ds.instancenumbers)])
                for i, ds in enumerate(self.dsl)
                    if self.successes[i] < self.minsuccesses
                    and self.evaluations[i] < self.budget_multiplier * ds.dim
                    and (ds.funcId, ds.dim) not in self.exclude))
    def __init__(self, folder_name, minsuccesses=9, budget_multiplier=np.inf,
                 success_threshold=1e-8, exclude=()):
        """`folder_name` can also be a filename or a data entry or a `DataSetList`"""
        self.input_parameters = {it[0]: it[1] for it in list(locals().items())
                                 if it[0] != 'self'}  # for the record
        self.minsuccesses = minsuccesses
        self.budget_multiplier = budget_multiplier
        self.exclude = exclude
        self.success_threshold = success_threshold
        if isinstance(folder_name, list):
            self.dsl = folder_name
        else:
            _bi = _genericsettings.balance_instances
            _genericsettings.balance_instances = False  # not strictly necessary anymore
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                self.dsl = load(folder_name)
            _genericsettings.balance_instances = _bi
            if not self.dsl:
                _warnings.warn("Sorry, could not find any coco data in {}"
                               .format(folder_name))
        self.trials = [true_number_of_trials(ds) for ds in self.dsl]
        """number of trials in each data set, for the record only"""
        self.successes = self.compute_successes().successes  # declarative assignment
        """list of successful trials, depends on `success_threshold`.
           Can be recomputed by calling `compute_successes`.
           """
        self.evaluations = self.compute_evaluations()
        """list of average evaluations per instance for each function+dimension"""
    def compute_successes(self, success_threshold=None):
        """Assign `successes` attribute as a `list` of number of successful trials

        in the data sets of `self.dsl` and return `self`. When given, reassign
        also the `success_threshold` attribute.
        """
        if success_threshold is not None:
            self.success_threshold = success_threshold
        try:
            self.successes = [ds.detSuccesses([self.success_threshold], raw_values=True)[0]
                              for ds in self.dsl]
        except TypeError:  # this should never happen
            _warnings.warn("calling `detSuccesses(..., raw_values=True)` failed, "
                          "falling back to no argument (this should never happen)")
            self.successes = [ds.detSuccesses([self.success_threshold])[0] for ds in self.dsl]
        return self
    def compute_evaluations(self):
        """assign `.evaluations` as a `list` of overall evaluations spent per instance (on average)

        for each function+dimension.
        """
        self.evaluations = [sum(ds.maxevals[:true_number_of_trials(ds)])
                            / len(set(ds.instancenumbers)) for ds in self.dsl]
        # self.evaluations = [ds.detAverageEvals([self.success_threshold])[0] * ds.nbRuns() for ds in self.dsl]
        return self.evaluations
    def print(self):
        """return a `str` with the number of data sets with too few successes"""
        return 'DataWithFewSuccesses: {}/{}'.format(len(self.result), len(self.dsl))
    def __len__(self):
        return len(self.result)


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
