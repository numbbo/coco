#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Raw post-processing routines.

This module implements class :py:class:`DataSet`, unit element in the
post-processing and class :py:class:`DataSetList`, sequence of instances
of :py:class:`DataSet`.

Futhermore it implements methods for dealing with a third data structure
which is a dictionary of :py:class:`DataSetList` which is handy when
dealing with :py:class:`DataSetList` instances from multiple algorithms
for comparisons.

"""

# TODO: dictAlg should become the class DictALg that is a dictionary of DataSetLists with
# usecase DictAlg(algdict).by_dim() etc  

from __future__ import absolute_import

import sys
import os
import ast
import re
import pickle, gzip  # gzip is for future functionality: we probably never want to pickle without gzip anymore
import warnings
from pdb import set_trace
import numpy, numpy as np
import matplotlib.pyplot as plt
from bbob_pproc import genericsettings, findfiles, toolsstats, toolsdivers
from bbob_pproc import genericsettings as gs
from bbob_pproc.readalign import split, alignData, HMultiReader, VMultiReader
from bbob_pproc.readalign import HArrayMultiReader, VArrayMultiReader, alignArrayData
from bbob_pproc.ppfig import consecutiveNumbers

do_assertion = genericsettings.force_assertions # expensive assertions
targets_displayed_for_info = [10, 1., 1e-1, 1e-3, 1e-5, 1e-8]  # only to display info in DataSetList.info
maximal_evaluations_only_to_last_target = False  # was true in release 13.03, leads naturally to better results


def _DataSet_complement_data(self, step=10**0.2, final_target=1e-8):
    """insert a line for each target value.

    To be resolved: old data sets don't have this method,
    therefore it must be global in the module

    """
    try:
        if self._is_complemeted_data:
            return
    except AttributeError:
        pass
    if step < 1:
        step = 1. / step
    assert step > 1
    
    # check that step splits 10 into uniform intervals on the log scale
    if np.abs(0.2 / np.log10(step) - np.round(0.2 / np.log10(step))) > 1e-11:
        print np.log10(step) 
        raise NotImplementedError('0.2 / log10(step) must be an integer to be compatible with previous data')

    i = 0
    newdat = []
    self.evals = np.array(self.evals, copy=False)
    for i in xrange(len(self.evals) - 1):
        newdat.append(self.evals[i])
        target = self.evals[i][0] / step
        while target >= final_target and target > self.evals[i+1][0] and target / self.evals[i+1][0] - 1 > 1e-9:
            newdat.append(self.evals[i+1])
            newdat[-1][0] = target
            target /= step
    newdat.append(self.evals[-1])
    
    # raise NotImplementedError('needs yet to be re-checked, tested, approved')  # check newdat and self.evals
    self.evals = np.array(newdat)  # for portfolios, self.evals is not an array
    assert np.max(np.abs(self.evals[:-1, 0] / self.evals[1:, 0] - 10**step)) < 1e-11
    self._is_complemented_data = True # TODO: will remain true forever, this needs to be set to False again somewhere? 

def cocofy(filename):
    """Replaces bbob_pproc references in pickles files with coco_pproc
        This could become necessary for future backwards compatibility,
        however rather should become a class method. """
    import fileinput
    for line in fileinput.input(filename, inplace=1):
#       if "bbob" in line:
        sys.stdout.write(line.replace("bbob_pproc","coco_pproc"))
    fileinput.close

# CLASS DEFINITIONS

class TargetValues(object):
    """store and retrieve a list of target function values::
    
        >>> import bbob_pproc.pproc as pp
        >>> targets = [10**(i/5.0) for i in xrange(2, -8, -1)]
        >>> targets_as_class = pp.TargetValues(targets)
        >>> assert all(targets_as_class() == targets)
    
    In itself this class is useless, as it does not more than a simple list
    could do, but it serves as interface for derived classes, where ``targets()``
    requires an actual argument ``targets(fun_dim)``. 
    
    Details: The optional argument for calling the class instance is needed to 
    be consistent with the derived ``class RunlengthBasedTargetValues``. 
    
    """
    def __init__(self, target_values):
        if 11 < 3 and isinstance(target_values, TargetValues):  # type cast passing behavior
            self.__dict__ = target_values.__dict__
            return
        self.target_values = sorted(target_values, reverse=True)
        self.short_info = ""

    @staticmethod
    def cast(target_values_or_class_instance, *args, **kwargs):
        """idempotent cast to ``TargetValues`` class type, specifically
        ``return TargetValues(target_values_or_class_instance) 
            if not isinstance(target_values_or_class_instance, TargetValues) 
            else target_values_or_class_instance`` 
            
        """
        if isinstance(target_values_or_class_instance, TargetValues):
            return target_values_or_class_instance
        else:
            return TargetValues(target_values_or_class_instance, *args, **kwargs)

    def __len__(self):
        return len(self.target_values)

    def __call__(self, fun_dim_but_not_use=None):
        return self.target_values

    def label(self, i):
        """return the ``i``-th target value as ``str``, to be overwritten by a derived class"""
        return toolsdivers.num2str(self.target_values[i], significant_digits=2)

    def loglabel(self, i, decimals=0):
        """return ``log10`` of the ``i``-th target value as ``str``, to be overwritten by a derived class"""
        # return str(int(np.round(np.log10(self.target_values[i]), decimals)))
        return toolsdivers.num2str(np.log10(self.target_values[i]), significant_digits=decimals+1)

    def labels(self):
        """target values as a list of ``str``"""
        i, res = 0, []
        try:
            while True:
                res.append(self.label(i))
                i += 1
        except IndexError:
            return res
 
    def loglabels(self, decimals=0):
        """``log10`` of the target values as a list of ``str``"""
        i, res = 0, []
        try:
            while True:
                res.append(self.loglabel(i, decimals))
                i += 1
        except IndexError:
            return res

    def label_name(self):
        return 'Df'

class RunlengthBasedTargetValues(TargetValues):
    """a class instance call returns f-target values based on 
    reference runlengths::
    
        >>> import bbob_pproc as bb
        >>> targets = bb.pproc.RunlengthBasedTargetValues([0.5, 1.2, 3, 10, 50])  # by default times_dimension=True 
        >>> targets(fun_dim=(1, 20))
        
    returns a list of target f-values for F1 in 20-D, based on the 
    ERT values ``[0.5,...,50]``. 
        
    Details: The computation starts from the smallest budget and the resulting f-target 
    must always be at least a factor of ``force_different_targets_factor`` smaller 
    than the previous one. If the ``smallest_target`` is superseded, the log values
    are linearly rescaled such that the easiest found target remains the same and 
    the smallest target becomes ``smallest_target``. 
    
    TODO: see compall/determineFtarget2.FunTarget
    
    """
    @staticmethod
    def cast(run_lengths_or_class_instance, *args, **kwargs):
        """idempotent cast to ``RunlengthBasedTargetValues`` class type"""
        if isinstance(run_lengths_or_class_instance, RunlengthBasedTargetValues):
            return run_lengths_or_class_instance
        else:
            return RunlengthBasedTargetValues(run_lengths_or_class_instance, *args, **kwargs)
        
    @property
    def short_info(self):
        return self._short_info    
        
    def __init__(self, run_lengths, reference_data='bestGECCO2009', #  
                 smallest_target=1e-8, times_dimension=True, 
                 force_different_targets_factor=10**0.04,
                 unique_target_values=False,
                 step_to_next_difficult_target=10**0.2):
        """calling the class instance returns run-length based
        target values based on the reference data, individually
        computed for a given ``(funcId, dimension)``. 
        
        :param run_lengths: sequence of values. 
        :param reference_data: 
            can be a string like ``"bestGECCO2009"`` or a 
            ``DataSetList`` (not thoroughly tested). 
        :param smallest_target:
        :param times_dimension:
        :param force_different_targets_factor:
            given the target values are computed from the 
            ``reference_data_set``, enforces that all target
            values are different by at last ``forced_different_targets_factor``
            if ``forced_different_targets_factor``. Default ``10**0.04`` means 
            that within the typical precision of ``10**0.2`` at most five 
            consecutive targets can be identical.
        :param step_to_next_difficult_target:
            the next more difficult target (just) not reached within the
            target run length is chosen, where ``step_to_next_difficult_target``
            defines "how much more difficult".

        """
        self.reference_data = reference_data
        if force_different_targets_factor < 1:
            force_different_targets_factor **= -1 
        # TODO: known_names collects only bestalg stuff, while also algorithm data can be used (see def initialize below) 
        self.known_names = ['bestGECCO2009', 'bestGECCOever'] # TODO: best-ever is not a time-invariant thing and therefore ambiguous
        self._short_info = "budget-based"
        self.run_lengths = sorted(run_lengths)
        self.smallest_target = smallest_target
        self.step_to_next_difficult_target = step_to_next_difficult_target**np.sign(np.log(step_to_next_difficult_target))
        self.times_dimension = times_dimension
        self.unique_target_values = unique_target_values
        # force_different_targets and target_discretization could talk to each other? 
        self.force_different_targets_factor = force_different_targets_factor
        self.target_discretization_factor = 10**0.2  # in accordance with default recordings
        self.reference_algorithm = ''
        self.initialized = False
    def initialize(self):
        """lazy initialization to prevent slow import"""
        if self.initialized:
            return self
        if self.reference_data in self.known_names: # bestalg data are loaded
            self.reference_algorithm = self.reference_data
            self._short_info = 'reference budgets from ' + self.reference_data
            if self.reference_data == 'bestGECCO2009':
                from bbob_pproc import bestalg
                bestalg.loadBBOB2009() # this is an absurd interface
                self.reference_data = bestalg.bestalgentries2009
                # TODO: remove targets smaller than 1e-8 
            elif self.reference_data == 'bestGECCOever':
                from bbob_pproc import bestalg
                bestalg.loadBBOBever() # this is an absurd interface
                self.reference_data = bestalg.bestalgentriesever
            else:
                ValueError('reference algorithm name')
        elif type(self.reference_data) is str:  # self.reference_data in ('RANDOMSEARCH', 'IPOP-CMA-ES') should work 
            self._short_info = 'reference budgets from ' + self.reference_data
            dsl = DataSetList(os.path.join(sys.modules[globals()['__name__']].__file__.split('bbob_pproc')[0], 
                                           'bbob_pproc', 'data', self.reference_data))  
            dsd = {}
            for ds in dsl:
                # ds._clean_data()
                dsd[(ds.funcId, ds.dim)] = ds
            self.reference_data = dsd
        else:
            # assert len(byalg) == 1
            self.reference_algorithm = self.reference_data[self.reference_data.keys()[0]].algId
        self.initialized = True
        return self
    def __len__(self):
        return len(self.run_lengths)  
    def __call__(self, fun_dim=None):
        """Get all target values for the respective function and dimension  
        and reference ERT values (passed during initializatio). `fun_dim` is 
        a tuple ``(fun_nb, dimension)`` like ``(1, 20)`` for the 20-D sphere. 
        
        Details: f_target = arg min_f { ERT_best(f) > max(1, target_budget * dimension**times_dimension_flag) }, 
        where f are the values of the ``DataSet`` ``target`` attribute. The next difficult target is chosen
        not smaller as target / 10**0.2. 
        
        Shown is the ERT for targets that, within the given budget, the best 2009 algorithm just failed to achieve.

        """            
        self.initialize()
        if self.force_different_targets_factor**len(self.run_lengths) > 1e3:
                warnings.warn('enforced different target values might spread more than three orders of magnitude')
        if fun_dim is None:
            raise ValueError('call to RunlengthbasedTargetValues class instance needs the parameter ``fun_dim``, none given')
        fun_dim = tuple(fun_dim)
        dim_fun = tuple(reversed(fun_dim))
        if fun_dim[0] > 100 and self.run_lengths[-1] * fun_dim[1]**self.times_dimension < 1e3:
            ValueError("short running times don't work on noisy functions")
        ds = self.reference_data[dim_fun]
        if 11 < 3:   
            try:
                ds._complement_data() # is not fully implemented and not here not necessary
            except AttributeError: # loaded classes might not have this method
                # need a hack here
                _DataSet_complement_data(ds, self.target_discretization_factor)
        # end is the first index in ds.target with a values smaller than smallest_target
        try:
            end = np.nonzero(ds.target >= self.smallest_target)[0][-1] + 1 
            # same as end = np.where(ds.target >= smallest_target)[0][-1] + 1 
        except IndexError:
            end = len(ds.target)

        if genericsettings.test:
            if 11 < 3 and not toolsdivers.equals_approximately(ds.target[end-2] / ds.target[end-1], 10**0.2, 1e-8):
                print 'last two targets before index', end
                print ds.target[end-2:end]
            try: 
                assert ds.ert[0] == 1  # we might have to compute these the first time
            except AssertionError:
                print fun_dim, ds.ert[0], 'ert[0] != 1 in TargetValues.__call__' 
            try: 
                # check whether there are gaps between the targets 
                assert all(toolsdivers.equals_approximately(10**0.2, ds.target[i] / ds.target[i+1]) for i in xrange(end-1))
                # if this fails, we need to insert the missing target values 
            except AssertionError:
                if 1 < 3:
                    print fun_dim, ds.ert[0], 'not all targets are recorded in TargetValues.__call__ (this could be a bug)' 
                    print ds.target
                    # 1/0
        
        # here the actual computation starts
        targets = []
        if genericsettings.test: 
            for rl in self.run_lengths:
                # choose largest target not achieved by reference ERT
                indices = np.nonzero(ds.ert[:end] > np.max((1, rl * (fun_dim[1] if self.times_dimension else 1))))[0]
                if len(indices):  # larger ert exists
                    targets.append(np.max((ds.target[indices[0]],  # first missed target 
                                           (1 + 1e-9) * ds.target[indices[0] - 1] / self.step_to_next_difficult_target))) # achieved target / 10*0.2
                else:
                    # TODO: check whether this is the final target! If not choose a smaller than the last achieved one. 
                    targets.append(ds.target[end-1])  # last target
                    if targets[-1] > (1 + 1e-9) * self.smallest_target:
                        targets[-1] = (1 + 1e-9) * targets[-1] / self.step_to_next_difficult_target
                
                if len(targets) > 1 and targets[-1] >= targets[-2] and self.force_different_targets_factor > 1 and targets[-1] > self.smallest_target:
                    targets[-1] = targets[-2] / self.force_different_targets_factor
            targets = np.array(targets, copy=False)
            targets[targets < self.smallest_target] = self.smallest_target
            
            # a few more sanity checks
            if targets[-1] < self.smallest_target:
                print 'runlength based targets', fun_dim, ': correction for small smallest target applied (should never happen)'
                b = float(targets[0])
                targets = np.exp(np.log(targets) * np.log(b / self.smallest_target) / np.log(b / targets[-1]))
                targets *= (1 + 1e-12) * self.smallest_target / targets[-1]
                assert b <= targets[0] * (1 + 1e-10)
                assert b >= targets[0] / (1 + 1e-10)
            assert targets[-1] >= self.smallest_target
            assert len(targets) == 1 or all(np.diff(targets) <= 0)
            assert len(ds.ert) == len(ds.target)
        
        # here the actual computation starts
        old_targets = targets
        targets = [] 
        for rl in self.run_lengths:
            # choose best target achieved by reference ERT times step_to_next_difficult_target
            indices = np.nonzero(ds.ert[:end] <= np.max((1, rl * (fun_dim[1] if self.times_dimension else 1))))[0]
            if not len(indices):
                warnings.warn('  too easy runlength ' + str(rl) + ' for (f,dim)=' + str(fun_dim))
                targets.append(ds.target[0])
            else:
                targets.append((1 + 1e-9) * ds.target[indices[-1]] / self.step_to_next_difficult_target)
            
            if (len(targets) > 1 and targets[-1] >= targets[-2] and 
                self.force_different_targets_factor > 1):
                targets[-1] = targets[-2] / self.force_different_targets_factor
        targets = np.array(targets, copy=False)
        targets[targets < self.smallest_target] = self.smallest_target
        
        # a few more sanity checks
        if targets[-1] < self.smallest_target:
            print 'runlength based targets', fun_dim, ': correction for small smallest target applied (should never happen)'
            b = float(targets[0])
            targets = np.exp(np.log(targets) * np.log(b / self.smallest_target) / np.log(b / targets[-1]))
            targets *= (1 + 1e-12) * self.smallest_target / targets[-1]
            assert b <= targets[0] * (1 + 1e-10)
            assert b >= targets[0] / (1 + 1e-10)
        assert targets[-1] >= self.smallest_target
        assert len(targets) == 1 or all(np.diff(targets) <= 0)
        assert len(ds.ert) == len(ds.target)

        if genericsettings.test and not all(targets == old_targets): # or (fun_dim[0] == 19 and len(targets) > 1):
            print 'WARNING: target values are different compared to previous version'
            print fun_dim
            print targets / old_targets - 1
            print targets

        if self.unique_target_values:
            len_ = len(targets)
            targets = np.array(list(reversed(sorted(set(targets)))))
            # print(' '.join((str(len(targets)), 'of', str(len_), 'targets kept')))
        return targets    

    get_targets = __call__  # an alias
    
    def label(self, i):
        """return i-th target value as string"""
        return toolsdivers.num2str(self.run_lengths[i], significant_digits=2)

    def loglabel(self, i, decimals=1):
        """``decimals`` is used for ``round``"""
        return toolsdivers.num2str(np.log10(self.run_lengths[i]), significant_digits=decimals+1)
        # return str(np.round(np.log10(self.run_lengths[i]), decimals))
    
    def label_name(self):
        return 'RL' + ('/dim' if self.times_dimension else '')

    def _generate_erts(self, ds, target_values):
        """compute for all target values, starting with 1e-8, the ert value
        and store it in the reference_data_set attribute
        
        """
        raise NotImplementedError
              
class DataSet():
    """Unit element for the COCO post-processing.

    An instance of this class is created from one unit element of
    experimental data. One unit element would correspond to data for a
    given algorithm (a given :py:attr:`algId` and a :py:attr:`comment`
    line) and a given problem (:py:attr:`funcId` and
    :py:attr:`dimension`).

    Class attributes:

      - *funcId* -- function Id (integer)
      - *dim* -- dimension (integer)
      - *indexFiles* -- associated index files (list of strings)
      - *dataFiles* -- associated data files (list of strings)
      - *comment* -- comment for the setting (string)
      - *targetFuncValue* -- final target function value (float), might be missing
      - *precision* -- final ftarget - fopt (float), data with 
                       target[idat] < precision are optional and not relevant.  
      - *algId* -- algorithm name (string)
      - *evals* -- data aligned by function values (2xarray, list of data rows [f_val, eval_run1, eval_run2,...]), cave: in a portfolio data rows can have different lengths
      - *funvals* -- data aligned by function evaluations (2xarray)
      - *maxevals* -- maximum number of function evaluations (array)
      - *finalfunvals* -- final function values (array)
      - *readmaxevals* -- maximum number of function evaluations read
                          from index file (array)
      - *readfinalFminusFtarget* -- final function values - ftarget read
                                    from index file (array)
      - *pickleFile* -- associated pickle file name (string)
      - *target* -- target function values attained (array)
      - *ert* -- ert for reaching the target values in target (array)
      - *instancenumbers* -- list of numbers corresponding to the instances of
                     the test function considered (list of int)
      - *isFinalized* -- list of bool for if runs were properly finalized

    :py:attr:`evals` and :py:attr:`funvals` are arrays of data collected
    from :py:data:`N` data sets.

    Both have the same format: zero-th column is the value on which the
    data of a row is aligned, the :py:data:`N` subsequent columns are
    either the numbers of function evaluations for :py:attr:`evals` or
    function values for :py:attr:`funvals`.
    
    A short example::
    
        >>> import sys
        >>> import os
        >>> os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
        >>> import bbob_pproc as bb
        >>> dslist = bb.load('BIPOP-CMA-ES_hansen_noiseless/bbobexp_f2.info')
        >>> dslist  # nice display in particular in IPython
        [DataSet(cmaes V3.30.beta on f2 2-D),
         DataSet(cmaes V3.30.beta on f2 3-D),
         DataSet(cmaes V3.30.beta on f2 5-D),
         DataSet(cmaes V3.30.beta on f2 10-D),
         DataSet(cmaes V3.30.beta on f2 20-D),
         DataSet(cmaes V3.30.beta on f2 40-D)]
        >>> type(dslist)
        <class 'bbob_pproc.pproc.DataSetList'>
        >>> len(dslist)
        6
        >>> ds = dslist[3]  # a single data set of type DataSet
        >>> ds
        DataSet(cmaes V3.30.beta on f2 10-D)
        >>> for d in dir(ds): print d  # dir(ds) shows attributes and methods of ds
        _DataSet__parseHeader
        __doc__
        __eq__
        __init__
        __module__
        __ne__
        __repr__
        _attributes
        _detEvals2
        _extra_attr
        algId
        comment
        computeERTfromEvals
        createDictInstance
        createDictInstanceCount
        dataFiles
        detAverageEvals
        detERT
        detEvals
        detSuccessRates
        detSuccesses
        dim
        ert
        evals
        finalfunvals
        funcId
        funvals
        generateRLData
        indexFiles
        info
        instancenumbers
        isFinalized
        mMaxEvals
        maxevals
        nbRuns
        pickle
        precision
        readfinalFminusFtarget
        readmaxevals
        splitByTrials
        target
        >>> all(ds.evals[:, 0] == ds.target)  # first column of ds.evals is the "target" f-value
        True
        >>> ds.evals[0::10,:][:, (0,5,6)]  # show row 0,10,20,... and of the result columns 0,5,6, index 0 is ftarget
        array([[  3.98107171e+07,   1.00000000e+00,   1.00000000e+00],
               [  3.98107171e+05,   2.00000000e+01,   8.40000000e+01],
               [  3.98107171e+03,   1.61600000e+03,   1.04500000e+03],
               [  3.98107171e+01,   3.04400000e+03,   3.21000000e+03],
               [  3.98107171e-01,   4.42400000e+03,   5.11800000e+03],
               [  3.98107171e-03,   4.73200000e+03,   5.41300000e+03],
               [  3.98107171e-05,   5.04000000e+03,   5.74800000e+03],
               [  3.98107171e-07,   5.36200000e+03,   6.07000000e+03],
               [  3.98107171e-09,   5.68200000e+03,              nan]])

        >>> ds.evals[-1,:][:, (0,5,6)]  # show last row, same columns
        array([  1.58489319e-09,              nan,              nan])
        >>> ds.info()  # prints similar data more nicely formated 
        Algorithm: cmaes V3.30.beta
        Function ID: 2
        Dimension:10
             Df  evals: best    10%     25%     50%     75%     90%     max
          __________________________________________________________________
          1.0e+03  |    1018    1027    1381    1698    1921    2093    2348
          1.0e+01  |    2778    2942    3379    3645    4328    4576    4798
          1.0e-01  |    4019    4429    4812    4966    5180    5275    5361
          1.0e-03  |    4802    5150    5194    5287    5509    5546    5672
          1.0e-05  |    5131    5421    5490    5626    5812    5862    5929
          1.0e-08  |    5676    5886    5974    6112    6248    6307    6346

        >>> import numpy as np  # not necessary in IPython
        >>> idx = range(0, 50, 10) + [-1]
        >>> np.array([idx, ds.target[idx], ds.ert[idx]]).T  # ERT expected runtime for some targets
        array([[  0.00000000e+00,   3.98107171e+07,   1.00000000e+00],
               [  1.00000000e+01,   3.98107171e+05,   6.12666667e+01],
               [  2.00000000e+01,   3.98107171e+03,   1.13626667e+03],
               [  3.00000000e+01,   3.98107171e+01,   3.07186667e+03],
               [  4.00000000e+01,   3.98107171e-01,   4.81333333e+03]])
        
    """

    # TODO: unit element of the post-processing: one algorithm, one problem
    # TODO: if this class was to evolve, how do we make previous data
    # compatible?

    # Private attribute used for the parsing of info files.
    _attributes = {'funcId': ('funcId', int), 
                   'DIM': ('dim', int),
                   'Precision': ('precision', float), 
                   'Fopt': ('fopt', float),
                   'targetFuncValue': ('targetFuncValue', float),
                   'algId': ('algId', str)}

    def __init__(self, header, comment, data, indexfile, verbose=True):
        """Instantiate a DataSet.

        The first three input argument corresponds to three consecutive
        lines of an index file (.info extension).

        :keyword string header: information of the experiment
        :keyword string comment: more information on the experiment
        :keyword string data: information on the runs of the experiment
        :keyword string indexfile: string for the file name from where
                                   the information come
        :keyword bool verbose: controls verbosity

        """
        # Extract information from the header line.
        self._extra_attr = []
        self.__parseHeader(header)

        # Read in second line of entry (comment line). The information
        # is only stored if the line starts with "%", else it is ignored.
        if comment.startswith('%'):
            self.comment = comment.strip()
        else:
            #raise Exception()
            warnings.warn('Comment line: %s is skipped,' % (comment) +
                          'it does not start with \%.')
            self.comment = ''

        filepath = os.path.split(indexfile)[0]
        self.indexFiles = [indexfile]
        self.dataFiles = []
        self.instancenumbers = []
        self.evals = []  # to be removed if evals becomes a property, see below
        """``evals`` are the central data. Each line ``evals[i]`` has a 
        (target) function value in ``evals[i][0]`` and the function evaluation
        for which this target was reached the first time in trials 1,...
        in ``evals[i][1:]``.""" 
        self._evals = []  # not in use
        self.isFinalized = []
        self.readmaxevals = []
        self.readfinalFminusFtarget = []

        # Split line in data file name(s) and run time information.
        parts = data.split(', ')
        for elem in parts:
            if elem.endswith('dat'):
                #Windows data to Linux processing
                filename = elem
                # while elem.find('\\ ') >= 0:
                #     filename = filename.replace('\\ ', '\\')
                filename = filename.replace('\\', os.sep)
                #Linux data to Windows processing
                filename = filename.replace('/', os.sep)
                self.dataFiles.append(filename)
            else:
                if not ':' in elem:
                    # if elem does not have ':' it means the run was not
                    # finalized properly.
                    self.instancenumbers.append(ast.literal_eval(elem))
                    # In this case, what should we do? Either we try to process
                    # the corresponding data anyway or we leave it out.
                    # For now we leave it in.
                    self.isFinalized.append(False)
                    warnings.warn('Caught an ill-finalized run in %s for %s'
                                  % (indexfile,
                                     os.path.join(filepath, self.dataFiles[-1])))
                    self.readmaxevals.append(0)
                    self.readfinalFminusFtarget.append(numpy.inf)
                else:
                    itrial, info = elem.split(':', 1)
                    self.instancenumbers.append(ast.literal_eval(itrial))
                    self.isFinalized.append(True)
                    readmaxevals, readfinalf = info.split('|', 1)
                    self.readmaxevals.append(int(readmaxevals))
                    self.readfinalFminusFtarget.append(float(readfinalf))

        if verbose:
            print "%s" % self.__repr__()

        # Treat successively the data in dat and tdat files:
        # put into variable dataFiles the files where to look for data
        dataFiles = list(os.path.join(filepath, os.path.splitext(i)[0] + '.dat')
                         for i in self.dataFiles)
        data = HMultiReader(split(dataFiles))
        if verbose:
            print ("Processing %s: %d/%d trials found."
                   % (dataFiles, len(data), len(self.instancenumbers)))
        (adata, maxevals, finalfunvals) = alignData(data)
        self.evals = adata
        try:
            for i in range(len(maxevals)):
                self.maxevals[i] = max(maxevals[i], self.maxevals[i])
                self.finalfunvals[i] = min(finalfunvals[i], self.finalfunvals[i])
        except AttributeError:
            self.maxevals = maxevals
            self.finalfunvals = finalfunvals

        dataFiles = list(os.path.join(filepath, os.path.splitext(i)[0] + '.tdat')
                         for i in self.dataFiles)
        data = VMultiReader(split(dataFiles))
        if verbose:
            print ("Processing %s: %d/%d trials found."
                   % (dataFiles, len(data), len(self.instancenumbers)))
        (adata, maxevals, finalfunvals) = alignData(data)
        self.funvals = adata
        try:
            for i in range(len(maxevals)):
                self.maxevals[i] = max(maxevals[i], self.maxevals[i])
                self.finalfunvals[i] = min(finalfunvals[i], self.finalfunvals[i])
        except AttributeError:
            self.maxevals = maxevals
            self.finalfunvals = finalfunvals
        #TODO: take for maxevals the max for each trial, for finalfunvals the min...

        #extensions = {'.dat':(HMultiReader, 'evals'), '.tdat':(VMultiReader, 'funvals')}
        #for ext, info in extensions.iteritems(): # ext is defined as global
            ## put into variable dataFiles the files where to look for data
            ## basically append 
            #dataFiles = list(i.rsplit('.', 1)[0] + ext for i in self.dataFiles)
            #data = info[0](split(dataFiles))
            ## split is a method from readalign, info[0] is a method of readalign
            #if verbose:
                #print ("Processing %s: %d/%d trials found." #% (dataFiles, len(data), len(self.itrials)))
            #(adata, maxevals, finalfunvals) = alignData(data)
            #setattr(self, info[1], adata)
            #try:
                #if all(maxevals > self.maxevals):
                    #self.maxevals = maxevals
                    #self.finalfunvals = finalfunvals
            #except AttributeError:
                #self.maxevals = maxevals
                #self.finalfunvals = finalfunvals
        #CHECKING PROCEDURE
        tmp = []
        for i in range(min((len(self.maxevals), len(self.readmaxevals)))):
            tmp.append(self.maxevals[i] == self.readmaxevals[i])
        if not all(tmp) or len(self.maxevals) != len(self.readmaxevals):
            warnings.warn('There is a difference between the maxevals in the '
                          '*.info file and in the data files.')

        self._cut_data()
        # Compute ERT
        self.computeERTfromEvals()
        assert all(self.evals[0][1:] == 1)
        if not self.consistency_check(): # prints also warnings itself
            warnings.warn("Inconsistent data found for function " + str(self.funcId) + " in %d-D (see also above)" % self.dim) 

    @property
    def evals_(self):
        """Shall become ``evals`` attribute in future.
        
        ``evals`` are the central data. Each line ``evals[i]`` has a 
        (target) function value in ``evals[i][0]`` and the function evaluation
        for which this target was reached the first time in trials 1,...
        in ``evals[i][1:]``. 
        
        """
        return self._evals
    @evals_.setter
    def evals_(self, value):
        self._evals = value
    @evals_.deleter
    def evals_(self):
        del self._evals
        
    def _cut_data(self):
        """attributes `target`, `evals`, and `ert` are truncated to target values not 
        much smaller than defined in attribute `precision` (typically ``1e-8``). 
        Attribute `maxevals` is recomputed for columns that reach the final target
        precision. 
        
        """
        if isinstance(genericsettings.current_testbed, genericsettings.GECCOBBOBTestbed):
            Ndata = np.size(self.evals, 0)
            i = Ndata
            while i > 1 and self.evals[i-1][0] <= self.precision:
                i -= 1
            i += 1
            if i < Ndata:
                self.evals = self.evals[:i, :]  # assumes that evals is an array
                try:
                    self.target = self.target[:i]
                    assert self.target[-1] == self.evals[-1][0] 
                except AttributeError:
                    pass
                try:
                    self.ert = self.ert[:i]
                except AttributeError:
                    pass
            assert self.evals.shape[0] == 1 or self.evals[-2][0] > self.precision
            if self.evals[-1][0] < self.precision: 
                self.evals[-1][0] = np.max((self.precision / 1.001, self.evals[-1, 0])) 
                # warnings.warn('exact final precision was not recorded, next lower value set close to final precision')
                # print '*** warning: final precision was not recorded'
                assert self.evals[-1][0] < self.precision # shall not have changed
            assert self.evals[-1][0] > 0
            self.maxevals = self._detMaxEvals()

    def _complement_data(self, step=10**0.2, final_target=1e-8):
        """insert a line for each target value"""
        _DataSet_complement_data(self, step, final_target)

    def consistency_check(self):
        """yet a stump"""
        is_consistent = True
        if len(set(self.instancenumbers)) < len(self.instancenumbers):
            is_consistent = False
            warnings.warn('  double instances in ' + str(self.instancenumbers))
        return is_consistent
            
    def computeERTfromEvals(self):
        """Sets the attributes ert and target from the attribute evals."""
        self.ert = []
        self.target = []
        for i in self.evals:
            data = i.copy()
            data = data[1:]
            succ = (numpy.isnan(data)==False)
            if any(numpy.isnan(data)):
                data[numpy.isnan(data)] = self.maxevals[numpy.isnan(data)]
            self.ert.append(toolsstats.sp(data, issuccessful=succ)[0])
            self.target.append(i[0])

        self.ert = numpy.array(self.ert)
        self.target = numpy.array(self.target)

    def __eq__(self, other):
        """Compare indexEntry instances."""
        res = (self.__class__ is other.__class__ and
               self.funcId == other.funcId and
               self.dim == other.dim and
               self.precision == other.precision and
               self.algId == other.algId and
               self.comment == other.comment)
        if hasattr(self, '_extra_attr'): # Backward compatibility
            for i in self._extra_attr:
                res = (res and hasattr(other, i) and
                       getattr(other, i) == getattr(self, i))
                if not res:
                    break
        return res

    def __ne__(self,other):
        return not self.__eq__(other)

    def __repr__(self):
        res = ('DataSet(%s on f%s %d-D'
               % (self.algId, str(self.funcId), self.dim))
        for i in getattr(self, '_extra_attr', ()):
            res += ', %s = %s' % (i, getattr(self, i))
        res += ')'
        return res

    def info(self, targets=None):
        """print text info to stdout"""
        if targets is None:
            targets = [1e3, 10, 0.1, 1e-3, 1e-5, 1e-8]
            if self.target[-1] < targets[-1]:
                targets += [self.target[-1]]  
        if targets[-1] < self.target[-1]:
            targets[-1] = self.target[-1]
        targets = sorted(set(targets), reverse=True)  # remove dupicates and sort

        sinfo = 'Algorithm: ' + str(self.algId)
        sinfo += '\nFunction ID: ' + str(self.funcId)
        sinfo += '\nDimension DIM = ' + str(self.dim)
        sinfo += '\nNumber of trials: ' + str(self.nbRuns())
        sinfo += '\nFinal target Df: ' + str(self.precision)
        sinfo += '\nmin / max number of evals: '  + str(int(min(self.evals[0]))) + ' / '  + str(int(max(self.maxevals)))
        sinfo += '\n    evals/DIM: best    15%     50%     85%     max  |  ERT/DIM  nsucc'
        sinfo += '\n  ---Df---|-----------------------------------------|----------------'
        evals = self.detEvals(targets, copy=False)
        nsucc = self.detSuccesses(targets)
        ert = self.detERT(targets)
        for i, target in enumerate(targets):
            line = '  %.1e |' % target
            for val in toolsstats.prctile(evals[i], (0, 15, 50, 85, 100)): 
                line += ' %7d' % (val / self.dim) if not np.isnan(val) else '     .  ' 
            line += ' |' + ('%9.1f' % (ert[i] / self.dim) if np.isfinite(ert[i]) else '    inf  ') 
            # line += '  %4.2f' % (nsucc[i] / float(Nruns)) if nsucc[i] < Nruns else '  1.0 '
            line += '  %2d' % nsucc[i]
            sinfo += '\n' + line
            if target < self.target[-1]:
                break
        print sinfo
        # return sinfo 

    def mMaxEvals(self):
        """Returns the maximum number of function evaluations over all runs (trials), 
        obsolete and replaced by attribute `maxeval`
        
        """
        return max(self.maxevals)
    
    def _detMaxEvals(self, final_target=None):
        """computes for each data column the (maximal) evaluation 
        until final_target was reached, or ``self.maxevals`` otherwise. 
        
        """
        if final_target is None:
            final_target = self.precision
        res = np.array(self.maxevals, copy=True) if not maximal_evaluations_only_to_last_target else np.nanmax(self.evals, 0)
        final_evals = self.detEvals([final_target])[0]
        idx = np.isfinite(final_evals)
        res[idx] = final_evals[idx] 
        assert sum(res < np.inf) == len(res)
        assert len(res) == self.nbRuns()
        return res 
        
    
    @property  # cave: setters work only with new style classes
    def max_eval(self):
        """maximum number of function evaluations over all runs (trials)""" 
        return max(self.maxevals)

    def nbRuns(self):
        """Returns the number of runs."""
        return numpy.shape(self.evals)[1] - 1 

    def __parseHeader(self, header):
        """Extract data from a header line in an index entry."""
        
        # Split header into a list of key-value
        for attrname, attrvalue in parseinfo(header):
            try:
                setattr(self, self._attributes[attrname][0], attrvalue)
            except KeyError:
                warnings.warn('%s is an additional attribute.' % (attrname))
                setattr(self, attrname, attrvalue)
                self._extra_attr.append(attrname)
                # the attribute is set anyway, this might lead to some errors.
                continue
        #TODO: check that no compulsory attributes is missing:
        #dim, funcId, algId, precision
        return

    def pickle(self, outputdir=None, verbose=True, gzipped=True):
        """Save this instance to a pickle file.

        Saves this instance to a (by default gzipped) pickle file. If not 
        specified by argument outputdir, the location of the pickle is 
        given by the location of the first index file associated to this
        instance.

        This method will overwrite existing files.
        
        """
        # the associated pickle file does not exist
        if outputdir is not None and getattr(self, 'pickleFile', False):
            NotImplementedError('outputdir and pickleFile attribute are in conflict')

        if not getattr(self, 'pickleFile', False):  # no attribute returns False, == not hasattr(self, 'pickleFile')
            if outputdir is None:
                outputdir = os.path.split(self.indexFiles[0])[0] + '-pickle'
            if not os.path.isdir(outputdir):
                try:
                    os.mkdir(outputdir)
                except OSError:
                    print ('Could not create output directory % for pickle files'
                           % outputdir)
                    raise

            self.pickleFile = os.path.join(outputdir,
                                           'ppdata_f%03d_%02d.pickle'
                                            %(self.funcId, self.dim))

        if getattr(self, 'modsFromPickleVersion', True):
            try:
                if gzipped:
                    if self.pickleFile.find('.gz') < 0:
                        self.pickleFile += '.gz'
                    f = gzip.open(self.pickleFile, 'w')
                else:        
                    f = open(self.pickleFile, 'w') # TODO: what if file already exist?
                pickle.dump(self, f)
                f.close()
                if verbose:
                    print 'Saved pickle in %s.' %(self.pickleFile)
            except IOError, (errno, strerror):
                print "I/O error(%s): %s" % (errno, strerror)
            except pickle.PicklingError:
                print "Could not pickle %s" %(self)
                
    def createDictInstance(self):
        """Returns a dictionary of the instances.

        The key is the instance Id, the value is a list of index.

        """
        dictinstance = {}
        for i in range(len(self.instancenumbers)):
            dictinstance.setdefault(self.instancenumbers[i], []).append(i)

        return dictinstance

    def createDictInstanceCount(self):
        """Returns a dictionary of the instances and their count.
        
        The keys are instance id and the values are the number of
        repetitions of such instance.
        
        """
        return dict((j, self.instancenumbers.count(j)) for j in set(self.instancenumbers))

    def splitByTrials(self, whichdata=None):
        """Splits the post-processed data arrays by trials.

        :keyword string whichdata: either 'evals' or 'funvals'
                                   determines the output
        :returns: this method returns dictionaries of arrays,
                  the key of the dictionaries being the instance id, the
                  value being a smaller post-processed data array
                  corresponding to the instance Id.
                  If whichdata is 'evals' then the array contains
                  function evaluations (1st column is alignment
                  targets).
                  Else if whichdata is 'funvals' then the output data
                  contains function values (1st column is alignment
                  budgets).
                  Otherwise this method returns a tuple of these two
                  arrays in this order.

        """
        dictinstance = self.createDictInstance()
        evals = {}
        funvals = {}

        for instanceid, idx in dictinstance.iteritems():
            evals[instanceid] = self.evals[:,
                                           numpy.ix_(list(i + 1 for i in idx))]
            funvals[instanceid] = self.funvals[:,
                                           numpy.ix_(list(i + 1 for i in idx))]

        if whichdata :
            if whichdata == 'evals':
                return evals
            elif whichdata == 'funvals':
                return funvals

        return (evals, funvals)

    def generateRLData(self, targets):
        """Determine the running lengths for reaching the target values.

        :keyword list targets: target function values of interest

        :returns: dict of arrays, one array for each target. Each array
                  are copied from attribute :py:attr:`evals` of
                  :py:class:`DataSetList`: first element is a target
                  function value smaller or equal to the element of
                  targets considered and has for other consecutive
                  elements the corresponding number of function
                  evaluations.

        """
        res = {}
        # expect evals to be sorted by decreasing function values
        it = reversed(self.evals)
        prevline = numpy.array([-numpy.inf] + [numpy.nan] * self.nbRuns())
        try:
            line = it.next()
        except StopIteration:
            # evals is an empty array
            return res #list()

        for t in sorted(targets):
            while line[0] <= t:
                prevline = line
                try:
                    line = it.next()
                except StopIteration:
                    break
            res[t] = prevline.copy()

        return res
        # return list(res[i] for i in targets)
        # alternative output sorted by targets
        
    def detAverageEvals(self, targets):
        """Determine the average number of f-evals for each target 
        in ``targets`` list. If a target is not reached within trial
        itrail, self.maxevals[itrial] contributes to the average. 
        
        Equals to sum(evals(target)) / nbruns. If ERT is finite 
        this equals to ERT * psucc == (sum(evals) / ntrials / psucc) * psucc, 
        where ERT, psucc, and evals are a function of target.  
          
        """
        assert not any(np.isnan(self.evals[:][0]))  # target value cannot be nan

        evalsums = []
        for evalrow in self.detEvals(targets):
            idxnan = np.isnan(evalrow)
            evalsums.append(sum(evalrow[idxnan==False]) + sum(self.maxevals[idxnan]))
        
        averages = np.array(evalsums, copy=False) / self.nbRuns()
            
        if do_assertion:
            assert all([(ert == np.inf and ps == 0) or toolsdivers.equals_approximately(ert,  averages[i] / ps)
                            for i, (ert, ps) in enumerate(zip(self.detERT(targets), self.detSuccessRates(targets)))]) 
        
        return averages
    
    def detSuccesses(self, targets):
        """Determine for each target in targets the number of 
        successful runs, keeping in return list the order in targets. 
        
            dset.SuccessRates(targets) == np.array(dset.detNbSuccesses(targets)) / dset.nbRuns()
            
        are the respective success rates. 
        
        """
        succ = []
        for evalrow in self.detEvals(targets, copy=False):
            assert len(evalrow) == self.nbRuns()
            succ.append(self.nbRuns() - sum(np.isnan(evalrow)))
        return succ

    def detSuccessRates(self, targets):
        """return a np.array with the success rate for each target 
        in targets, easiest target first
        
        """
        return np.array(self.detSuccesses(targets)) / float(self.nbRuns())

    def detERT(self, targets):
        """Determine the expected running time to reach target values.
        The value is numpy.inf, if the target was never reached. 

        :keyword list targets: target function values of interest

        :returns: list of expected running times (# f-evals) for the
                  respective targets.

        """
        res = {}
        tmparray = numpy.vstack((self.target, self.ert)).transpose()
        it = reversed(tmparray)
        # expect this array to be sorted by decreasing function values

        prevline = numpy.array([-numpy.inf, numpy.inf])
        try:
            line = it.next()
        except StopIteration:
            # evals is an empty array
            return list()

        for t in sorted(targets):
            while line[0] <= t:
                prevline = line
                try:
                    line = it.next()
                except StopIteration:
                    break
            res[t] = prevline.copy() # is copy necessary? Yes. 

        # Return a list of ERT corresponding to the input targets in
        # targets, sorted along targets
        return list(res[i][1] for i in targets)

    def detEvals(self, targets, copy=True):
        """returns len(targets) data rows self.evals[idata, 1:] each row with 
        the closest but not larger target such that self.evals[idata, 0] <= target, 
        and self.evals[idata-1, 0] > target or in the "limit" cases the
        idata==0 line or the line np.array(self.nbRuns() * [np.nan]). 
        
        Makes by default a copy of the data, however this might change in
        future.
        
        """
        evalsrows = {}  # data rows, easiest target first
        idata = self.evals.shape[0] - 1  # current data line index 
        for target in sorted(targets):  # smallest most difficult target first
            if self.evals[-1, 0] > target:  # last entry is worse than target
                evalsrows[target] = np.array(self.nbRuns() * [np.nan])
                continue
            while idata > 0 and self.evals[idata - 1, 0] <= target:  # idata-1 line is good enough
                idata -= 1  # move up
            assert self.evals[idata, 0] <= target and (idata == 0 or self.evals[idata - 1, 0] > target)
            evalsrows[target] = self.evals[idata, 1:].copy() if copy else self.evals[idata, 1:]
            
        if do_assertion:
            assert all([all((np.isnan(evalsrows[target]) + (evalsrows[target] == self._detEvals2(targets)[i]))) for i, target in enumerate(targets)])
    
        return [evalsrows[t] for t in targets]
        
    def _detEvals2(self, targets):
        """Determine the number of evaluations to reach target values.

        :keyword seq or float targets: target precisions
        :returns: list of len(targets) values, each being an array of nbRuns FEs values

        """
        tmp = {}  # dict of eval-arrays (of size Ntrials), each entry for a given target
        # expect evals to be sorted by decreasing function values
        it = reversed(self.evals)
        prevline = numpy.array([-numpy.inf] + [numpy.nan] * self.nbRuns())
        try:
            line = it.next()
        except StopIteration:
            # evals is an empty array
            return tmp #list()

        for t in sorted(targets):
            while line[0] <= t:
                prevline = line
                try:
                    line = it.next()
                except StopIteration:
                    break
            tmp[t] = prevline.copy()

        return list(tmp[i][1:] for i in targets)

    def plot(self):
        for evals in self.evals[:, 1:].transpose(): # loop over the rows of the transposed array
            idx = self.evals[:, 0] > 0
            # plt.semilogx(self.evals[idx, 0], evals[idx])
            plt.loglog(self.evals[idx, 0], evals[idx])
            plt.gca().invert_xaxis()
            plt.xlabel('target $\Delta f$ value')
            plt.ylabel('number of function evaluations')
        return plt.gca()


class DataSetList(list):
    """List of instances of :py:class:`DataSet`.

    This class implements some useful slicing functions.

    Also it will merge data of DataSet instances that are identical
    (according to function __eq__ of DataSet).

    """
    #Do not inherit from set because DataSet instances are mutable which means
    #they might change over time.

    def __init__(self, args=[], verbose=False, check_data_type=True):
        """Instantiate self from a list of folder- or filenames or 
        ``DataSet`` instances.

        :keyword list args: strings being either info file names, folder
                            containing info files or pickled data files,
                            or a list of DataSets.
        :keyword bool verbose: controls verbosity.

        Exceptions:
        Warning -- Unexpected user input.
        pickle.UnpicklingError

        """


        if not args:
            super(DataSetList, self).__init__()
            return

        if isinstance(args, basestring):
            args = [args]

        if len(args) and (isinstance(args[0], DataSet) or
                not check_data_type and hasattr(args[0], 'algId')):
            # TODO: loaded instances are not DataSets but
            # ``or hasattr(args[0], 'algId')`` fails in self.append
            # initialize a DataSetList from a sequence of DataSet
            for ds in args:
                self.append(ds, check_data_type)
            return

        fnames = []
        for name in args:
            if isinstance(name, basestring) and findfiles.is_recognized_repository_filetype(name):
                fnames.extend(findfiles.main(name, verbose))
            else:
                fnames.append(name)
        for name in fnames: 
            if isinstance(name, DataSet):
                self.append(name)
            elif name.endswith('.info'):
                self.processIndexFile(name, verbose)
            elif name.endswith('.pickle') or name.endswith('.pickle.gz'):
                try:
                    # cocofy(name)
                    if name.endswith('.gz'):
                        f = gzip.open(name)
                    else:
                        f = open(name,'r')
                    try:
                        entry = pickle.load(f)
                    except pickle.UnpicklingError:
                        print '%s could not be unpickled.' %(name)
                    f.close()
                    if verbose > 1:
                        print 'Unpickled %s.' % (name)
                    try:
                        entry.instancenumbers = entry.itrials  # has been renamed
                        del entry.itrials
                    except:
                        pass
                    # if not hasattr(entry, 'detAverageEvals')
                    self.append(entry)
                    #set_trace()
                except IOError, (errno, strerror):
                    print "I/O error(%s): %s" % (errno, strerror)
            else:
                s = ('File or folder ' + name + ' not found. ' +
                              'Expecting as input argument either .info ' +
                              'file(s), .pickle file(s) or a folder ' +
                              'containing .info file(s).')
                warnings.warn(s)
                print s
            self.sort()
            
    def processIndexFile(self, indexFile, verbose=True):
        """Reads in an index (.info?) file information on the different runs."""

        try:
            f = open(indexFile)
            if verbose:
                print 'Processing %s.' % indexFile

            # Read all data sets within one index file.
            nbLine = 1
            while True:
                try:
                    header = f.next()
                    while not header.strip(): # remove blank lines
                        header = f.next()
                        nbLine += 1
                    comment = f.next()
                    if not comment.startswith('%'):
                        warnings.warn('Entry in file %s at line %d is faulty: '
                                      % (indexFile, nbLine) +
                                      'it will be skipped.')
                        nbLine += 2
                        continue
                    data = f.next()
                    nbLine += 3
                    #TODO: check that something is not wrong with the 3 lines.
                    self.append(DataSet(header, comment, data, indexFile,
                                        verbose))
                except StopIteration:
                    break
            # Close index file
            f.close()

        except IOError:
            print 'Could not open %s.' % indexFile

    def append(self, o, check_data_type=False):
        """Redefines the append method to check for unicity."""

        if check_data_type and not isinstance(o, DataSet):
            warnings.warn('appending a non-DataSet to the DataSetList')
            raise Exception('Expect DataSet instance.')
        isFound = False
        for i in self:
            if i == o:
                isFound = True
                if i.instancenumbers == o.instancenumbers:
                    warnings.warn("same DataSet found twice, second one from " + str(o.indexFiles) + " is disregarded")
                    break
                if set(i.instancenumbers).intersection(o.instancenumbers):
                    warnings.warn('instances ' + str(set(i.instancenumbers).intersection(o.instancenumbers))
                                  + (' found several times. Read data for F%d in %d-D' % (i.funcId, i.dim)) 
                                  # + ' found several times. Read data for F%(argone)d in %(argtwo)d-D ' % {'argone':i.funcId, 'argtwo':i.dim}
                                  + 'are likely to be inconsistent. ')
                # tmp = set(i.dataFiles).symmetric_difference(set(o.dataFiles))
                #Check if there are new data considered.
                if 1 < 3:
                    i.dataFiles.extend(o.dataFiles)
                    i.indexFiles.extend(o.indexFiles)
                    i.funvals = alignArrayData(VArrayMultiReader([i.funvals, o.funvals]))
                    i.finalfunvals = numpy.r_[i.finalfunvals, o.finalfunvals]
                    i.evals = alignArrayData(HArrayMultiReader([i.evals, o.evals]))
                    i.maxevals = numpy.r_[i.maxevals, o.maxevals]
                    i.computeERTfromEvals()
                    if getattr(i, 'pickleFile', False):
                        i.modsFromPickleVersion = True

                    for j in dir(i):
                        if isinstance(getattr(i, j), list):
                            getattr(i, j).extend(getattr(o, j))

                else:
                    if getattr(i, 'pickleFile', False):
                        i.modsFromPickleVersion = False
                    elif getattr(o, 'pickleFile', False):
                        i.modsFromPickleVersion = False
                        i.pickleFile = o.pickleFile
                break
        if not isFound:
            list.append(self, o)

    def extend(self, o):
        """Extend with elements.

        This method is implemented to prevent problems since append was
        superseded. This method could be the origin of efficiency issue.

        """
        for i in o:
            self.append(i)

    def pickle(self, *args, **kwargs):
        """Loop over self to pickle each element."""
        for i in self:
            i.pickle(*args, **kwargs)

    def dictByAlg(self):
        """Returns a dictionary of instances of this class by algorithm.

        The resulting dict uses algId and comment as keys and the
        corresponding slices as values.

        """
        d = DictAlg()
        for i in self:
            d.setdefault((i.algId, i.comment), DataSetList()).append(i)
        return d

    def dictByDim(self):
        """Returns a dictionary of instances of this class by dimensions.

        Returns a dictionary with dimension as keys and the
        corresponding slices as values.

        """
        d = {}
        for i in self:
            d.setdefault(i.dim, DataSetList()).append(i)
        return d

    def dictByFunc(self):
        """Returns a dictionary of instances of this class by functions.

        Returns a dictionary with the function id as keys and the
        corresponding slices as values.

        """
        d = {}
        for i in self:
            d.setdefault(i.funcId, DataSetList()).append(i)
        return d

    def dictByDimFunc(self):
        """Returns a dictionary of instances of this class 
        by dimensions and for each dimension by function.

        Returns a dictionary with dimension as keys and the
        corresponding slices as values.
        
            ds = dsl.dictByDimFunc[40][2]  # DataSet dimension 40 on F2

        """
        dsld = self.dictByDim()
        for k in dsld:
            dsld[k] = dsld[k].dictByFunc()
        return dsld
        
    def dictByNoise(self):
        """Returns a dictionary splitting noisy and non-noisy entries."""
        sorted = {}
        for i in self:
            if i.funcId in range(1, 25):
                sorted.setdefault('noiselessall', DataSetList()).append(i)
            elif i.funcId in range(101, 131):
                sorted.setdefault('nzall', DataSetList()).append(i)
            else:
                warnings.warn('Unknown function id.')

        return sorted

    def dictByFuncGroup(self):
        """Returns a dictionary of instances of this class by function groups.

        The output dictionary has function group names as keys and the
        corresponding slices as values. Current groups are based on the
        GECCO-BBOB 2009-2013 function testbeds. 

        """
        sorted = {} 
        for i in self:
            if i.funcId in range(1, 6):
                sorted.setdefault('separ', DataSetList()).append(i)
            elif i.funcId in range(6, 10):
                sorted.setdefault('lcond', DataSetList()).append(i)
            elif i.funcId in range(10, 15):
                sorted.setdefault('hcond', DataSetList()).append(i)
            elif i.funcId in range(15, 20):
                sorted.setdefault('multi', DataSetList()).append(i)
            elif i.funcId in range(20, 25):
                sorted.setdefault('mult2', DataSetList()).append(i)
            elif i.funcId in range(101, 107):
                sorted.setdefault('nzmod', DataSetList()).append(i)
            elif i.funcId in range(107, 122):
                sorted.setdefault('nzsev', DataSetList()).append(i)
            elif i.funcId in range(122, 131):
                sorted.setdefault('nzsmm', DataSetList()).append(i)
            else:
                warnings.warn('Unknown function id.')

        return sorted

    def dictByParam(self, param):
        """Returns a dictionary of DataSetList by parameter values.

        :returns: a dictionary with values of parameter param as keys and
                  the corresponding slices of DataSetList as values.

        """

        d = {}
        for i in self:
            d.setdefault(getattr(i, param, None), DataSetList()).append(i)
        return d

    def info(self, opt=None):
        """Display some information onscreen.

        :keyword string opt: changes size of output, can be 'all'
                             (default), 'short'

        """
        #TODO: do not integrate over dimension!!!
        #      loop over all data sets!?

        if len(self) > 0:
            print '%d data set(s)' % (len(self))
            dictAlg = self.dictByAlg()
            algs = dictAlg.keys()
            algs.sort()
            sys.stdout.write('Algorithm(s): %s' % (algs[0][0]))
            for i in range(1, len(algs)):
                sys.stdout.write(', %s' % (algs[0][0]))
            sys.stdout.write('\n')

            dictFun = self.dictByFunc()
            functions = dictFun.keys()
            functions.sort()
            nbfuns = len(set(functions))
            splural = 's' if nbfuns > 1 else ''
            print '%d Function%s with ID%s %s' % (nbfuns, splural, splural, consecutiveNumbers(functions))

            dictDim = self.dictByDim()
            dimensions = dictDim.keys()
            dimensions.sort()
            sys.stdout.write('Dimension(s): %d' % (dimensions[0]))
            for i in range(1, len(dimensions)):
                sys.stdout.write(', %d' % (dimensions[i]))
            sys.stdout.write('\n')

            maxevals = []
            for i in xrange(len(dimensions)):
                maxeval = []
                for d in dictDim[dimensions[i]]:
                    maxeval = int(max((d.mMaxEvals(), maxeval)))
                maxevals.append(maxeval)
            print 'Max evals: %s' % str(maxevals)

            if opt == 'all':
                print 'Df      |     min       10      med       90      max'
                print '--------|--------------------------------------------'
                evals = list([] for i in targets_displayed_for_info)
                for i in self:
                    tmpevals = i.detEvals(targets_displayed_for_info)
                    for j in range(len(targets_displayed_for_info)):
                        evals[j].extend(tmpevals[j])
                for i, j in enumerate(targets_displayed_for_info): # never aggregate over dim...
                    tmp = toolsstats.prctile(evals[i], [0, 10, 50, 90, 100])
                    tmp2 = []
                    for k in tmp:
                        if not numpy.isfinite(k):
                            tmp2.append('%8f' % k)
                        else:
                            tmp2.append('%8d' % k)
                    print '%2.1e |%s' % (j, ' '.join(tmp2))

            # display distributions of final values
        else:
            print self

    def sort(self, key1='dim', key2='funcId'):
        def cmp_fun(a, b):
            if getattr(a, key1) == getattr(b, key1):
                if getattr(a, key2) == getattr(b, key2):
                    return 0
                return 1 if getattr(a, key2) > getattr(b, key2) else -1                
            else:
                return 1 if getattr(a, key1) > getattr(b, key1) else -1
        sorted_self = list(sorted(self, cmp=cmp_fun))
        for i, ds in enumerate(sorted_self):
            self[i] = ds
        return self
    
        # interested in algorithms, number of datasets, functions, dimensions
        # maxevals?, funvals?, success rate?


    def run_length_distributions(self, dimension, target_values,
                                 fun_list=None,
                                 reference_data_set_list=None,
                                 reference_scoring_function=lambda x: toolsstats.prctile(x, [5])[0],
                                 data_per_target=15,
                                 flatten_output_dict=True):
        """return a dictionary with an entry for each algorithm (or
        the dictionary value for only one algorithm if
        ``flatten_output_dict is True``) and
        the left envelope rld-array. For each algorithm:
        a sorted rld-array of evaluations to reach the targets on all
        functions in ``func_list`` or all available functions, the
        list of solved functions, the list of processed functions. If
        the sorted rld-array is normalized by the reference score (after
        sorting), the last entry is the original rld.

        TODO: change interface to return always rld_original and optional
        the scores to compare with. Example::

            rld = dsl.run_length_distributions(...)
            semilogy([x if np.isfinite(x) else np.inf
                      for x in rld[0][0] / rld[0][-1]])

        If ``reference_data_set_list is not None`` evaluations
        are normalized by the reference data, however
        the data remain to be sorted without normalization.
        TODO:

        """
        dsl_dict = self.dictByDim()[dimension].dictByAlg()
        # selected dimension and go by algorithm
        rld_dict = {}  # result for each algorithm
        reference_scores = {}  # for each funcId a list of len(target_values)
        left_envelope = np.inf
        for alg in dsl_dict:
            rld_data = []  # 15 evaluation-counts per function and target
            ref_scores = []  # to compute rld_data / ref_scores element-wise
            funcs_processed = []
            funcs_solved = []
            for ds in dsl_dict[alg]:  # ds is a DataSet, i.e. typically 15 trials
                if fun_list and ds.funcId not in fun_list:
                    continue
                assert dimension == ds.dim
                funcs_processed.append(ds.funcId)
                evals = ds.detEvals(target_values((ds.funcId, ds.dim)))
                if data_per_target is not None:
                    evals = [toolsstats.fix_data_number(d, data_per_target)
                                for d in evals]
                    # make sure to get 15 numbers for each target
                if reference_data_set_list is not None:
                    if ds.funcId not in reference_scores:
                        reference_scores[ds.funcId] \
                            = reference_data_set_list.det_best_data_lines(
                                target_values((ds.funcId, ds.dim)),
                                ds.funcId, ds.dim,
                                reference_scoring_function)[1]
                        # value checking, could also be done later
                        for i, val in enumerate(reference_scores[ds.funcId]):
                            if not np.isfinite(val) and any(np.isfinite(evals[i])):
                                raise ValueError('reference_value is not finite')
                                # a possible solution would be to set ``val = 1``
                    ref_scores.append(np.hstack([data_per_target * [val]
                                                 for val in reference_scores[ds.funcId]]))
                    # 'needs to be checked', qqq
                evals = np.hstack(evals)  # "stack" len(targets) * 15 values
                if any(np.isfinite(evals)):
                    funcs_solved.append(ds.funcId)
                rld_data.append(evals)

            funcs_processed.sort()
            funcs_solved.sort()
            assert np.__version__ >= '1.4.0'
            # if this fails, replacing nan with inf might work for sorting
            rld_data = np.hstack(rld_data)
            if reference_data_set_list is not None:
                ref_scores = np.hstack(ref_scores)
                idx = rld_data.argsort()
                rld_original = rld_data[idx]
                rld_data = rld_original / ref_scores[idx]
            else:
                rld_data.sort()  # returns None
            # nan are at the end now (since numpy 1.4.0)

            # check consistency
            if len(funcs_processed) > len(set(funcs_processed)):
                warnings.warn("function processed twice "
                              + str(funcs_processed))
                raise ValueError("function processed twice")
            if fun_list is not None and set(funcs_processed) != set(fun_list):
                warnings.warn("not all functions found for " + str(alg)
                    + " and computations disregarded " + str(ds.algId))
                continue

            left_envelope = np.fmin(left_envelope, rld_data)
            # fails if number of computed data are different
            rld_dict[alg] = [rld_data,
                             sorted(funcs_solved),
                             funcs_processed]
            if reference_data_set_list is not None:
                rld_dict[alg].append(rld_original)

        for k, v in rld_dict.items():
            assert v[2] == funcs_processed

        if flatten_output_dict and len(rld_dict) == 1:
            return rld_dict.values()[0], left_envelope
        return rld_dict, left_envelope

    def get_all_data_lines(self, target_value, fct, dim):
        """return a list of all data lines in ``self`` for each
        algorithm and a list of the respective
        computed ERTs.

        Example
        -------
        Get all run lengths of all trials on f1 in 20-D to reach
        target 1e-7::

            data = dsl.get_all_data_lines(1e-7, 1, 20)[0]
            flat_data = np.hstack(data)
            plot(np.arange(1, 1+len(flat_data)) / len(flat_data),
                 sort(flat_data))  # sorted fails on nan

        """
        # TODO: make sure to get the same amount of data for each
        # algorithm / target!?
        assert np.isscalar(target_value)

        lines = []
        scores = []
        for ds in self:
            if ds.funcId != fct or ds.dim != dim:
                continue
            lines.append(ds.detEvals([target_value])[0])
            scores.append(ds.detERT([target_value])[0])

        return lines, scores

    def det_best_data_lines(self, target_values, fct, dim,
                           scoring_function=None):
        """return a list of the respective best data lines over all data
        sets in ``self`` for each ``target in target_values`` and an
        array of the computed scores (ERT ``if scoring_function is None``).

        If ``scoring_function is None``, the best is determined with method
        ``detERT``. Using ``scoring_function=lambda x:
        toolsstat.prctile(x, [5], ignore_nan=False)`` is another useful
        alternative.

        """
        try:
            target_values = target_values((fct, dim))
        except TypeError:
            target_values = target_values
        best_scores = np.array(len(target_values) * [np.inf])
        best_lines = len(target_values) * [None]
        for ds in self:
            if ds.funcId != fct or ds.dim != dim:
                continue
            current_lines = ds.detEvals(target_values)
            if scoring_function is None:
                current_scores = ds.detERT(target_values)
            else:
                current_scores = np.array([scoring_function(d)
                                           for d in current_lines], copy=False)
            assert len(current_lines) == len(current_scores) \
                     == len(best_scores) == len(best_lines)
            for i in np.where(toolsdivers.less(current_scores, best_scores))[0]:
                best_lines[i] = current_lines[i]
                best_scores[i] = current_scores[i]

        if any(line is None for line in best_lines):
            warnings.warn('best data lines not determined, (f, dim)='
                          + str((fct, dim)))
        return best_lines, best_scores

    def get_sorted_algorithms(self, dimension, target_values,
                              fun_list=None,
                              reference_dataset_list=None,
                              smallest_evaluation_to_use=3):
        """return list of the algorithms from ``self``, sorted by
        minimum loss factor in the ECDF.

        Best means to be within `loss` of the best algorithm at
        at least one point of the ECDF from the functions `fun_list`,
        i.e. minimal distance to the left envelope in the semilogx plot.

        ``target_values`` gives for each function-dimension pair a list of
        target values.

        TODO: data generation via run_length_distributions and sorting
        should probably be separated.

        """
        rld, left_envelope = self.run_length_distributions(
            dimension, target_values, fun_list,
            reference_data_set_list=reference_dataset_list)
        best_algorithms = []
        idx = left_envelope >= smallest_evaluation_to_use
        for alg in rld:
            try:
                if reference_dataset_list:
                    best_algorithms.append([alg, toolsstats.prctile(rld[alg][0][idx], [5])[0],
                                            rld[alg], left_envelope,
                                            toolsstats.prctile(rld[alg][0], [2, 5, 15, 25, 50], ignore_nan=False)])
                else:
                    best_algorithms.append([alg, np.nanmin(rld[alg][0][idx] / left_envelope[idx]),
                                            rld[alg], left_envelope,
                                            toolsstats.prctile(rld[alg][0] / left_envelope, [2, 5, 15, 25, 50], ignore_nan=False)])
            except ValueError:
                warnings.warn(str(alg) + ' could not be processed for get_sorted_algorithms ')

        best_algorithms.sort(key=lambda x: x[1])
        return best_algorithms


def parseinfoold(s):
    """Deprecated: Extract data from a header line in an index entry.

    Older but verified version of :py:meth:`parseinfo`

    The header line should be a string of comma-separated pairs of
    key=value, for instance: key = value, key = 'value'

    Keys should not use comma or quote characters.

    """
    # TODO should raise some kind of error...
    # Split header into a list of key-value
    list_kv = s.split(', ')

    # Loop over all elements in the list and extract the relevant data.
    # We loop backward to make sure that we did not split inside quotes.
    # It could happen when the key algId and the value is a string.
    p = re.compile('([^,=]+)=(.*)')
    list_kv.reverse()
    it = iter(list_kv)
    res = []
    while True:
        try:
            elem = it.next()
            tmp = p.match(elem)
            while not tmp:
                elem = it.next() + elem
                # add up elements of list_kv until we have a whole key = value

            elem0, elem1 = tmp.groups()
            elem0 = elem0.strip()
            elem1 = elem1.strip()
            if elem1.startswith('\'') and elem1.endswith('\''): # HACK
                elem1 = ('\'' + re.sub('([\'])', r'\\\1', elem1[1:-1]) + '\'')
            elem1 = ast.literal_eval(elem1)
            res.append((elem0, elem1))
        except StopIteration:
            break

    return res

def parseinfo(s):
    """Extract data from a header line in an index entry.

    Use a 'smarter' regular expression than :py:meth:`parseinfoold`.
    The header line should be a string of comma-separated pairs of
    key=value, for instance: key = value, key = 'value'

    Keys should not use comma or quote characters.

    """
    p = re.compile('\ *([^,=]+?)\ *=\ *(".+?"|\'.+?\'|[^,]+)\ *(?=,|$)')
    res = []
    for elem0, elem1 in p.findall(s):
        if elem1.startswith('\'') and elem1.endswith('\''): # HACK
            elem1 = ('\'' + re.sub(r'(?<!\\)(\')', r'\\\1', elem1[1:-1]) + '\'')
        elem1 = ast.literal_eval(elem1)
        res.append((elem0, elem1))
    return res


def set_unique_algId(ds_list, ds_list_reference, taken_ids=None):
    """on return, elements in ``ds_list`` do not have an ``algId``
    attribute value from ``taken_ids`` or from
    ``ds_list_reference`` ``if taken_ids is None``.

    In case, ``BFGS`` becomes ``BFGS 2`` etc.

    """
    if taken_ids is None:
        taken_ids = []
        for ds in ds_list_reference:
            taken_ids.append(ds.algId)
    taken_ids = set(taken_ids)

    for ds in ds_list:
        if ds.algId in taken_ids:
            algId = ds.algId
            i = 2
            while algId + ' ' + str(i) in taken_ids:
                i += 1
            ds.algId = algId + ' ' + str(i)

def processInputArgs(args, verbose=True):
    """Process command line arguments.

    Returns several instances of :py:class:`DataSetList`, and a list of 
    algorithms from a list of strings representing file and folder names,
    see below for details. This command operates folder-wise: one folder 
    corresponds to one algorithm.

    It is recommended that if a folder listed in args contain both
    :file:`info` files and the associated :file:`pickle` files, they be
    kept in different locations for efficiency reasons.

    :keyword list args: string arguments for folder names
    :keyword bool verbose: controlling verbosity

    :returns (dsList, sortedAlgs, dictAlg):
      dsList
        is a list containing all DataSet instances, this is to
        prevent the regrouping done in instances of DataSetList.
        Caveat: algorithms with the same name are overwritten!?
      dictAlg
        is a dictionary which associates algorithms to an instance
        of DataSetList,
      sortedAlgs
        is a list of keys of dictAlg with the ordering as
        given by the input argument args.

    """
    dsList = list()
    sortedAlgs = list()
    dictAlg = {}
    for i in args:
        i = i.strip()
        if i == '': # might cure an lf+cr problem when using cywin under Windows
            continue
        if findfiles.is_recognized_repository_filetype(i):
            filelist = findfiles.main(i, verbose)
            #Do here any sorting or filtering necessary.
            #filelist = list(i for i in filelist if i.count('ppdata_f005'))
            tmpDsList = DataSetList(filelist, verbose)
            #Nota: findfiles will find all info AND pickle files in folder i.
            #No problem should arise if the info and pickle files have
            #redundant information. Only, the process could be more efficient
            #if pickle files were in a whole other location.

            set_unique_algId(tmpDsList, dsList)
            dsList.extend(tmpDsList)
            #alg = os.path.split(i.rstrip(os.sep))[1]  # trailing slash or backslash
            #if alg == '':
            #    alg = os.path.split(os.path.split(i)[0])[1]
            alg = i.rstrip(os.path.sep)
            print '  using:', alg

            # Prevent duplicates
            if all(i != alg for i in sortedAlgs):
                sortedAlgs.append(alg)
                dictAlg[alg] = tmpDsList
        elif os.path.isfile(i):
            # TODO: a zipped tar file should be unzipped here, see findfiles.py 
            txt = 'The post-processing cannot operate on the single file ' + str(i)
            warnings.warn(txt)
            continue
        else:
            txt = "Input folder '" + str(i) + "' could not be found."
            raise Exception(txt)

    return dsList, sortedAlgs, dictAlg
    
class DictAlg(dict):
    def __init__(self, d={}):
        dict.__init__(self, d)  # was: super.__init(d)
        
    def by_dim(self):
        return dictAlgByDim(self) # TODO: put here actual implementation

def dictAlgByDim(dictAlg):
    """Returns a dictionary with problem dimension as key from
    a dictionary of DataSet lists. 

    The input argument is a dictionary with algorithm names as 
    keys and a list of :py:class:`DataSet` instances as values.
    The resulting dictionary will have dimension as key and as values
    dictionaries with algorithm names as keys.

    """
    # should become: 
    # return DictAlg(dictAlg).by_dim()
    
    res = {}

    # get the set of problem dimensions
    dims = set()
    tmpdictAlg = {} # will store DataSet by alg and dim
    for alg, dsList in dictAlg.iteritems():
        tmp = dsList.dictByDim()
        tmpdictAlg[alg] = tmp
        dims |= set(tmp.keys())

    for d in dims:
        for alg in dictAlg:
            tmp = DataSetList()
            try:
                tmp = tmpdictAlg[alg][d]
            except KeyError:
                txt = ('No data for algorithm %s in %d-D.'
                       % (alg, d))
                warnings.warn(txt)

            if res.setdefault(d, {}).has_key(alg):
                txt = ('Duplicate data for algorithm %s in %d-D.'
                       % (alg, d))
                warnings.warn(txt)

            res.setdefault(d, {}).setdefault(alg, tmp)
            # Only the first data for a given algorithm in a given dimension

    #for alg, dsList in dictAlg.iteritems():
        #for i in dsList:
            #res.setdefault(i.dim, {}).setdefault(alg, DataSetList()).append(i)

    return res

def dictAlgByDim2(dictAlg, remove_empty=False):
    """Returns a dictionary with problem dimension as key.

    The difference with :py:func:`dictAlgByDim` is that there is an
    entry for each algorithm even if the resulting
    :py:class:`DataSetList` is empty.

    This function is meant to be used with an input argument which is a
    dictionary with algorithm names as keys and which has list of
    :py:class:`DataSet` instances as values.
    The resulting dictionary will have dimension as key and as values
    dictionaries with algorithm names as keys.

    """
    res = {}

    for alg, dsList in dictAlg.iteritems():
        for i in dsList:
            res.setdefault(i.dim, {}).setdefault(alg, DataSetList()).append(i)

    if remove_empty:
        raise NotImplementedError
        for dim, ds_dict in res.iteritems():
            for alg, ds_dict2 in ds_dict.iteritems():
                if not len(ds_dict2):
                    pass
            if not len(ds_dict):
                pass

    return res

def dictAlgByFun(dictAlg):
    """Returns a dictionary with function id as key.

    This method is meant to be used with an input argument which is a
    dictionary with algorithm names as keys and which has list of
    :py:class:`DataSet` instances as values.
    The resulting dictionary will have function id as key and as values
    dictionaries with algorithm names as keys.

    """
    res = {}
    funcs = set()
    tmpdictAlg = {}
    for alg, dsList in dictAlg.iteritems():
        tmp = dsList.dictByFunc()
        tmpdictAlg[alg] = tmp
        funcs |= set(tmp.keys())

    for f in funcs:
        for alg in dictAlg:
            tmp = DataSetList()
            try:
                tmp = tmpdictAlg[alg][f]
            except KeyError:
                txt = ('No data for algorithm %s on function %d.'
                       % (alg, f)) # This message is misleading.
                warnings.warn(txt)

            if res.setdefault(f, {}).has_key(alg):
                txt = ('Duplicate data for algorithm %s on function %d-D.'
                       % (alg, f))
                warnings.warn(txt)

            res.setdefault(f, {}).setdefault(alg, tmp)
            # Only the first data for a given algorithm in a given dimension

    return res

def dictAlgByNoi(dictAlg):
    """Returns a dictionary with noise group as key.

    This method is meant to be used with an input argument which is a
    dictionary with algorithm names as keys and which has list of
    :py:class:`DataSet` instances as values.
    The resulting dictionary will have a string denoting the noise group
    ('noiselessall' or 'nzall') and as values dictionaries with
    algorithm names as keys.

    """
    res = {}
    ng = set()
    tmpdictAlg = {}
    for alg, dsList in dictAlg.iteritems():
        tmp = dsList.dictByNoise()
        tmpdictAlg[alg] = tmp
        ng |= set(tmp.keys())

    for n in ng:
        stmp = ''
        if n == 'nzall':
            stmp = 'noisy'
        elif n == 'noiselessall':
            stmp = 'noiseless'

        for alg in dictAlg:
            tmp = DataSetList()
            try:
                tmp = tmpdictAlg[alg][n]
            except KeyError:
                txt = ('No data for algorithm %s on %s function.'
                       % (alg, stmp))
                warnings.warn(txt)

            if res.setdefault(n, {}).has_key(alg):
                txt = ('Duplicate data for algorithm %s on %s functions.'
                       % (alg, stmp))
                warnings.warn(txt)

            res.setdefault(n, {}).setdefault(alg, tmp)
            # Only the first data for a given algorithm in a given dimension

    return res

def dictAlgByFuncGroup(dictAlg):
    """Returns a dictionary with function group as key.

    This method is meant to be used with an input argument which is a
    dictionary with algorithm names as keys and which has list of
    :py:class:`DataSet` instances as values.
    The resulting dictionary will have a string denoting the function
    group and as values dictionaries with algorithm names as keys.

    """
    res = {}
    fg = set()
    tmpdictAlg = {}
    for alg, dsList in dictAlg.iteritems():
        tmp = dsList.dictByFuncGroup()
        tmpdictAlg[alg] = tmp
        fg |= set(tmp.keys())  # | is bitwise OR

    for g in fg:
        for alg in dictAlg:
            tmp = DataSetList()
            try:
                tmp = tmpdictAlg[alg][g]
            except KeyError:
                txt = ('No data for algorithm %s on %s functions.'
                       % (alg, g))
                warnings.warn(txt)

            if res.setdefault(g, {}).has_key(alg):
                txt = ('Duplicate data for algorithm %s on %s functions.'
                       % (alg, g))
                warnings.warn(txt)

            res.setdefault(g, {}).setdefault(alg, tmp)
            # Only the first data for a given algorithm in a given dimension

    return res

# TODO: these functions should go to different modules. E.g. tools.py and toolsstats.py renamed as stats.py

