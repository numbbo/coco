#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Various tools. 

"""
from __future__ import absolute_import

import os
import numpy as np
import warnings

class TargetValues(object):
    """should go in a different module...
    Working modes:
    
        targets = TargetValues(reference_data).set_targets(ERT_values)
        targets((1, 10)) # returns the list of f-values for F1 in 10-D
    and
        targets = TargetValues().set_targets([10, 1, 1e-1, 1e-3, 1e-5, 1e-8])
        targets((1, 10))  # returns the above targets ignoring the input argument
        
        TODO: see compall/determineFtarget2.FunTarget
    
    """
    def __init__(self, reference_data_set=None):
        """Two modes of working: without `reference_data_set`, 
        a target function value list is returned on calling
        the instance. Otherwise target values will be computed 
        based on the reference data in the given function 
        and dimension. 
        
        """
        self.ref_data = reference_data_set
    
    def __call__(self, fun_dim=None, force_different_targets=True):
        """Get the target values for the respective function and dimension  
        and the reference ERT values set via ``set_targets``. `fun_dim` is 
        a tuple ``(fun_nb, dimension)`` like ``(1, 20)`` for the 20-D sphere. 
        
        Details: with ``force_different_targets is True`` the method relies 
        on the fixed target value "difference" of ``10**0.2``. 
        
        """
        if self.ref_data is None:
            return self.input_targets  # in this case ftarget values
        elif fun_dim is None:
            raise ValueError('function and dimension must be given via input argument ``fun_dim``')
        elif self.ref_data == 'bestGECCO2009':
            from bbob_pproc import bestalg
            bestalg.loadBBOB2009() # this is an absurd interface
            self.ref_data = bestalg.bestalgentries2009
        if force_different_targets and len(self.run_lengths) > 15:
                warnings.warn('more than 15 run_length targets are in use while enforcing different target values, which might not lead to the desired result')

        delta_f_factor = 10**0.2
        smallest_target = 1e-8  # true for all experimental setups at least until 2013
            
        fun_dim = tuple(fun_dim)
        dim_fun = tuple([i for i in reversed(fun_dim)])
        ds = self.ref_data[dim_fun]
        try:
            end = np.nonzero(ds.target >= smallest_target)[0][-1] + 1
        except IndexError:
            end = len(ds.target)
        for begin in xrange(len(ds.target)-1):
            if ds.ert[begin+1] > 1:
                break
        try: 
            # make sure the necessary attributs do exist
            assert ds.ert[begin] == 1  # we might have to compute these the first time
            # check whether there are gaps between the targets 
            assert all(equals_approximately(delta_f_factor, ds.target[i] / ds.target[i+1]) for i in xrange(begin, end-1))
            # if this fails, we need to insert the missing target values 
        except AssertionError:
            raise NotImplementedError
        assert len(ds.ert) == len(ds.target)
        
        targets = []
        for rl in reversed(self.run_lengths):
            indices = np.nonzero(ds.ert[begin:end] <= np.max((1, rl * (fun_dim[1] if self.times_dimension else 1))))[0]
            assert len(indices)
            targets.append(ds.target[indices[-1]])
            if force_different_targets and len(targets) > 1 and not targets[-1] < targets[-2]:
                targets[-1] = targets[-2] / delta_f_factor
        return targets
    
    get_targets = __call__  # an alias

    def set_targets(self, values, times_dimension=True):
        """target values are either run_lengths of the reference algorithm
        (in case the reference algorithm was given at class instantiation) 
        or target values. 
        
        """
        
        if self.ref_data is None:
            self.input_targets = sorted(values, reverse=True)
        else:
            self.run_lengths = sorted(values)
            self.times_dimension = times_dimension
        return self
                
    def _generate_erts(self, ds, target_values):
        """compute for all target values, starting with 1e-8, the ert value
        and store it in the reference_data_set attribute
        
        """
        raise NotImplementedError
              
    def labels(self):
        pass
    
def equals_approximately(a, b, eps=1e-12):
    if a < 0:
        a, b = -1 * a, -1 * b
    return a - eps < b < a + eps or (1 - eps) * a < b < (1 + eps) * a
 
def prepend_to_file(filename, lines, maxlines=1000, warn_message=None):
    """"prepend lines the tex-command filename """
    try:
        lines_to_append = list(open(filename, 'r'))
    except IOError:
        lines_to_append = []
    f = open(filename, 'w')
    for line in lines:
        f.write(line + '\n')
    for i, line in enumerate(lines_to_append):
        f.write(line)
        if i > maxlines:
            print warn_message
            break
    f.close()
        
def truncate_latex_command_file(filename, keeplines=200):
    """truncate file but keep in good latex shape"""
    open(filename, 'a').close()
    lines = list(open(filename, 'r'))
    f = open(filename, 'w')
    for i, line in enumerate(lines):
        if i > keeplines and line.startswith('\providecommand'):
            break
        f.write(line)
    f.close()
    
def strip_pathname(name):
    """remove ../ and ./ and leading/trainling blanks and path separators from input string ``name``"""
    return name.replace('..' + os.sep, '').replace('.' + os.sep, '').strip().strip(os.sep)

def str_to_latex(string):
    """do replacements in ``string`` such that it most likely compiles with latex """
    return string.replace('\\', r'\textbackslash{}').replace('_', '\\_').replace(r'^', r'\^\,').replace(r'%', r'\%').replace(r'~', r'\ensuremath{\sim}').replace(r'#', r'\#')
                    
                    