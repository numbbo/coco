#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is an attempt for a global configuration file for various parameters. 

The import of this module, :py:mod:`config`, changes default settings (attributes) 
of other modules. This works, because each module has only one instance. 

Before this module is imported somewhere, modules use their default settings. 

This file could be dynamically modified and reloaded. 

See also genericsettings.py which is a central place to define settings
used by other modules, but does not modify settings of other modules.

"""

import numpy as np
import ppfig, ppfigdim, pptable
from bbob_pproc import genericsettings, pproc, pprldistr
from bbob_pproc.comp2 import ppfig2, ppscatter
from bbob_pproc.compall import ppfigs, pprldmany, pptables

def target_values(is_expensive, dict_max_fun_evals={}, runlength_limit=1e3):
    """manage target values setting in "expensive" optimization scenario, 
    when ``is_expensive not in (True, False), the setting is based on 
    the comparison of entries in ``dict_max_fun_evals`` with ``runlength_limit``
    
    """
    # if len(dict_max_fun_evals):
    #     genericsettings.dict_max_fun_evals = dict_max_fun_evals
    is_runlength_based = True if is_expensive else None 
    if is_expensive:
        genericsettings.maxevals_fix_display = genericsettings.xlimit_expensive 
    if is_runlength_based:
        genericsettings.runlength_based_targets = True
    elif is_runlength_based is False:
        genericsettings.runlength_based_targets = False            
    else: # if genericsettings.runlength_based_targets == 'auto':  # automatic choice of evaluation setup, looks still like a hack
        if len(dict_max_fun_evals) and np.max([ val / dim for dim, val in dict_max_fun_evals.iteritems()]) < runlength_limit: 
            genericsettings.runlength_based_targets = True
            genericsettings.maxevals_fix_display = genericsettings.xlimit_expensive
        else:
            genericsettings.runlength_based_targets = False

    
def config():
    """called from a high level, e.g. rungeneric, to configure the lower level 
    modules via modifying parameter settings. 
    """
    # pprldist.plotRLDistr2 needs to be revised regarding run_length based targets 
    if genericsettings.runlength_based_targets in (True, 1):
        print 'Using bestGECCO2009 based target values: now for each function the target ' + \
              'values differ, but the "level of difficulty" is "the same". '
        # pprldmany: 
        if 1 < 3:  # not yet functional, captions need to be adjusted and the bug reported by Ilya sorted out
            pprldmany.target_values = pproc.RunlengthBasedTargetValues(np.logspace(np.log10(0.5), np.log10(50), 31), 
                                                                       force_different_targets_factor=1)
            # pprldmany.caption = ... captions are still hard coded in LaTeX
            pprldmany.x_limit = genericsettings.maxevals_fix_display  # always fixed
        # pprldistr:
        pprldistr.single_target_values = pproc.RunlengthBasedTargetValues(genericsettings.target_runlengths_in_single_rldistr, 
                                                                          force_different_targets_factor=10**-0.2)
        pprldistr.runlen_xlimits_max = genericsettings.maxevals_fix_display / 2 if genericsettings.maxevals_fix_display else None # can be None
        pprldistr.runlen_xlimits_min = 10**-0.3  # can be None 
        # ppfigdim:
        ppfigdim.values_of_interest = pproc.RunlengthBasedTargetValues(genericsettings.target_runlengths_in_scaling_figs,
                                                                       # [10**i for i in [2.0, 1.5, 1.0, 0.5, 0.1, -0.3]],
                                                                       # [10**i for i in [1.7, 1, 0.3, -0.3]]
                                                                       force_different_targets_factor=10**-0.2)
        ppfigdim.xlim_max = genericsettings.maxevals_fix_display
        if ppfigdim.xlim_max:
            ppfigdim.styles = [  # sort of rainbow style, most difficult (red) first
                      {'color': 'y', 'marker': '^', 'markeredgecolor': 'k', 'markeredgewidth': 2, 'linewidth': 4},
                      {'color': 'g', 'marker': '.', 'linewidth': 4},
                      {'color': 'r', 'marker': 'o', 'markeredgecolor': 'k', 'markeredgewidth': 2, 'linewidth': 4},
                      {'color': 'm', 'marker': '.', 'linewidth': 4},
                      {'color': 'c', 'marker': 'v', 'markeredgecolor': 'k', 'markeredgewidth': 2, 'linewidth': 4},
                      {'color': 'b', 'marker': '.', 'linewidth': 4},
                      {'color': 'k', 'marker': 'o', 'markeredgecolor': 'k', 'markeredgewidth': 2, 'linewidth': 4},
                    ] 
            
        # pptable:
        pptable.table_caption=pptable.table_caption_rlbased
        pptable.targetsOfInterest = pproc.RunlengthBasedTargetValues(genericsettings.target_runlengths_in_table, 
                                                                     force_different_targets_factor=10**-0.2)
        
        # pptables (for rungenericmany):
        #pptables.table_caption=pptable.table_caption_rlbased
        pptables.targetsOfInterest = pproc.RunlengthBasedTargetValues(genericsettings.target_runlengths_in_table, 
                                                                     force_different_targets_factor=10**-0.2)

        ppscatter.markersize = 16

    else:
        pass # here the default values of the modules apply
        # pprlmany.x_limit = ...should depend on noisy/noiseless
    if 11 < 3:  # for testing purpose
        # TODO: this case needs to be tested yet: the current problem is that no noisy data are in this folder
        pprldmany.target_values = pproc.RunlengthBasedTargetValues(10**np.arange(1, 4, 0.2), 'RANDOMSEARCH')
 

    pprldmany.fontsize = 20.0  # should depend on the number of data lines down to 10.0 ?
    
    ppscatter.markersize = 14
    
    ppfig2.linewidth = 4
    ppfig2.styles = ppfig2.styles   
    ppfigs.styles = ppfigs.styles
 
def main():
    config()

