#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is an attempt for a global configuration file for various parameters. 

The import of this module, :py:mod:`config`, changes default settings (attributes) 
of other modules. This works, because each module has only one instance. 

Before this module is imported somewhere, modules use their default settings. 

This file could be dynamically modified and reloaded. 

See also genericsettings.py which stores settings that are used by other 
modules, but does not modify other modules settings. 

"""

import numpy as np
import ppfig, ppfigdim, pptable
from bbob_pproc import genericsettings, pproc, pprldistr
from bbob_pproc.comp2 import ppfig2, ppscatter
from bbob_pproc.compall import ppfigs, pprldmany

def config():
    """called from a high level, e.g. rungeneric, to configure the lower level 
    modules via modifying parameter settings. 
    """
    # pprldist.plotRLDistr2 needs to be revised regarding run_length based targets 
    if genericsettings.runlength_based_targets:
        print 'Using bestGECCO2009 based target values: now for each function the target ' + \
              'values differ, but the "level of difficulty" is "the same". '
        # pprldmany: 
        pprldmany.target_values = pproc.RunlengthBasedTargetValues(10**np.arange(-0.3, 2.701, 0.1), 
                                                                   force_different_targets_factor=1)
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
    else:
        pass # here the default values of the modules apply
        # pprlmany.x_limit = ...should depend on noisy/noiseless
    if 11 < 3:  # for testing purpose
        # TODO: this case needs to be tested yet: the current problem is that no noisy data are in this folder
        pprldmany.target_values = pproc.RunlengthBasedTargetValues(10**np.arange(1, 4, 0.2), 'RANDOMSEARCH')
 

    pprldmany.fontsize = 20.0  # should depend on the number of data lines down to 10.0 ?
    
    ppscatter.markersize = 14.
    
    ppfig2.linewidth = 4.
    
    ppfigs.styles = ppfigs.styles
    ppfig2.styles = ppfig2.styles

def main():
    config()

