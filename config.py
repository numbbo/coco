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
import ppfig, ppfigdim
from bbob_pproc import genericsettings, pproc, pprldistr
from bbob_pproc.comp2 import ppfig2, ppscatter
from bbob_pproc.compall import ppfigs, pprldmany

def config():
    """called from a high level, e.g. rungeneric, to configure the lower level 
    modules via modified parameter settings. 
    """
    # pprldist.plotRLDistr2 needs to be revised regarding run_length based targets 
    if genericsettings.evaluation_setting == 1e3: 
        print 'taking bestGECCO2009 based target values'
        pprldmany.target_values = pproc.RunlengthBasedTargetValues('bestGECCO2009', 
                                                                10**np.arange(-0.3, 2.701, 0.1))
        pprldistr.single_target_values = pproc.RunlengthBasedTargetValues('bestGECCO2009', [0.5, 2, 10, 50])
        pprldistr.caption_single = pprldistr.caption_single_rlbased
        ppfigdim.values_of_interest = pproc.RunlengthBasedTargetValues('bestGECCO2009',
                                                                       [10**i for i in [2.0, 1.5, 1.0, 0.5, 0.1, -0.3]],
                                                                       # [10**i for i in [1.7, 1, 0.3, -0.3]]
                                                                       force_different_targets_factor=1)
        ppfigdim.scaling_figure_caption = ppfigdim.scaling_figure_caption_rlbased
    else:
        pprldmany.target_values = pproc.TargetValues(10**np.arange(2, -8, -0.2))
        pprldistr.single_target_values = pproc.TargetValues((10., 1e-1, 1e-4, 1e-8))
        # pprlmany.x_limit = ...should depend on noisy/noiseless
    if 11 < 3:  # for testing purpose
        # TODO: this case needs to be tested yet: the current problem is that no noisy data are in this folder
        pprldmany.target_values = pproc.RunlengthBasedTargetValues('RANDOMSEARCH').set_runlengths(10**np.arange(1, 4, 0.2))
 

    pprldmany.fontsize = 20.0  # should depend on the number of data lines down to 10.0 ?
    
    ppscatter.markersize = 14.
    
    ppfig2.linewidth = 4.
    
    ppfigs.styles = ppfigs.styles
    ppfig2.styles = ppfig2.styles


