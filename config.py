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
from bbob_pproc import genericsettings, pproc
from bbob_pproc.comp2 import ppfig2, ppscatter
from bbob_pproc.compall import ppfigs, pprldmany

def config():

    if 11 < 3: 
        pprldmany.target_values = pproc.TargetValues().set_targets(10**np.arange(2, -8, -0.2))
    elif 1 < 3: 
        pprldmany.target_values = pproc.TargetValues('bestGECCO2009').set_targets(10**np.arange(-0.3, 1.8, 0.2))
        print 'taking bestGECCO2009 based target values'
    else:
        # TODO: this case needs to be tested yet: the current problem is that no noisy data are in this folder
        pprldmany.target_values = pproc.TargetValues('RANDOMSEARCH').set_targets(10**np.arange(1, 4, 0.2))
    
    pprldmany.fontsize = 20.0  # should depend on the number of data lines down to 10.0 ?
    
    ppscatter.markersize = 14.
    
    ppfig2.linewidth = 4.
    
    ppfigs.styles = ppfigs.styles
    ppfig2.styles = ppfig2.styles

