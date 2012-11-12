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
from bbob_pproc.comp2 import ppfig2, ppscatter
from bbob_pproc.compall import ppfigs, pprldmany

pprldmany.fontsize = 20.0  # should be 12.0 ?
ppscatter.markersize = 14.
ppfig2.linewidth = 4.

ppfigs.styles = ppfigs.styles

ppfig2.styles = ppfig2.styles


