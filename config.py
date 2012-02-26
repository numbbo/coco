#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is an attempt for a global configuration file for various parameters. 
See also genericsettings.py (should be merged?)

The import of this module, :py:mod:`config`, changes default settings (attributes) 
of other modules. This works, because each module has only one instance. 

Before this module is imported somewhere, modules use their default settings. 

This file could be dynamically modified and reloaded. 

"""
import numpy as np
raise NotImplementedError()

dimensions_to_display = (2,3,5,10,20,40)  # this could be used to set the dimensions below

from bbob_pproc import ppfigdim
ppfigdim.dimsBBOB = dimensions_to_display

from bbob_pproc.compall import ppfigs
ppfigs.styles = ppfigs.styles

from bbob_pproc.comp2 import ppfig2
ppfig2.dimensions = dimensions_to_display
ppfig2.styles = ppfig2.styles
ppfig2.figformat = ppfig2.figformat

