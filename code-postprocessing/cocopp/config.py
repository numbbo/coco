#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is an attempt for a global configuration file for various
parameters.

The import of this module changes default settings (attributes)
of other modules. This works, because each module has only one instance.

Before this module is imported somewhere, modules use their default settings.

This file could be dynamically modified and reloaded.

See also `genericsettings` which is a central place to define settings
used by other modules, but does not modify settings of other modules.

"""

import importlib
import warnings
import numpy as np
from . import ppfigdim
from . import genericsettings as settings, pproc, pprldistr
from . import testbedsettings as tbs
from . import dataformatsettings
from .comp2 import ppfig2, ppscatter
from .compall import pprldmany
from . import __path__  # import path for default genericsettings

if settings.test:
    np.seterr(all='raise')
np.seterr(under='ignore')  # ignore underflow


# genericsettings needs a pristine copy of itself to compare
# against so that it can output the changed settings.
gs_spec = importlib.util.find_spec('cocopp.genericsettings')
gs = importlib.util.module_from_spec(gs_spec)
gs_spec.loader.exec_module(gs)
settings.default_settings = gs


def config_target_values_setting(is_expensive, is_runlength_based):
    """manage target values setting in "expensive" optimization scenario.
    """
    if is_expensive:
        settings.maxevals_fix_display = settings.xlimit_expensive
    settings.runlength_based_targets = is_runlength_based or is_expensive


def config(suite_name=None):
    """called from a high level, e.g. rungeneric, to configure the lower level
    modules via modifying parameter settings.
    """
    config_target_values_setting(settings.isExpensive, settings.runlength_based_targets)
    if suite_name:
        tbs.load_current_testbed(suite_name, pproc.TargetValues)

    settings.simulated_runlength_bootstrap_sample_size = 10 + 990 / (1 + 10 * max(0, settings.in_a_hurry))

    if tbs.current_testbed and tbs.current_testbed.name not in tbs.suite_to_testbed:
        if ((settings.isExpensive in (True, 1) or
                settings.runlength_based_targets in (True, 1)) and
                tbs.current_testbed.reference_algorithm_filename == ''):
            warnings.warn('Expensive setting not yet supported with ' +
                          tbs.current_testbed.name +
                          ' testbed; using non-expensive setting instead.')
            settings.isExpensive = False
            settings.runlength_based_targets = False

    # pprldist.plotRLDistr2 needs to be revised regarding run_length based targets
    if settings.runlength_based_targets in (True, 1) and not tbs.current_testbed:
        # this message may be removed at some point
        print('  runlength-based targets are on, but there is no testbed available (yet)')
    if settings.runlength_based_targets in (True, 1) and tbs.current_testbed:
        
        print('Reference algorithm based target values, using ' +
              tbs.current_testbed.reference_algorithm_filename +
              ':\n  now for each function, the target '
              'values differ, but the "level of difficulty" '
              'is "the same".')

        reference_data = 'testbedsettings'
        # pprldmany:
        if 1 < 3:  # not yet functional, captions need to be adjusted and the bug reported by Ilya sorted out
            # pprldmany.caption = ... captions are still hard coded in LaTeX
            pprldmany.x_limit = settings.maxevals_fix_display  # always fixed

        if tbs.current_testbed:

            testbed = tbs.current_testbed

            testbed.scenario = tbs.scenario_rlbased
            # settings (to be used in rungenericmany while calling pprldistr.comp(...)):
            testbed.rldValsOfInterest = pproc.RunlengthBasedTargetValues(
                                        settings.target_runlengths_in_single_rldistr,
                                        reference_data=reference_data,
                                        force_different_targets_factor=10**-0.2)

            testbed.ppfigdim_target_values = pproc.RunlengthBasedTargetValues(
                                             settings.target_runlengths_in_scaling_figs,
                                             reference_data=reference_data,
                                             force_different_targets_factor=10**-0.2)

            testbed.pprldistr_target_values = pproc.RunlengthBasedTargetValues(
                                              settings.target_runlengths_in_single_rldistr,
                                              reference_data=reference_data,
                                              force_different_targets_factor=10**-0.2)

            testbed.pprldmany_target_values = pproc.RunlengthBasedTargetValues(
                                              settings.target_runlengths_pprldmany,
                                              reference_data=reference_data,
                                              smallest_target=1e-8 * 10**0.000,
                                              force_different_targets_factor=1,
                                              unique_target_values=True)

            testbed.ppscatter_target_values = pproc.RunlengthBasedTargetValues(
                                              settings.target_runlengths_ppscatter)
            # pptable:
            testbed.pptable_targetsOfInterest = pproc.RunlengthBasedTargetValues(
                                                testbed.pptable_target_runlengths,
                                                reference_data=reference_data,
                                                force_different_targets_factor=10**-0.2)
            # pptables:
            testbed.pptablemany_targetsOfInterest = pproc.RunlengthBasedTargetValues(
                                                 testbed.pptables_target_runlengths,
                                                 reference_data=reference_data,
                                                 force_different_targets_factor=10**-0.2)
            # ppfigs
            testbed.ppfigs_ftarget = pproc.RunlengthBasedTargetValues([settings.target_runlength],
                                                                      reference_data=reference_data)

        # pprldistr:
        pprldistr.runlen_xlimits_max = \
            settings.maxevals_fix_display / 2 if settings.maxevals_fix_display else None  # can be None
        pprldistr.runlen_xlimits_min = 10**-0.3  # can be None
        # ppfigdim:
        ppfigdim.xlim_max = settings.maxevals_fix_display
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

        ppscatter.markersize = 16

    else:
        # here the default values of the modules apply
        pprldmany.x_limit = settings.xlimit_pprldmany  # ...should depend on noisy/noiseless
    if 11 < 3:  # for testing purpose
        if tbs.current_testbed:
            # TODO: this case needs to be tested yet: the current problem is that no noisy data are in this folder
            tbs.current_testbed.pprldmany_target_values = \
                pproc.RunlengthBasedTargetValues(10**np.arange(1, 4, 0.2), 'RANDOMSEARCH')

    pprldmany.fontsize = 20.0  # should depend on the number of data lines down to 10.0 ?

    ppscatter.markersize = 14

    ppfig2.linewidth = 4
 

def main():
    config()
