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

import warnings
import numpy as np
import ppfigdim
from . import genericsettings, pproc, pprldistr
from . import testbedsettings as tbs
from .comp2 import ppfig2, ppscatter
from .compall import pprldmany

def target_values(is_expensive, dict_max_fun_evals={}, runlength_limit=1e3):
    """manage target values setting in "expensive" optimization scenario.

    """

    if is_expensive:
        genericsettings.runlength_based_targets = True
        genericsettings.maxevals_fix_display = genericsettings.xlimit_expensive
    else:
        genericsettings.runlength_based_targets = False
        genericsettings.maxevals_fix_display = None


def config(testbed_name=None):
    """called from a high level, e.g. rungeneric, to configure the lower level
    modules via modifying parameter settings.
    """

    if testbed_name:
        tbs.load_current_testbed(testbed_name, pproc.TargetValues)

    genericsettings.simulated_runlength_bootstrap_sample_size = (10 + 990 / (1 + 10 * max(0, genericsettings.in_a_hurry)))

    # TODO: implement runlength based targets once we have a reference
    # bestAlg for the biobjective case
    # TODO: once this is solved, make sure that expensive setting is not
    # available if no bestAlg or other reference algorithm is available
    if tbs.current_testbed and tbs.current_testbed.name == tbs.testbed_name_bi:
        if (genericsettings.isExpensive in (True, 1) or
                genericsettings.runlength_based_targets in (True, 1)):
            warnings.warn('Expensive setting not yet supported with ' +
                          'bbob-biobj testbed; using non-expensive setting ' +
                          'instead.')
            genericsettings.isExpensive = False
            genericsettings.runlength_based_targets = False

    # pprldist.plotRLDistr2 needs to be revised regarding run_length based targets
    if genericsettings.runlength_based_targets in (True, 1):
        
        print('Reference algorithm based target values, using ' +
              tbs.current_testbed.best_algorithm_filename +
              ': now for each function, the target ' + 
              'values differ, but the "level of difficulty" ' +
              'is "the same". ')

        reference_data = 'testbedsettings'
        # pprldmany:
        if 1 < 3:  # not yet functional, captions need to be adjusted and the bug reported by Ilya sorted out
            # pprldmany.caption = ... captions are still hard coded in LaTeX
            pprldmany.x_limit = genericsettings.maxevals_fix_display  # always fixed


        if tbs.current_testbed:

            testbed = tbs.current_testbed

            testbed.scenario = tbs.scenario_rlbased
            # genericsettings (to be used in rungeneric2 while calling pprldistr.comp(...)):
            testbed.rldValsOfInterest = pproc.RunlengthBasedTargetValues(
                                        genericsettings.target_runlengths_in_single_rldistr,
                                        reference_data=reference_data,
                                        force_different_targets_factor=10**-0.2)

            testbed.ppfigdim_target_values = pproc.RunlengthBasedTargetValues(
                                             genericsettings.target_runlengths_in_scaling_figs,
                                             # [10**i for i in [2.0, 1.5, 1.0, 0.5, 0.1, -0.3]],
                                             # [10**i for i in [1.7, 1, 0.3, -0.3]]
                                             reference_data=reference_data,
                                             force_different_targets_factor=10**-0.2)

            testbed.pprldistr_target_values = pproc.RunlengthBasedTargetValues(
                                              genericsettings.target_runlengths_in_single_rldistr,
                                              reference_data=reference_data,
                                              force_different_targets_factor=10**-0.2)

            testbed.pprldmany_target_values = pproc.RunlengthBasedTargetValues(
                                              np.logspace(np.log10(0.5), np.log10(50), 31),
                                              reference_data=reference_data,
                                              smallest_target=1e-8 * 10**0.000,
                                              force_different_targets_factor=1,
                                              unique_target_values=True)

            testbed.ppscatter_target_values = pproc.RunlengthBasedTargetValues(
                                              np.logspace(np.log10(0.5),
                                                          np.log10(50), 8))
            # pptable:
            testbed.pptable_targetsOfInterest = pproc.RunlengthBasedTargetValues(
                                                testbed.pptable_target_runlengths,
                                                reference_data=reference_data,
                                                force_different_targets_factor=10**-0.2)
            # pptable2:
            testbed.pptable2_targetsOfInterest = pproc.RunlengthBasedTargetValues(
                                                 testbed.pptable2_target_runlengths,
                                                 reference_data=reference_data,
                                                 force_different_targets_factor=10**-0.2)
            # pptables:
            testbed.pptables_targetsOfInterest = pproc.RunlengthBasedTargetValues(
                                                 testbed.pptables_target_runlengths,
                                                 reference_data=reference_data,
                                                 force_different_targets_factor=10**-0.2)
            # ppfigs
            testbed.ppfigs_ftarget = pproc.RunlengthBasedTargetValues([genericsettings.target_runlength],
                                                                      reference_data=reference_data)

        # pprldistr:
        pprldistr.runlen_xlimits_max = genericsettings.maxevals_fix_display / 2 if genericsettings.maxevals_fix_display else None # can be None
        pprldistr.runlen_xlimits_min = 10**-0.3  # can be None
        # ppfigdim:
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

        ppscatter.markersize = 16

    else:
        pass # here the default values of the modules apply
        # pprlmany.x_limit = ...should depend on noisy/noiseless
    if 11 < 3:  # for testing purpose
        if tbs.current_testbed:
            # TODO: this case needs to be tested yet: the current problem is that no noisy data are in this folder
            tbs.current_testbed.pprldmany_target_values = pproc.RunlengthBasedTargetValues(10**np.arange(1, 4, 0.2), 'RANDOMSEARCH')


    pprldmany.fontsize = 20.0  # should depend on the number of data lines down to 10.0 ?

    ppscatter.markersize = 14

    ppfig2.linewidth = 4
 
def main():
    config()

