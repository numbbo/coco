import os
import numpy as np
import warnings

scenario_rlbased = 'rlbased'
scenario_fixed = 'fixed'
scenario_biobjfixed = 'biobjfixed'
all_scenarios = [scenario_rlbased, scenario_fixed, scenario_biobjfixed]

testbed_name_single = 'bbob'
testbed_name_bi = 'bbob-biobj'
testbed_name_cons = 'bbob-constrained'

default_testbed_single = 'CONSBBOBTestbed'
default_testbed_bi = 'GECCOBiObjBBOBTestbed'

current_testbed = None


def load_current_testbed(testbed_name, target_values):
    global current_testbed

    if testbed_name in globals():
        constructor = globals()[testbed_name]
        current_testbed = constructor(target_values)
    else:
        raise ValueError('Testbed class %s does not exist. Add it to testbedsettings.py to process this data.' % testbed_name)

    return current_testbed


def get_benchmarks_short_infos(is_biobjective):
    return 'biobj-benchmarkshortinfos.txt' if is_biobjective else 'consbenchmarkshortinfos.txt'


def get_short_names(file_name):
    try:
        info_list = open(os.path.join(os.path.dirname(__file__), file_name), 'r').read().split('\n')
        info_dict = {}
        for info in info_list:
            key_val = info.split(' ', 1)
            if len(key_val) > 1:
                info_dict[int(key_val[0])] = key_val[1]

        return info_dict
    except:
        warnings.warn('benchmark infos not found')


class Testbed(object):
    """this might become the future way to have settings related to testbeds
    TODO: should go somewhere else than genericsettings.py
    TODO: how do we pass information from the benchmark to the post-processing?

    """

    def info(self, fun_number=None):
        """info on the testbed if ``fun_number is None`` or one-line info
        for function with number ``fun_number``.

        """
        if fun_number is None:
            return self.__doc__

        for line in open(os.path.join(os.path.abspath(os.path.split(__file__)[0]),
                                      self.info_filename)).readlines():
            if line.split():  # ie if not empty
                try:  # empty lines are ignored
                    fun = int(line.split()[0])
                    if fun == fun_number:
                        return 'F' + str(fun) + ' ' + ' '.join(line.split()[1:])
                except ValueError:
                    continue  # ignore annotations


class GECCOBBOBTestbed(Testbed):
    """Testbed used in the GECCO BBOB workshops 2009, 2010, 2012, 2013, 2015.
    """

    def __init__(self, targetValues):
        # TODO: should become a function, as low_budget is a display setting
        # not a testbed setting
        # only the short info, how to deal with both infos?
        self.info_filename = 'GECCOBBOBbenchmarkinfos.txt'
        self.name = testbed_name_single
        self.short_names = {}
        self.hardesttargetlatex = '10^{-8}'  # used for ppfigs, pptable, pptable2, and pptables
        self.ppfigs_ftarget = 1e-8
        self.ppfigdim_target_values = targetValues((10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8))  # possibly changed in config
        self.pprldistr_target_values = targetValues((10., 1e-1, 1e-4, 1e-8))  # possibly changed in config
        self.pprldmany_target_values = targetValues(10 ** np.arange(2, -8.2, -0.2))  # possibly changed in config
        self.pprldmany_target_range_latex = '$10^{[-8..2]}$'
        self.ppscatter_target_values = targetValues(np.logspace(-8, 2, 46))
        self.rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8)  # possibly changed in config
        self.ppfvdistr_min_target = 1e-8
        self.functions_with_legend = (1, 24, 101, 130)
        self.number_of_functions = 24
        self.pptable_ftarget = 1e-8  # value for determining the success ratio in all tables
        self.pptable_targetsOfInterest = targetValues((10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7))  # for pptable and pptables
        self.pptable2_targetsOfInterest = targetValues((1e+1, 1e-1, 1e-3, 1e-5, 1e-7))  # used for pptable2
        self.pptablemany_targetsOfInterest = self.pptable_targetsOfInterest
        self.scenario = scenario_fixed
        self.best_algorithm_filename = 'bestalgentries2009.pickle.gz'
        self.short_names = get_short_names(get_benchmarks_short_infos(False))
        # expensive optimization settings:
        self.pptable_target_runlengths = [0.5, 1.2, 3, 10, 50]  # [0.5, 2, 10, 50]  # used in config for expensive setting
        self.pptable2_target_runlengths = self.pptable_target_runlengths  # [0.5, 2, 10, 50]  # used in config for expensive setting
        self.pptables_target_runlengths = self.pptable_target_runlengths  # used in config for expensive setting

class CONSBBOBTestbed(Testbed):
    """Testbed for constrained problems.
    """

    def __init__(self, targetValues):
        # TODO: should become a function, as low_budget is a display setting
        # not a testbed setting
        # only the short info, how to deal with both infos?
        self.info_filename = 'CONSBBOBbenchmarkinfos.txt'
        self.name = testbed_name_cons
        self.short_names = {}
        self.hardesttargetlatex = '10^{-8}'  # used for ppfigs, pptable, pptable2, and pptables
        self.ppfigs_ftarget = 1e-8
        self.ppfigdim_target_values = targetValues((10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8))  # possibly changed in config
        self.pprldistr_target_values = targetValues((10., 1e-1, 1e-4, 1e-8))  # possibly changed in config
        self.pprldmany_target_values = targetValues(10 ** np.arange(2, -8.2, -0.2))  # possibly changed in config
        self.pprldmany_target_range_latex = '$10^{[-8..2]}$'
        self.ppscatter_target_values = targetValues(np.logspace(-8, 2, 46))
        self.rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8)  # possibly changed in config
        self.ppfvdistr_min_target = 1e-8
        self.functions_with_legend = (1, 24, 101, 130)
        self.number_of_functions = 48
        self.pptable_ftarget = 1e-8  # value for determining the success ratio in all tables
        self.pptable_targetsOfInterest = targetValues((10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7))  # for pptable and pptables
        self.pptable2_targetsOfInterest = targetValues((1e+1, 1e-1, 1e-3, 1e-5, 1e-7))  # used for pptable2
        self.pptablemany_targetsOfInterest = self.pptable_targetsOfInterest
        self.scenario = scenario_fixed
        self.best_algorithm_filename = ''
        self.short_names = get_short_names(get_benchmarks_short_infos(False))
        # expensive optimization settings:
        self.pptable_target_runlengths = [0.5, 1.2, 3, 10, 50]  # [0.5, 2, 10, 50]  # used in config for expensive setting
        self.pptable2_target_runlengths = self.pptable_target_runlengths  # [0.5, 2, 10, 50]  # used in config for expensive setting
        self.pptables_target_runlengths = self.pptable_target_runlengths  # used in config for expensive setting


class GECCOBiObjBBOBTestbed(Testbed):
    """Testbed used in the GECCO biobjective BBOB workshop 2016.
    """

    def __init__(self, targetValues):
        # TODO: should become a function, as low_budget is a display setting
        # not a testbed setting
        # only the short info, how to deal with both infos?
        self.info_filename = 'GECCOBBOBbenchmarkinfos.txt'
        self.name = testbed_name_bi
        self.short_names = {}
        self.hardesttargetlatex = '10^{-5}'  # used for ppfigs, pptable, pptable2, and pptables
        self.ppfigs_ftarget = 1e-5
        self.ppfigdim_target_values = targetValues((1e-1, 1e-2, 1e-3, 1e-4, 1e-5))  # possibly changed in config
        self.pprldistr_target_values = targetValues((1e-1, 1e-2, 1e-3, 1e-5))  # possibly changed in config
        target_values = np.append(np.append(10 ** np.arange(0, -5.1, -0.1), [0]), -10 ** np.arange(-5, -3.9, 0.2))
        self.pprldmany_target_values = targetValues(target_values)  # possibly changed in config
        self.pprldmany_target_range_latex = '$\{-10^{-4}, -10^{-4.2}, $ $-10^{-4.4}, -10^{-4.6}, -10^{-4.8}, -10^{-5}, 0, 10^{-5}, 10^{-4.9}, 10^{-4.8}, \dots, 10^{-0.1}, 10^0\}$'
        # ppscatter_target_values are copied from the single objective case. Define the correct values!
        self.ppscatter_target_values = targetValues(np.logspace(-8, 2, 46))  # that does not look right here!
        self.rldValsOfInterest = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)  # possibly changed in config
        self.ppfvdistr_min_target = 1e-5
        self.functions_with_legend = (1, 30, 31, 55)
        self.number_of_functions = 55
        self.pptable_ftarget = 1e-5  # value for determining the success ratio in all tables
        self.pptable_targetsOfInterest = targetValues(
            (1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5))  # possibly changed in config for all tables
        self.pptable2_targetsOfInterest = targetValues((1e-1, 1e-2, 1e-3, 1e-4, 1e-5))  # used for pptable2
        self.pptablemany_targetsOfInterest = targetValues((1e-0, 1e-2, 1e-5))  # used for pptables
        self.scenario = scenario_biobjfixed
        self.best_algorithm_filename = ''
        self.short_names = get_short_names(get_benchmarks_short_infos(True))
        # expensive optimization settings:
        self.pptable_target_runlengths = [0.5, 1.2, 3, 10, 50]  # [0.5, 2, 10, 50]  # used in config for expensive setting
        self.pptable2_target_runlengths = [0.5, 1.2, 3, 10, 50]  # [0.5, 2, 10, 50]  # used in config for expensive setting
        self.pptables_target_runlengths = [2, 10, 50]  # used in config for expensive setting
