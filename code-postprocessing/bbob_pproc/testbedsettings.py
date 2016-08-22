import os
import numpy as np
import warnings

from . import genericsettings

scenario_rlbased = 'rlbased'
scenario_fixed = 'fixed'
scenario_biobjfixed = 'biobjfixed'
all_scenarios = [scenario_rlbased, scenario_fixed, scenario_biobjfixed]

testbed_name_single = 'bbob'
testbed_name_bi = 'bbob-biobj'

default_testbed_single = 'GECCOBBOBTestbed'
default_testbed_single_noisy = 'GECCOBBOBNoisyTestbed'
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


def get_short_names(file_name):
    try:
        info_list = open(os.path.join(os.path.dirname(__file__), file_name), 'r').read().split('\n')
        info_dict = {}
        for line in info_list:
            if len(line) == 0 or line.startswith('%') or line.isspace() :
                continue
            key_val = line.split(' ', 1)
            if len(key_val) > 1:
                info_dict[int(key_val[0])] = key_val[1]

        return info_dict
    except:
        warnings.warn('benchmark infos not found')
        print(os.path.join(os.path.dirname(__file__), file_name))


class Testbed(object):
    """this might become the future way to have settings related to testbeds
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

    shortinfo_filename = 'benchmarkshortinfos.txt'
    pptable_target_runlengths = [0.5, 1.2, 3, 10, 50] # used in config for expensive setting
    pptable_targetsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7) # for pptable and pptablemany

    settings = dict(
        info_filename = 'GECCOBBOBbenchmarkinfos.txt',
        shortinfo_filename = shortinfo_filename,
        name = testbed_name_single,
        short_names = get_short_names(shortinfo_filename),
        hardesttargetlatex = '10^{-8}',  # used for ppfigs, pptable, pptable2, and pptables
        ppfigs_ftarget = 1e-8,
        ppfig2_ftarget = 1e-8,
        ppfigdim_target_values = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
        pprldistr_target_values = (10., 1e-1, 1e-4, 1e-8),
        pprldmany_target_values = 10 ** np.arange(2, -8.2, -0.2),
        pprldmany_target_range_latex = '$10^{[-8..2]}$',
        ppscatter_target_values = np.logspace(-8, 2, 46),
        rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8),  # possibly changed in config
        ppfvdistr_min_target = 1e-8,
        functions_with_legend = (1, 24, 101, 130),
        first_function_number = 1,
        last_function_number = 24,
        pptable_ftarget = 1e-8,  # value for determining the success ratio in all tables
        pptable_targetsOfInterest = pptable_targetsOfInterest,
        pptable2_targetsOfInterest = (1e+1, 1e-1, 1e-3, 1e-5, 1e-7),  # used for pptable2
        pptablemany_targetsOfInterest = pptable_targetsOfInterest,
        scenario = scenario_fixed,
        #.best_algorithm_filename = 'best2009-bbob.tar.gz',
        best_algorithm_filename = 'bestalgentries2009.pickle.gz',
        best_algorithm_displayname = 'best 2009',  # TODO: should be read in from data set in best_algorithm_filename
        #.best_algorithm_filename = 'data/RANDOMSEARCH'
        #.best_algorithm_displayname = "RANDOMSEARCH"  # TODO: should be read in from data set in best_algorithm_filename
        # expensive optimization settings:
        pptable_target_runlengths = pptable_target_runlengths,  
        pptable2_target_runlengths = pptable_target_runlengths,
        pptables_target_runlengths = pptable_target_runlengths,
        instancesOfInterest = None # None: consider all instances
        #.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
        #                   10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1,
        #                   21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1,
        #                   31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1, 40: 1,
        #                   41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1, 50: 1,
        #                   51: 1, 52: 1, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1, 60: 1} # consider only 2009-2016 instances
        #.instancesOfInterest = {1: 1, 2: 1}
    ) 

    def __init__(self, targetValues):
        
        for key, val in GECCOBBOBTestbed.settings.items():
            setattr(self, key, val)

        # set targets according to targetValues class (possibly all changed
        # in config:
        self.ppfigdim_target_values = targetValues(self.ppfigdim_target_values)
        self.pprldistr_target_values = targetValues(self.pprldistr_target_values)
        self.pprldmany_target_values = targetValues(self.pprldmany_target_values)
        self.ppscatter_target_values = targetValues(self.ppscatter_target_values)
        self.pptable_targetsOfInterest = targetValues(self.pptable_targetsOfInterest)
        self.pptable2_targetsOfInterest = targetValues(self.pptable2_targetsOfInterest)
        self.pptablemany_targetsOfInterest = targetValues(self.pptablemany_targetsOfInterest)
            
        if 11 < 3:
            # override settings if needed...
            self.best_algorithm_filename = 'bestalgentries2009.pickle.gz'
            self.best_algorithm_displayname = 'best 2009'  # TODO: should be read in from data set in best_algorithm_filename
            #self.best_algorithm_filename = 'data/RANDOMSEARCH'
            #self.best_algorithm_displayname = "RANDOMSEARCH"  # TODO: should be read in from data set in best_algorithm_filename
            self.short_names = get_short_names(self.shortinfo_filename)
            self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}


class GECCOBBOBNoisyTestbed(GECCOBBOBTestbed):
    """The noisy testbed used in the GECCO BBOB workshops 2009, 2010, 2012, 2013, 2015.
    """

    settings = dict(
        name = testbed_name_single, # TODO: until we clean the code which uses this name, we need to use it also here.
        functions_with_legend = (101, 130),
        first_function_number = 101,
        last_function_number = 130,
        best_algorithm_filename = 'best2009-bbob-noisy.tar.gz',
        best_algorithm_displayname = 'best 2009'  # TODO: should be read in from data set in best_algorithm_filename
    )
    

    def __init__(self, target_values):
        super(GECCOBBOBNoisyTestbed, self).__init__(target_values)

        for key, val in GECCOBBONoisyTestbed.settings.items():
            setattr(self, key, val)
        if 11 < 3:
            # override settings if needed...
            self.best_algorithm_filename = 'best2009-bbob-noisy.tar.gz'
            self.best_algorithm_displayname = 'best 2009'  # TODO: should be read in from data set in best_algorithm_filename


class GECCOBiObjBBOBTestbed(Testbed):
    """Testbed used in the GECCO biobjective BBOB workshop 2016.
    """

    def __init__(self, targetValues):
        # TODO: should become a function, as low_budget is a display setting
        # not a testbed setting
        # only the short info, how to deal with both infos?
        self.info_filename = 'GECCOBBOBbenchmarkinfos.txt'
        self.shortinfo_filename = 'biobj-benchmarkshortinfos.txt'
        self.name = testbed_name_bi
        self.short_names = {}
        self.hardesttargetlatex = '10^{-5}'  # used for ppfigs, pptable, pptable2, and pptables
        self.ppfigs_ftarget = 1e-5
        self.ppfig2_ftarget = 1e-5                
        self.ppfigdim_target_values = targetValues((1e-1, 1e-2, 1e-3, 1e-4, 1e-5))  # possibly changed in config
        self.pprldistr_target_values = targetValues((1e-1, 1e-2, 1e-3, 1e-5))  # possibly changed in config
        #target_values = np.append(np.append(10 ** np.arange(0, -5.1, -0.1), [0]), -10 ** np.arange(-5, -3.9, 0.2))
        target_values = 10 ** np.arange(0, -5.1, -0.1)
        self.pprldmany_target_values = targetValues(target_values)  # possibly changed in config
        #self.pprldmany_target_range_latex = '$\{-10^{-4}, -10^{-4.2}, $ $-10^{-4.4}, -10^{-4.6}, -10^{-4.8}, -10^{-5}, 0, 10^{-5}, 10^{-4.9}, 10^{-4.8}, \dots, 10^{-0.1}, 10^0\}$'
        self.pprldmany_target_range_latex = '$\{-10^{-4}, -10^{-4.2}, $ $-10^{-4.4}, -10^{-4.6}, -10^{-4.8}, -10^{-5}\}$'
        # ppscatter_target_values are copied from the single objective case. Define the correct values!
        self.ppscatter_target_values = targetValues(np.logspace(-5, 1, 51))  # that does not look right here!
        self.rldValsOfInterest = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)  # possibly changed in config
        self.ppfvdistr_min_target = 1e-5
        self.functions_with_legend = (1, 30, 31, 55)
        self.first_function_number = 1
        self.last_function_number = 55
        self.pptable_ftarget = 1e-5  # value for determining the success ratio in all tables
        self.pptable_targetsOfInterest = targetValues(
            (1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5))  # possibly changed in config for all tables
        self.pptable2_targetsOfInterest = targetValues((1e-1, 1e-2, 1e-3, 1e-4, 1e-5))  # used for pptable2
        #self.pptablemany_targetsOfInterest = self.pptable_targetsOfInterest
        self.pptablemany_targetsOfInterest = targetValues((1, 1e-1, 1e-2, 1e-3))  # used for pptables
        self.scenario = scenario_biobjfixed
        self.best_algorithm_filename = 'best2016-bbob-biobj.tar.gz'
        self.best_algorithm_filename = 'refalgs/best2016-bbob-biobj-v2.0.tar.gz' # TODO produce correct best2016 algo and delete this line
        self.best_algorithm_displayname = 'best 2016' # TODO: should be read in from data set in best_algorithm_filename
        self.short_names = get_short_names(self.shortinfo_filename)
        # expensive optimization settings:
        self.pptable_target_runlengths = [0.5, 1.2, 3, 10, 50]  # [0.5, 2, 10, 50]  # used in config for expensive setting
        self.pptable2_target_runlengths = [0.5, 1.2, 3, 10, 50]  # [0.5, 2, 10, 50]  # used in config for expensive setting
        self.pptables_target_runlengths = [2, 10, 50]  # used in config for expensive setting
        self.instancesOfInterest = None # None: consider all instances
        #self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}  # 2016 biobjective instances
        #self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}