import os
import numpy as np
import warnings

from . import genericsettings

scenario_rlbased = 'rlbased'
scenario_fixed = 'fixed'
scenario_biobjfixed = 'biobjfixed'
scenario_biobjrlbased = 'biobjrlbased'
scenario_biobjextfixed = 'biobjextfixed'
scenario_largescalefixed = 'largescalefixed'
all_scenarios = [scenario_rlbased, scenario_fixed,
                 scenario_biobjfixed, scenario_biobjrlbased,
                 scenario_biobjextfixed, scenario_largescalefixed]

testbed_name_single = 'bbob'
testbed_name_single_noisy = 'bbob-noisy'
testbed_name_bi = 'bbob-biobj'
testbed_name_bi_ext = 'bbob-biobj-ext'
testbed_name_largescale = 'bbob-largescale'

default_testbed_single = 'GECCOBBOBTestbed'
default_testbed_single_noisy = 'GECCOBBOBNoisyTestbed'
default_testbed_bi = 'GECCOBiObjBBOBTestbed'
default_testbed_largescale = 'LargeScaleTestbed'
default_testbed_bi_ext = 'GECCOBiObjExtBBOBTestbed'

current_testbed = None

suite_to_testbed = {
    'bbob' : default_testbed_single,
    'bbob-noisy' : default_testbed_single_noisy,
    'bbob-biobj' : default_testbed_bi,
    'bbob-largescale' : default_testbed_largescale,
    'bbob-biobj-ext' : default_testbed_bi_ext
}


def load_current_testbed(testbed_name, target_values):
    global current_testbed

    if testbed_name in globals():
        constructor = globals()[testbed_name]
        current_testbed = constructor(target_values)
    else:
        raise ValueError('Testbed class %s does not exist. Add it to testbedsettings.py to process this data.'
                         % testbed_name)

    return current_testbed


def get_testbed_from_suite(suite_name):

    if suite_name in suite_to_testbed:
        return suite_to_testbed[suite_name]
    else:
        raise ValueError('Mapping from suite name to testbed class for suite %s does not exist. '
                         'Add it to suite_to_testbed dictionary in testbedsettings.py to process this data.'
                         % suite_name)

reference_values = {}


def update_reference_values(algorithm, reference_value):

    global reference_values

    if reference_values and reference_values[reference_values.keys()[0]] != reference_value:
        warnings.warn(" Reference values for the algorithm '%s' are different from the algorithm '%s'"
                      % (algorithm, reference_values.keys()[0]))

    reference_values[algorithm] = reference_value


def copy_reference_values(old_algorithm_id, new_algorithm_id):

    global reference_values

    if reference_values and old_algorithm_id in reference_values and new_algorithm_id not in reference_values:
        reference_values[new_algorithm_id] = reference_values[old_algorithm_id]


def get_reference_values(algorithm):
    """ Returns the hash value of the hypervolume reference values
        for the specified algorithm (if available, i.e. if the algorithm
        has been run on the `bbob-biobj` testbed).
        If algorithm=None, all hash values are returned as a set
        (i.e. with no duplicates) if more than one hash is available
        or as a string if all hashes are the same.
    """

    global reference_values

    if reference_values and algorithm in reference_values:
        return reference_values[algorithm]
    if reference_values and algorithm is None:
        return set(reference_values.values()) if len(set(reference_values.values())) > 1 else reference_values.values()[0]

    return None


def get_first_reference_values():

    global reference_values

    if reference_values and len(reference_values) > 0:
        return reference_values[reference_values.keys()[0]]

    return None


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

    reference_algorithm_displayname = None

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
    """Testbed used in the GECCO BBOB workshops 2009, 2010, 2012, 2013, 2015,
       and 2016.
    """

    shortinfo_filename = 'bbob-benchmarkshortinfos.txt'
    pptable_target_runlengths = [0.5, 1.2, 3, 10, 50] # used in config for expensive setting
    pptable_targetsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7) # for pptable and pptablemany

    settings = dict(
        info_filename = 'bbob-benchmarkinfos.txt',
        shortinfo_filename = shortinfo_filename,
        name = testbed_name_single,
        short_names = get_short_names(shortinfo_filename),
        hardesttargetlatex = '10^{-8}',  # used for ppfigs, pptable, pptable2, and pptables
        ppfigs_ftarget = 1e-8,  # to set target runlength in expensive setting, use genericsettings.target_runlength
        ppfig2_ftarget = 1e-8,
        ppfigdim_target_values = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
        pprldistr_target_values = (10., 1e-1, 1e-4, 1e-8),
        pprldmany_target_values = 10 ** np.arange(2, -8.2, -0.2),
        pprldmany_target_range_latex = '$10^{[-8..2]}$',
        ppscatter_target_values = np.logspace(-8, 2, 21),  # 21 was 46
        rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8),  # possibly changed in config
        ppfvdistr_min_target = 1e-8,
        functions_with_legend = (1, 24, 101, 130),
        first_function_number = 1,
        last_function_number = 24,
        reference_values_hash_dimensions = [],
        pptable_ftarget = 1e-8,  # value for determining the success ratio in all tables
        pptable_targetsOfInterest = pptable_targetsOfInterest,
        pptable2_targetsOfInterest = (1e+1, 1e-1, 1e-3, 1e-5, 1e-7),  # used for pptable2
        pptablemany_targetsOfInterest = pptable_targetsOfInterest,
        scenario = scenario_fixed,
        reference_algorithm_filename = 'refalgs/best2009-bbob.tar.gz',
        reference_algorithm_displayname = 'best 2009',  # TODO: should be read in from data set in reference_algorithm_filename
        #.reference_algorithm_filename = 'data/RANDOMSEARCH'
        #.reference_algorithm_displayname = "RANDOMSEARCH"  # TODO: should be read in from data set in reference_algorithm_filename
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

        # Wassim: should probably be set in a mother class for "regular" dims
        self.tabDimsOfInterest = (5, 20)
        self.rldDimsOfInterest = (5, 20)
        self.htmlDimsOfInterest = (5, 20)

        if 11 < 3:
            # override settings if needed...
            #self.reference_algorithm_filename = 'best09-16-bbob.tar.gz'
            #self.reference_algorithm_displayname = 'best 2009--16'  # TODO: should be read in from data set in reference_algorithm_filename
            #self.reference_algorithm_filename = 'data/RANDOMSEARCH'
            #self.reference_algorithm_displayname = "RANDOMSEARCH"  # TODO: should be read in from data set in reference_algorithm_filename
            self.short_names = get_short_names(self.shortinfo_filename)
            self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}


class GECCOBBOBNoisyTestbed(GECCOBBOBTestbed):
    """The noisy testbed used in the GECCO BBOB workshops 2009, 2010, 2012,
       2013, 2015, and 2016.
    """

    shortinfo_filename = 'bbob-noisy-benchmarkshortinfos.txt'

    settings = dict(
        info_filename = 'bbob-noisy-benchmarkinfos.txt',
        shortinfo_filename = shortinfo_filename,
        short_names = get_short_names(shortinfo_filename),
        name = testbed_name_single, # TODO: until we clean the code which uses this name, we need to use it also here.
        functions_with_legend = (101, 130),
        first_function_number = 101,
        last_function_number = 130,
        reference_algorithm_filename = 'refalgs/best2009-bbob-noisy.tar.gz',
        reference_algorithm_displayname = 'best 2009'  # TODO: should be read in from data set in reference_algorithm_filename
    )
    

    def __init__(self, target_values):
        super(GECCOBBOBNoisyTestbed, self).__init__(target_values)

        for key, val in GECCOBBOBNoisyTestbed.settings.items():
            setattr(self, key, val)
        if 11 < 3:
            # override settings if needed...
            self.reference_algorithm_filename = 'best09-16-bbob-noisy.tar.gz'
            self.reference_algorithm_displayname = 'best 2009--16'  # TODO: should be read in from data set in reference_algorithm_filename


class GECCOBiObjBBOBTestbed(Testbed):
    """Testbed used in the BBOB workshops to display
       data sets run on the `bbob-biobj` test suite.
    """

    shortinfo_filename = 'bbob-biobj-benchmarkshortinfos.txt'
    pptable_target_runlengths = [0.5, 1.2, 3, 10, 50] # used in config for expensive setting
    pptable_targetsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7) # for pptable and pptablemany

    settings = dict(
        info_filename = 'bbob-biobj-benchmarkinfos.txt',
        shortinfo_filename = shortinfo_filename,
        name = testbed_name_bi,
        short_names = get_short_names(shortinfo_filename),
        hardesttargetlatex = '10^{-5}',  # used for ppfigs, pptable, pptable2, and pptables
        ppfigs_ftarget = 1e-5,  # to set target runlength in expensive setting, use genericsettings.target_runlength
        ppfig2_ftarget = 1e-5,
        ppfigdim_target_values = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        pprldistr_target_values = (1e-1, 1e-2, 1e-3, 1e-5),
        pprldmany_target_values = np.append(np.append(10 ** np.arange(0, -5.1, -0.1), [0]), -10 ** np.arange(-5, -3.9, 0.2)),
        pprldmany_target_range_latex = '$\{-10^{-4}, -10^{-4.2}, $ $-10^{-4.4}, -10^{-4.6}, -10^{-4.8}, -10^{-5}, 0, 10^{-5}, 10^{-4.9}, 10^{-4.8}, \dots, 10^{-0.1}, 10^0\}$',
        ppscatter_target_values = np.logspace(-5, 1, 21),  # 21 was 51
        rldValsOfInterest = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        ppfvdistr_min_target = 1e-5,
        functions_with_legend = (1, 30, 31, 55),
        first_function_number = 1,
        last_function_number = 55,
        reference_values_hash_dimensions = [2, 3, 5, 10, 20],
        pptable_ftarget = 1e-5,  # value for determining the success ratio in all tables
        pptable_targetsOfInterest = (1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        pptable2_targetsOfInterest = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5),  # used for pptable2
        pptablemany_targetsOfInterest = (1, 1e-1, 1e-2, 1e-3),  # used for pptables
        scenario = scenario_biobjfixed,
        reference_algorithm_filename = 'refalgs/best2016-bbob-biobj.tar.gz', # TODO produce correct best2016 algo and delete this line
        reference_algorithm_displayname = 'best 2016', # TODO: should be read in from data set in reference_algorithm_filename
        instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}, # None, # None: consider all instances
        # expensive optimization settings:
        pptable_target_runlengths = [0.5, 1.2, 3, 10, 50],  # [0.5, 2, 10, 50]  # used in config for expensive setting
        pptable2_target_runlengths = [0.5, 1.2, 3, 10, 50],  # [0.5, 2, 10, 50]  # used in config for expensive setting
        pptables_target_runlengths = [2, 10, 50]  # used in config for expensive setting
    ) 

    def __init__(self, targetValues):
        
        for key, val in GECCOBiObjBBOBTestbed.settings.items():
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

        # Wassim: should probably be set in a mother class for "regular" dims
        self.tabDimsOfInterest = (5, 20)
        self.rldDimsOfInterest = (5, 20)
        self.htmlDimsOfInterest = (5, 20)
        if 11 < 3:
            # override settings if needed...
            #self.reference_algorithm_filename = 'refalgs/best2016-bbob-biobj-NEW.tar.gz'
            #self.reference_algorithm_displayname = 'best 2016'  # TODO: should be read in from data set in reference_algorithm_filename
            #self.short_names = get_short_names(self.shortinfo_filename)
            self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

class LargeScaleTestbed(GECCOBBOBTestbed):
    """First large scale Testbed.
    """


    shortinfo_filename = 'bbob-benchmarkshortinfos.txt'
    pptable_target_runlengths = [0.5, 1.2, 3, 10, 50] # used in config for expensive setting
    pptable_targetsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7) # for pptable and pptablemany

    # Wassim: there should be a more elegant way of defining this via inheritance
    settings = dict(
        info_filename = 'bbob-benchmarkinfos.txt',
        shortinfo_filename = shortinfo_filename,
        name = testbed_name_largescale,
        short_names = get_short_names(shortinfo_filename),
        hardesttargetlatex = '10^{-8}',  # used for ppfigs, pptable, pptable2, and pptables
        ppfigs_ftarget = 1e-8,  # to set target runlength in expensive setting, use genericsettings.target_runlength
        ppfig2_ftarget = 1e-8,
        ppfigdim_target_values = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
        pprldistr_target_values = (10., 1e-1, 1e-4, 1e-8),
        pprldmany_target_values = 10 ** np.arange(2, -8.2, -0.2),
        pprldmany_target_range_latex = '$10^{[-8..2]}$',
        ppscatter_target_values = np.logspace(-8, 2, 21),  # 21 was 46
        rldValsOfInterest = (10, 1e-1, 1e-4, 1e-8),  # possibly changed in config
        ppfvdistr_min_target = 1e-8,
        functions_with_legend = (1, 24, 101, 130),
        first_function_number = 1,
        last_function_number = 24,
        reference_values_hash_dimensions = [],
        pptable_ftarget = 1e-8,  # value for determining the success ratio in all tables
        pptable_targetsOfInterest = pptable_targetsOfInterest,
        pptable2_targetsOfInterest = (1e+1, 1e-1, 1e-3, 1e-5, 1e-7),  # used for pptable2
        pptablemany_targetsOfInterest = pptable_targetsOfInterest,
        scenario = scenario_fixed,
        reference_algorithm_filename = '',
        reference_algorithm_displayname = '',  # TODO: should be read in from data set in reference_algorithm_filename
        #.reference_algorithm_filename = 'data/RANDOMSEARCH'
        #.reference_algorithm_displayname = "RANDOMSEARCH"  # TODO: should be read in from data set in reference_algorithm_filename
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
        super(LargeScaleTestbed, self).__init__(targetValues)
        self.name = testbed_name_largescale
        self.first_dimension = 20
        self.scenario = scenario_largescalefixed
        # Wassim: added the following
        self.dimensions_to_display = (20, 40, 80, 160, 320, 640)
        self.tabDimsOfInterest = (80, 320)
        self.rldDimsOfInterest = (80, 320)
        self.htmlDimsOfInterest = (80, 320)
        self.best_algorithm_filename = ''
        self.best_algorithm_year = None
        self.reference_algorithm_filename = ''
        self.reference_algorithm_displayname = ''


class GECCOBiObjExtBBOBTestbed(GECCOBiObjBBOBTestbed):
    """Biobjective testbed to display data sets run on the `bbob-biobj-ext`
       test suite.
    """

    shortinfo_filename = 'bbob-biobj-benchmarkshortinfos.txt'
    
    settings = dict(
        info_filename = 'bbob-biobj-benchmarkinfos.txt',
        shortinfo_filename = shortinfo_filename,
        name = testbed_name_bi_ext,
        short_names = get_short_names(shortinfo_filename),
        functions_with_legend = (1, 30, 31, 60, 61, 92),
        first_function_number = 1,
        last_function_number = 92,
        scenario = scenario_biobjextfixed,
        reference_algorithm_filename = '', # TODO produce correct best2017 algo and delete this line
        reference_algorithm_displayname = '', # TODO: should be read in from data set in reference_algorithm_filename
        instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}, # None: consider all instances
    ) 

    def __init__(self, targetValues):        
        super(GECCOBiObjExtBBOBTestbed, self).__init__(targetValues)
        
        for key, val in GECCOBiObjExtBBOBTestbed.settings.items():
            setattr(self, key, val)
            
        if 11 < 3:
            # override settings if needed...
            self.reference_algorithm_filename = 'refalgs/best2017-bbob-biobj-ext.tar.gz'
            self.reference_algorithm_displayname = 'best 2017'  # TODO: should be read in from data set in reference_algorithm_filename
            self.short_names = get_short_names(self.shortinfo_filename)
            self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

