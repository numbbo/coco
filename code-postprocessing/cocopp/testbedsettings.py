from __future__ import absolute_import

import os
import numpy as np
import warnings
from collections import OrderedDict

from . import dataformatsettings

inf_like = 3e21  # what is considers as inf in logged data, here to make an inf target

scenario_rlbased = 'rlbased'
scenario_fixed = 'fixed'
scenario_biobjfixed = 'biobjfixed'

scenario_biobjrlbased = 'biobjrlbased'
scenario_biobjextfixed = 'biobjextfixed'
scenario_constrainedfixed = 'constrainedfixed'
scenario_largescalefixed = 'largescalefixed'
scenario_mixintfixed = 'mixintfixed'
scenario_biobjmixintfixed = 'biobjmixintfixed'
scenario_sboxcostfixed = 'sboxcostfixed'

all_scenarios = [scenario_rlbased, scenario_fixed,
                 scenario_biobjfixed, scenario_biobjrlbased,
                 scenario_biobjextfixed, scenario_constrainedfixed,
                 scenario_largescalefixed, scenario_mixintfixed,
                 scenario_biobjmixintfixed, scenario_sboxcostfixed]

suite_name_single = 'bbob'        # TODO: This looks like a superfluous alias for GECCOBBOBTestbed.settings['name']
suite_name_single_noisy = 'bbob-noisy'  # Shouldn't suite names better be defined in the classes which defined/describe
suite_name_bi = 'bbob-biobj'            # the suite? Isn't that the whole point of having these classes?
suite_name_bi_ext = 'bbob-biobj-ext'
suite_name_cons = 'bbob-constrained'
suite_name_ls = 'bbob-largescale'
suite_name_mixint = 'bbob-mixint'
suite_name_bi_mixint = 'bbob-biobj-mixint'

default_suite_single = 'bbob'
default_suite_single_noisy = 'bbob-noisy'
default_suite_bi = 'bbob-biobj'

# TODO: these should be class names, not strings. The introduced
# indirection in form of a one-to-one mapping looks quite superfluous.
# Why shoudn't the class names be used directly (for this, the
# assignment of suite_to_testbed has to move to after the classes are
# defined)?
default_testbed_single = 'GECCOBBOBTestbed'
default_testbed_single_noisy = 'GECCOBBOBNoisyTestbed'
default_testbed_bi = 'GECCOBiObjBBOBTestbed'
default_testbed_bi_ext = 'GECCOBiObjExtBBOBTestbed'
default_testbed_cons = 'CONSBBOBTestbed'
default_testbed_ls = 'BBOBLargeScaleTestbed'
default_testbed_mixint = 'GECCOBBOBMixintTestbed'

current_testbed = None

suite_to_testbed = {
    default_suite_single: default_testbed_single,
    default_suite_single_noisy: default_testbed_single_noisy,
    default_suite_bi: default_testbed_bi,
    'bbob-biobj-ext': default_testbed_bi_ext,
    'bbob-constrained': default_testbed_cons,
    'bbob-largescale': default_testbed_ls,
    'bbob-mixint': default_testbed_mixint,
    'bbob-biobj-mixint': 'GECCOBBOBBiObjMixintTestbed',
    'bbob-JOINED-bbob-largescale': 'BBOBLargeScaleJOINEDTestbed',
    'sbox-cost': 'SBOXCOSTTestbed',
    'bbob-JOINED-sbox-cost': 'SboxCostJOINEDTestbed'
}


def reset_current_testbed():
    global current_testbed
    current_testbed = None


def load_current_testbed(suite_name, target_values):
    """ Loads testbed class corresponding to suite_name.

    The parameter suite_name is a string, defining the
    test suite to be loaded, for example `bbob`, `bbob-biobj`,
    `bbob-largescale`, ...
    """
    global current_testbed

    testbed_name = get_testbed_from_suite(suite_name)
    if testbed_name in globals():
        constructor = globals()[testbed_name]
        current_testbed = constructor(target_values)
    else:
        raise ValueError('Testbed class %s (corresponding to suite %s) does not exist.\n  Add it to testbedsettings.py to process this data.'
                         % (testbed_name, suite_name))
    return current_testbed

def get_testbed_from_suite(suite_name):

    if suite_name in suite_to_testbed:
        return suite_to_testbed[suite_name]
    else:
        raise ValueError('Mapping from suite name to testbed class for suite %s does not exist. '
                         'Add it to suite_to_testbed dictionary in testbedsettings.py to process this data.'
                         % suite_name)

def SuiteClass(suite_name):
    """return the `class` matching the string `suite_name` (e.g. the attribute of `DataSet`)
    """
    return globals()[get_testbed_from_suite(suite_name)]  # this indirection is here for historical reasons only

reference_values = {}


def reset_reference_values():

    global reference_values
    reference_values = {}


def update_reference_values(algorithm, reference_value):

    global reference_values

    if reference_values and reference_values[list(reference_values.keys())[0]] != reference_value:
        warnings.warn(" Reference values for the algorithm '%s' are different from the algorithm '%s'"
                      % (algorithm, list(reference_values.keys())[0]))

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
        return set(reference_values.values()) if len(set(reference_values.values())) > 1 \
            else list(reference_values.values())[0]

    return None


def get_first_reference_values():

    global reference_values

    if reference_values and len(reference_values) > 0:
        return reference_values[list(reference_values.keys())[0]]

    return None


def get_short_names(file_name):
    try:
        info_list = open(os.path.join(os.path.dirname(__file__), file_name), 'r').read().split('\n')
        info_dict = {}
        for line in info_list:
            if len(line) == 0 or line.startswith('%') or line.isspace():
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
    instances_are_uniform = True
    "False for biobjective suites, used (so far only) for simulated restarts in pprldmany"
    has_constraints = False  # default

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

    def instantiate_attributes(self, class_, suffix_list=['target_values', 'targetsOfInterest']):
        """assign ``self.some_attr = class_(self.some_attr)`` if "some_attr" ends with any
        value in the `suffix_list`
        """
        for name in self.__dict__:
            for suffix in suffix_list:
                if name.endswith(suffix):
                    setattr(self, name, class_(getattr(self, name)))

    def filter(self, dsl):
        """
        Interface to make DataSetList or list of DataSets dsl
        consistent with the retrieved Testbed class(es) in rungenericmany.
        
        Initially used for making bbob-biobj and bbob-biobj-ext suites
        consistent.
        """
        return dsl

    @staticmethod
    def number_of_constraints(dimension, function_id, **kwargs):
        raise NotImplementedError  # this needs to be implemented for the constrained testbed class
        return 0

    @property
    def string_evals(self):
        if self.has_constraints:
            return "evaluations"
        else:
            return "$f$-evaluations"

    @property
    def string_evals_short(self):
        if self.has_constraints:
            return "evals"
        else:
            return "FEvals"

    @property
    def string_evals_legend(self):
        if self.has_constraints:
            return "evals"
        else:
            return "# f-evals"


class GECCOBBOBTestbed(Testbed):
    """Testbed of 24 noiseless functions used in the GECCO BBOB workshops since 2009.
    """

    shortinfo_filename = 'bbob-benchmarkshortinfos.txt'
    pptable_target_runlengths = [0.5, 1.2, 3, 10, 50]  # used in config for expensive setting
    pptable_targetsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7)  # for pptable and pptablemany
    dimsOfInterest = (5, 20)

    settings = dict(
        info_filename='bbob-benchmarkinfos.txt',
        shortinfo_filename=shortinfo_filename,
        name=suite_name_single,
        short_names=get_short_names(shortinfo_filename),
        dimensions_to_display=(2, 3, 5, 10, 20, 40),
        goto_dimension=20,  # auto-focus on this dimension in html
        rldDimsOfInterest=dimsOfInterest,
        tabDimsOfInterest=dimsOfInterest,
        hardesttargetlatex='10^{-8}',  # used for ppfigs, pptable and pptables
        ppfigs_ftarget=1e-8,  # to set target runlength in expensive setting, use genericsettings.target_runlength
        ppfig2_ftarget=1e-8,
        ppfigdim_target_values=(10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
        pprldistr_target_values=(10., 1e-1, 1e-4, 1e-8),
        pprldmany_target_values=10 ** np.arange(2, -8.2, -0.2),
        pprldmany_target_range_latex='$10^{[-8..2]}$',
        ppscatter_target_values=np.array(list(np.logspace(-8, 2, 21)) + [inf_like]),  # 21 was 46
        rldValsOfInterest=(10, 1e-1, 1e-4, 1e-8),  # possibly changed in config
        ppfvdistr_min_target=1e-8,
        functions_with_legend=(1, 24),
        first_function_number=1,
        last_function_number=24,
        reference_values_hash_dimensions=[],
        pptable_ftarget=1e-8,  # value for determining the success ratio in all tables
        pptable_targetsOfInterest=pptable_targetsOfInterest,
        pptablemany_targetsOfInterest=pptable_targetsOfInterest,
        scenario=scenario_fixed,
        reference_algorithm_filename='refalgs/best2009-bbob.tar.gz',
        reference_algorithm_displayname='best 2009',  # TODO: should be read in from data set in reference_algorithm_filename
        # .reference_algorithm_filename='data/RANDOMSEARCH'
        # .reference_algorithm_displayname="RANDOMSEARCH"  # TODO: should be read in from data set in reference_algorithm_filename
        # expensive optimization settings:
        pptable_target_runlengths=pptable_target_runlengths,
        pptables_target_runlengths=pptable_target_runlengths,
        data_format=dataformatsettings.BBOBOldDataFormat(),  #  we cannot assume the 2nd column have constraints evaluation
        # why do we want the data format hard coded in the testbed?
        # Isn't the point that the data_format should be set
        # independently of the testbed constrained to the data we actually
        # see, that is, not assigned here?
        number_of_points=5,  # nb of target function values for each decade
        instancesOfInterest=None,  # None: consider all instances
        # .instancesOfInterest={1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1,
        #                   10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1,
        #                   21: 1, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 1, 28: 1, 29: 1, 30: 1,
        #                   31: 1, 32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1, 40: 1,
        #                   41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 47: 1, 48: 1, 49: 1, 50: 1,
        #                   51: 1, 52: 1, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1, 60: 1} # consider only 2009-2016 instances
        # .instancesOfInterest={1: 1, 2: 1}
        plots_on_main_html_page = ['pprldmany_02D_noiselessall.svg', 'pprldmany_03D_noiselessall.svg',
                               'pprldmany_05D_noiselessall.svg', 'pprldmany_10D_noiselessall.svg',
                               'pprldmany_20D_noiselessall.svg', 'pprldmany_40D_noiselessall.svg'],
    ) 

    def __init__(self, targetValues):

        if 11 < 3:
            # override settings if needed...
            # self.reference_algorithm_filename = 'best09-16-bbob.tar.gz'
            # self.reference_algorithm_displayname = 'best 2009--16'  # TODO: should be read in from data set in reference_algorithm_filename
            # self.reference_algorithm_filename = 'data/RANDOMSEARCH'
            # self.reference_algorithm_displayname = "RANDOMSEARCH"  # TODO: should be read in from data set in reference_algorithm_filename
            #self.settings.short_names = get_short_names(self.shortinfo_filename)
            self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

        for key, val in GECCOBBOBTestbed.settings.items():
            setattr(self, key, val)

        #self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

        # set targets according to targetValues class (possibly all changed
        # in config:
        self.instantiate_attributes(targetValues)


    def filter(self, dsl):
        """ Updates the dimensions in all of dsl's entries
            if both bbob and bbob-largescale data is in dsl
            and sets the corresponding suite to
            BBOBLargeScaleJOINEDTestbed in this case.
            
            Returns the filtered list as a flat list.
            
            Gives an error if the data in the given
            list of DataSets dsl is not compatible.
        """
        global current_testbed

        # find out whether we have to do something:
        bbob_detected = False
        bbob_largescale_detected = False
        sboxcost_detected = False
        for ds in dsl:
            detected_suite = ds.suite_name
            if detected_suite == 'bbob':
                bbob_detected = True
            elif detected_suite == 'bbob-largescale':
                bbob_largescale_detected = True
            elif detected_suite == 'bbob-JOINED-bbob-largescale':
                continue
            elif detected_suite == 'sbox-cost':
                scenario_sboxcostfixed = True
            else:
                raise ValueError("Data from %s suite is not "
                                 "compatible with other data from "
                                 "the bbob and/or bbob-largescale "
                                 "suites" % str(ds.suite_name))

        # now update all elements in flattened list if needed:
        if bbob_detected and bbob_largescale_detected:
            for ds in dsl:
                ds.suite = 'bbob-largescale'  # to be available via ds.suite_name
            # make sure that the right testbed is loaded:
        elif bbob_detected and sboxcost_detected:
            for ds in dsl:
                ds.suite = 'bbob-JOINED-sboxcost'

        return dsl



class SBOXCOSTTestbed(GECCOBBOBTestbed):
    """Testbed of box-constrained versions of the bbob functions.
    """

    def __init__(self, targetValues):

        if 11 < 3:
            # override settings if needed...
            # self.reference_algorithm_filename = 'best09-16-bbob.tar.gz'
            # self.reference_algorithm_displayname = 'best 2009--16'  # TODO: should be read in from data set in reference_algorithm_filename
            # self.reference_algorithm_filename = 'data/RANDOMSEARCH'
            # self.reference_algorithm_displayname = "RANDOMSEARCH"  # TODO: should be read in from data set in reference_algorithm_filename
            # self.settings.short_names = get_short_names(self.shortinfo_filename)
            self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

        for key, val in GECCOBBOBTestbed.settings.items():
            setattr(self, key, val)

        self.name = 'sbox-cost'
        self.reference_algorithm_filename = ''   # no reference algorithm for now
        self.reference_algorithm_displayname = ''
        scenario = scenario_sboxcostfixed

        # self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

        # set targets according to targetValues class (possibly all changed
        # in config:
        self.instantiate_attributes(targetValues)


class SboxCostJOINEDTestbed(SBOXCOSTTestbed):
    def __init__(self, targetValues):
        for key, val in GECCOBBOBTestbed.settings.items():
            setattr(self, key, val)
        self.name = 'bbob and sbox-cost'
        self.instantiate_attributes(targetValues)
        self.reference_algorithm_filename = ''  # no reference algorithm for now
        self.reference_algorithm_displayname = ''
        scenario = scenario_sboxcostfixed



class BBOBLargeScaleJOINEDTestbed(GECCOBBOBTestbed):
    """Union of GECCOBBOBTestbed and BBOBLargeScaleTestbed with all their dimensions."""

    dimsOfInterest = (80, 320)

    settings = dict(
        dimensions_to_display=(2, 3, 5, 10, 20, 40, 80, 160, 320, 640),
        goto_dimension=160,  # auto-focus on this dimension in html
        reference_algorithm_filename=None,
        reference_algorithm_displayname=None,
        plots_on_main_html_page=['pprldmany_02D_noiselessall.svg', 'pprldmany_03D_noiselessall.svg',
                                 'pprldmany_05D_noiselessall.svg', 'pprldmany_10D_noiselessall.svg',
                                 'pprldmany_20D_noiselessall.svg', 'pprldmany_40D_noiselessall.svg',
                                 'pprldmany_80D_noiselessall.svg', 'pprldmany_160D_noiselessall.svg',
                                 'pprldmany_320D_noiselessall.svg', 'pprldmany_640D_noiselessall.svg'],

    )

    def __init__(self, targetValues):
        super(BBOBLargeScaleJOINEDTestbed, self).__init__(targetValues)

        if 11 < 3:
            # override settings if needed...
            self.settings.reference_algorithm_filename = ''  # TODO: prepare add correct reference algo
                                                             # with all dimensions 2..640
            self.settings.reference_algorithm_displayname = None  # TODO: add correct algo here

        for key, val in BBOBLargeScaleJOINEDTestbed.settings.items():
            setattr(self, key, val)
            if 'target_values' in key or 'targetsOfInterest' in key:
                self.instantiate_attributes(targetValues, [key])

    def filter(self, dsl):
        """ Does nothing but overwriting the method from superclass"""
        return dsl


class CONSBBOBTestbed(GECCOBBOBTestbed):
    """BBOB Testbed for constrained problems.
    """

    min_target = 1e-6
    min_target_latex = '-6'
    min_target_exponent = -6.2
    min_target_scatter = -6
    has_constraints = True

    shortinfo_filename = 'bbob-constrained-benchmarkshortinfos.txt'
    pptable_targetsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, min_target)  # for pptable and pptablemany

    settings = dict(
        info_filename='bbob-constrained-benchmarkinfos.txt',
        shortinfo_filename=shortinfo_filename,
        short_names=get_short_names(shortinfo_filename),
        name=suite_name_cons,
        functions_with_legend=(1, 54),
        first_function_number=1,
        last_function_number=54,
        reference_algorithm_filename='',
        reference_algorithm_displayname='',  # TODO: should be read in from data set in reference_algorithm_filename
        scenario=scenario_constrainedfixed,
        # what follows is all related to the smallest displayed target
        hardesttargetlatex = '10^{' + min_target_latex + '}',  # used for ppfigs, pptable and pptables
        ppfigs_ftarget = min_target,  # to set target runlength in expensive setting, use genericsettings.target_runlength
        ppfig2_ftarget = min_target,
        ppfigdim_target_values = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, min_target),
        pprldistr_target_values = (10., 1e-1, 1e-4, min_target),
        pprldmany_target_values = 10 ** np.arange(2, min_target_exponent, -0.2),
        pprldmany_target_range_latex = '$10^{[' + min_target_latex + '..2]}$',
        ppscatter_target_values = np.array(list(np.logspace(min_target_scatter, 2, 21)) + [inf_like]),
        rldValsOfInterest=(10, 1e-1, 1e-4, min_target),  # possibly changed in config
        pptable_ftarget = min_target,  # value for determining the success ratio in all tables
        pptable_targetsOfInterest = pptable_targetsOfInterest,
        pptablemany_targetsOfInterest = pptable_targetsOfInterest,
        ppfvdistr_min_target = min_target,
        data_format=dataformatsettings.BBOBNewDataFormat(),  # the 2nd column has constraints evaluations
        # why do we want the data format hard coded in the testbed?
        # Isn't the point that the data_format should be set
        # independently of the testbed constrained to the data we actually
        # see, that is, not assigned here?

    )

    func_cons_groups = OrderedDict({
        "Sphere": range(1, 7),
        "Separable Ellipsoid": range(7, 13),
        "Linear Slope": range(13, 19),
        "Rotated Ellipsoid": range(19, 25),
        "Discus": range(25, 31),
        "Bent Cigar": range(31, 37),
        "Different Powers": range(37, 43),
        "Separable Rastrigin": range(43, 49),
        "Rotated Rastrigin": range(49, 55)
    })

    def __init__(self, target_values):

        if 11 < 3:
            # override settings further if needed...
            self.settings.reference_algorithm_filename = 'best2018-bbob-constrained.tar.gz'  # TODO: implement
            self.settings.reference_algorithm_displayname = 'best 2018'  # TODO: should be read in from data set in reference_algorithm_filename

        super(CONSBBOBTestbed, self).__init__(target_values)

        for key, val in CONSBBOBTestbed.settings.items():
            setattr(self, key, val)
            if 'target_values' in key or 'targetsOfInterest' in key:
                self.instantiate_attributes(target_values, [key])

    def filter(self, dsl):
        """ Does nothing but overwriting the method from superclass"""
        return dsl

    @staticmethod
    def number_of_constraints(dimension, function_id, active_only=False):
        """Return the number of constraints of function `function_id`

        in the given dimension. If `active_only`, it is the number of
        constraints that are active in the global optimum.
        """

        if active_only:
            numbers = [1, 2, 6, 6 + int(dimension / 2),
                       6 + dimension, 6 + 3 * dimension]
        else:
            numbers = [1, 3, 9, 9 + int(3 * dimension / 4),
                       9 + int(3 * dimension / 2), 9 + int(9 * dimension / 2)]

        map_id_to_number = {k: n for k, n in enumerate(numbers)}

        return map_id_to_number[(function_id - 1) % 6]  # 6 is also len(numbers)

    @staticmethod
    def constraint_category(function_id, active_only=False):
        """Return the number of constraints as a string formula.

        The formula is the same for all dimensions and may contain 'n'
        which stands for dimension. If `active_only`, it gives the number
        of constraints that are active in the global optimum.
        """

        if active_only:
            numbers = ['1', '2', '6', '6+ndiv2', '6+n', '6+3n']
        else:
            numbers = ['0n+1', '0n+3', '0n+9', '3ndiv4+9', '6ndiv4+9', '9ndiv2+9']

        map_id_to_number = {k: n for k, n in enumerate(numbers)}

        return map_id_to_number[(function_id - 1) % 6]  # 6 is also len(numbers)


class GECCOBBOBNoisyTestbed(GECCOBBOBTestbed):
    """The noisy testbed used in the GECCO BBOB workshops 2009, 2010, 2012,
       2013, 2015, and 2016.
    """

    shortinfo_filename = 'bbob-noisy-benchmarkshortinfos.txt'

    settings = dict(
        info_filename='bbob-noisy-benchmarkinfos.txt',
        shortinfo_filename=shortinfo_filename,
        short_names=get_short_names(shortinfo_filename),
        name=suite_name_single, # TODO: until we clean the code which uses this name, we need to use it also here.
        functions_with_legend=(101, 130),
        first_function_number=101,
        last_function_number=130,
        reference_algorithm_filename='refalgs/best2009-bbob-noisy.tar.gz',
        reference_algorithm_displayname='best 2009',  # TODO: should be read in from data set in reference_algorithm_filename
        plots_on_main_html_page = ['pprldmany_02D_nzall.svg', 'pprldmany_03D_nzall.svg',
                                   'pprldmany_05D_nzall.svg', 'pprldmany_10D_nzall.svg',
                                   'pprldmany_20D_nzall.svg', 'pprldmany_40D_nzall.svg'],

    )
    
    def __init__(self, target_values):
        super(GECCOBBOBNoisyTestbed, self).__init__(target_values)

        if 11 < 3:
            # override settings if needed...
            self.settings.reference_algorithm_filename = 'best09-16-bbob-noisy.tar.gz'
            self.settings.reference_algorithm_displayname = 'best 2009--16'  # TODO: should be read in from data set in reference_algorithm_filename

        for key, val in GECCOBBOBNoisyTestbed.settings.items():
            setattr(self, key, val)
            if 'target_values' in key or 'targetsOfInterest' in key:
                self.instantiate_attributes(target_values, [key])


    def filter(self, dsl):
        """ Does nothing but overwriting the method from superclass"""
        return dsl


class GECCOBiObjBBOBTestbed(Testbed):
    """Testbed used in the BBOB workshops to display
       data sets run on the `bbob-biobj` test suite.
    """

    shortinfo_filename = 'bbob-biobj-benchmarkshortinfos.txt'
    pptable_target_runlengths = [0.5, 1.2, 3, 10, 50] # used in config for expensive setting
    pptable_targetsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7) # for pptable and pptablemany
    dimsOfInterest = (5, 20)

    settings = dict(
        info_filename='bbob-biobj-benchmarkinfos.txt',
        shortinfo_filename=shortinfo_filename,
        name=suite_name_bi,
        short_names=get_short_names(shortinfo_filename),
        dimensions_to_display=(2, 3, 5, 10, 20, 40),
        goto_dimension=20,  # auto-focus on this dimension in html
        rldDimsOfInterest=dimsOfInterest,
        tabDimsOfInterest=dimsOfInterest,
        hardesttargetlatex='10^{-5}',  # used for ppfigs, pptable and pptables
        ppfigs_ftarget=1e-5,  # to set target runlength in expensive setting, use genericsettings.target_runlength
        ppfig2_ftarget=1e-5,
        ppfigdim_target_values=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        pprldistr_target_values=(1e-1, 1e-2, 1e-3, 1e-5),
        pprldmany_target_values=
        np.append(np.append(10 ** np.arange(0, -5.1, -0.1), [0]), -10 ** np.arange(-5, -3.9, 0.2)),
        instances_are_uniform = False,
        pprldmany_target_range_latex='$\{-10^{-4}, -10^{-4.2}, $ $-10^{-4.4}, -10^{-4.6}, -10^{-4.8}, -10^{-5}, 0, 10^{-5}, 10^{-4.9}, 10^{-4.8}, \dots, 10^{-0.1}, 10^0\}$',
        ppscatter_target_values=np.array(list(np.logspace(-5, 1, 21)) + [inf_like]),  # 21 was 51
        rldValsOfInterest=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        ppfvdistr_min_target=1e-5,
        functions_with_legend=(1, 30, 31, 55),
        first_function_number=1,
        last_function_number=55,
        reference_values_hash_dimensions=[2, 3, 5, 10, 20],
        pptable_ftarget=1e-5,  # value for determining the success ratio in all tables
        pptable_targetsOfInterest=(1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5),
        pptablemany_targetsOfInterest=(1, 1e-1, 1e-2, 1e-3),  # used for pptables
        scenario=scenario_biobjfixed,
        reference_algorithm_filename='refalgs/best2016-bbob-biobj.tar.gz', # TODO produce correct best2016 algo and delete this line
        reference_algorithm_displayname='best 2016', # TODO: should be read in from data set in reference_algorithm_filename
        instancesOfInterest={1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}, # None, # None: consider all instances
        # expensive optimization settings:
        pptable_target_runlengths=[0.5, 1.2, 3, 10, 50],  # [0.5, 2, 10, 50]  # used in config for expensive setting
        pptables_target_runlengths=[2, 10, 50],  # used in config for expensive setting
        data_format=dataformatsettings.BBOBBiObjDataFormat(),
        # TODO: why do we want the data format hard coded in the testbed?
        # Isn't the point that the data_format should be set
        # independently of the testbed constrained to the data we actually
        # see, that is, not assigned here?
        number_of_points=10,  # nb of target function values for each decade
        plots_on_main_html_page=['pprldmany_02D_noiselessall.svg', 'pprldmany_03D_noiselessall.svg',
                                 'pprldmany_05D_noiselessall.svg', 'pprldmany_10D_noiselessall.svg',
                                 'pprldmany_20D_noiselessall.svg', 'pprldmany_40D_noiselessall.svg'],
    ) 

    def __init__(self, targetValues):
        
        for key, val in GECCOBiObjBBOBTestbed.settings.items():
            setattr(self, key, val)

        # set targets according to targetValues class (possibly all changed
        # in config:
        self.instantiate_attributes(targetValues)

        if 11 < 3:
            # override settings if needed...
            # self.reference_algorithm_filename = 'refalgs/best2016-bbob-biobj-NEW.tar.gz'
            # self.reference_algorithm_displayname = 'best 2016'  # TODO: should be read in from data set in reference_algorithm_filename
            # self.short_names = get_short_names(self.shortinfo_filename)
            self.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    def filter(self, dsl):
        """ Filters DataSetList or list of DataSets dsl to
            contain only the first 55 functions if data from
            both the bbob-biobj and the bbob-biobj-ext suite are detected.

            Returns the filtered list as a flat list.
        
            Gives an error if the data is not compatible.    
        """

        # find out whether we have to do something:
        bbobbiobj_detected = False
        bbobbiobjext_detected = False
        for ds in dsl:
            if ds.suite_name == 'bbob-biobj':
                bbobbiobj_detected = True
            elif ds.suite_name == 'bbob-biobj-ext':
                bbobbiobjext_detected = True
            else:
                raise ValueError("Data from %s suite is not "
                                 "compatible with other data from "
                                 "the bbob-biobj and/or bbob-biobj-ext "
                                 "suites" % str(ds.suite_name))

        # now filter all elements in flattened list if needed:
        if bbobbiobj_detected and bbobbiobjext_detected:
            dsl = list(filter(lambda ds: ds.funcId <= 55, dsl))
            for ds in dsl:
                ds.suite = 'bbob-biobj'  # to be available via ds.suite_name
        return dsl


class GECCOBiObjExtBBOBTestbed(GECCOBiObjBBOBTestbed):
    """Biobjective testbed to display data sets run on the `bbob-biobj-ext`
       test suite.
    """

    shortinfo_filename = 'bbob-biobj-ext-benchmarkshortinfos.txt'
    
    settings = dict(
        info_filename='bbob-biobj-ext-benchmarkinfos.txt',
        shortinfo_filename=shortinfo_filename,
        name=suite_name_bi_ext,
        short_names=get_short_names(shortinfo_filename),
        functions_with_legend=(1, 30, 31, 60, 61, 92),
        first_function_number=1,
        last_function_number=92,
        scenario=scenario_biobjextfixed,
        reference_algorithm_filename='', # TODO produce correct best2017 algo and delete this line
        reference_algorithm_displayname='', # TODO: should be read in from data set in reference_algorithm_filename
        instancesOfInterest={1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}, # None:consider all instances
    )

    def __init__(self, targetValues):        
        super(GECCOBiObjExtBBOBTestbed, self).__init__(targetValues)

        if 11 < 3:
            # override settings if needed...
            self.settings.reference_algorithm_filename = 'refalgs/best2017-bbob-biobj-ext.tar.gz'
            self.settings.reference_algorithm_displayname = 'best 2017'  # TODO: should be read in from data set in reference_algorithm_filename
            self.settings.short_names = get_short_names(self.shortinfo_filename)
            self.settings.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

        for key, val in GECCOBiObjExtBBOBTestbed.settings.items():
            setattr(self, key, val)
            if 'target_values' in key or 'targetsOfInterest' in key:
                self.instantiate_attributes(targetValues, [key])


class BBOBLargeScaleTestbed(GECCOBBOBTestbed):
    """ Settings related to `bbob-largescale` test suite.
    """
    
    shortinfo_filename = 'bbob-largescale-benchmarkshortinfos.txt'
    pptable_target_runlengths = [0.5, 1.2, 3, 10, 50]  # used in config for expensive setting
    pptable_targetsOfInterest = (10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7)  # for pptable and pptablemany
    dimsOfInterest = (80, 320)

    settings = dict(
        info_filename='bbob-largescale-benchmarkinfos.txt',
        shortinfo_filename=shortinfo_filename,
        name=suite_name_ls,
        short_names=get_short_names(shortinfo_filename),
        dimensions_to_display=(20, 40, 80, 160, 320, 640),
        goto_dimension=160,  # auto-focus on this dimension in html
        tabDimsOfInterest=dimsOfInterest,
        rldDimsOfInterest=dimsOfInterest,
        hardesttargetlatex='10^{-8}',  # used for ppfigs, pptable and pptables
        ppfigs_ftarget=1e-8,  # to set target runlength in expensive setting, use genericsettings.target_runlength
        ppfig2_ftarget=1e-8,
        ppfigdim_target_values=(10, 1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-8),
        pprldistr_target_values=(10., 1e-1, 1e-4, 1e-8),
        pprldmany_target_values=10 ** np.arange(2, -8.2, -0.2),
        pprldmany_target_range_latex='$10^{[-8..2]}$',
        ppscatter_target_values=np.array(list(np.logspace(-8, 2, 21)) + [inf_like]),
        rldValsOfInterest=(10, 1e-1, 1e-4, 1e-8),  # possibly changed in config
        ppfvdistr_min_target=1e-8,
        functions_with_legend=(1, 24),
        first_function_number=1,
        last_function_number=24,
        reference_values_hash_dimensions=[],
        pptable_ftarget=1e-8,  # value for determining the success ratio in all tables
        pptable_targetsOfInterest=pptable_targetsOfInterest,
        pptablemany_targetsOfInterest=pptable_targetsOfInterest,
        scenario=scenario_largescalefixed,
        reference_algorithm_filename='',
        reference_algorithm_displayname='',  # TODO: should be read in from data set in reference_algorithm_filename
        pptable_target_runlengths=pptable_target_runlengths,
        pptables_target_runlengths=pptable_target_runlengths,
        data_format=dataformatsettings.BBOBOldDataFormat(),  #  we cannot assume the 2nd column have constraints evaluation
        # TODO: why do we want the data format hard coded in the testbed?
        # Isn't the point that the data_format should be set
        # independently of the testbed constrained to the data we actually
        # see, that is, not assigned here?
        number_of_points=5,  # nb of target function values for each decade
        instancesOfInterest=None,  # None: consider all instances
        plots_on_main_html_page=['pprldmany_20D_noiselessall.svg', 'pprldmany_40D_noiselessall.svg',
                                 'pprldmany_80D_noiselessall.svg', 'pprldmany_160D_noiselessall.svg',
                                 'pprldmany_320D_noiselessall.svg', 'pprldmany_640D_noiselessall.svg']
    )

    def __init__(self, targetValues):
        super(BBOBLargeScaleTestbed, self).__init__(targetValues)

        if 11 < 3:
            # override settings if needed...
            self.settings.reference_algorithm_filename = 'refalgs/best2019-bbob-largescale.tar.gz'
            self.settings.reference_algorithm_displayname = 'best 2019'  # TODO: should be read in from data set in reference_algorithm_filename
            self.settings.short_names = get_short_names(self.shortinfo_filename)
            self.settings.instancesOfInterest = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

        for key, val in BBOBLargeScaleTestbed.settings.items():
            setattr(self, key, val)
            if 'target_values' in key or 'targetsOfInterest' in key:
                self.instantiate_attributes(targetValues, [key])


class GECCOBBOBMixintTestbed(GECCOBBOBTestbed):
    """Testbed used with the bbob-mixint test suite.
    """

    dimsOfInterest = (10, 40)

    settings = dict(
        name=suite_name_mixint,
        first_dimension=5,
        dimensions_to_display=[5, 10, 20, 40, 80, 160],
        goto_dimension=40,  # auto-focus on this dimension in html
        tabDimsOfInterest=dimsOfInterest,
        rldDimsOfInterest=dimsOfInterest,
        reference_algorithm_filename=None,  # TODO produce correct reference algo and update this line
        reference_algorithm_displayname=None,  # TODO: should be read in from data set in reference_algorithm_filename
        instancesOfInterest={1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1,
                             12: 1, 13: 1, 14: 1, 15: 1},
        scenario=scenario_mixintfixed,
        plots_on_main_html_page=['pprldmany_05D_noiselessall.svg', 'pprldmany_10D_noiselessall.svg',
                                 'pprldmany_20D_noiselessall.svg', 'pprldmany_40D_noiselessall.svg',
                                 'pprldmany_80D_noiselessall.svg', 'pprldmany_160D_noiselessall.svg'],
    )

    def __init__(self, targetValues):
        super(GECCOBBOBMixintTestbed, self).__init__(targetValues)

        for key, val in GECCOBBOBMixintTestbed.settings.items():
            setattr(self, key, val)
            if 'target_values' in key or 'targetsOfInterest' in key:
                self.instantiate_attributes(targetValues, [key])

    def filter(self, dsl):
        ''' Does nothing on dsl (overriding the filter method of the superclass). '''
        return dsl


class GECCOBBOBBiObjMixintTestbed(GECCOBiObjExtBBOBTestbed):
    """Testbed used with the bbob-biobj-mixint test suite.
    """

    dimsOfInterest = (10, 40)

    settings = dict(
        name=suite_name_bi_mixint,
        first_dimension=5,
        dimensions_to_display=[5, 10, 20, 40, 80, 160],
        goto_dimension=40,  # auto-focus on this dimension in html
        instances_are_uniform=False,
        tabDimsOfInterest=dimsOfInterest,
        rldDimsOfInterest=dimsOfInterest,
        reference_algorithm_filename=None,  # TODO produce correct reference algo and update this line
        reference_algorithm_displayname=None,  # TODO: should be read in from data set in reference_algorithm_filename
        instancesOfInterest={1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1,
                             12: 1, 13: 1, 14: 1, 15: 1},
        scenario=scenario_biobjmixintfixed,
        plots_on_main_html_page=['pprldmany_05D_noiselessall.svg', 'pprldmany_10D_noiselessall.svg',
                                 'pprldmany_20D_noiselessall.svg', 'pprldmany_40D_noiselessall.svg',
                                 'pprldmany_80D_noiselessall.svg', 'pprldmany_160D_noiselessall.svg'],
    )

    def __init__(self, targetValues):
        super(GECCOBBOBBiObjMixintTestbed, self).__init__(targetValues)

        for key, val in GECCOBBOBBiObjMixintTestbed.settings.items():
            setattr(self, key, val)
            if 'target_values' in key or 'targetsOfInterest' in key:
                self.instantiate_attributes(targetValues, [key])

    def filter(self, dsl):
        """ Checks if only data from the bbob-biobj-mixint
            suite is included in dsl. If also other test suite
            data sets is detected, an error is given.

            Returns the filtered list as a flat list.
        """

        # find out whether we have to do something:
        for ds in dsl:
            if not (ds.suite_name == 'bbob-biobj-mixint'):
                raise ValueError("Data from %s suite is not "
                                 "compatible with data from "
                                 "the bbob-biobj-mixint "
                                 "suites" % str(ds.suite_name))

        return dsl
