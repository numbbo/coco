#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" Best algorithm dataset module

    This module implements :py:class:`BestAlgSet` class which is used as
    data structure for the data set of the virtual best algorithm.
    Therefore this module will be imported by other modules which need
    to access best algorithm data set.

    The best algorithm data set can be accessed by the
    :py:data:`bestAlgorithmEntries` variable. This variable needs to be
    initialized by executing functions :py:func:`load_reference_algorithm(...)`

    This module can also be used generate the best algorithm data set
    with its generate method.

"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import pickle
import gzip
import warnings
import numpy as np
import tarfile
from six import advance_iterator

from . import readalign, pproc
from .toolsdivers import print_done
from .ppfig import Usage
from . import toolsstats, toolsdivers, testbedsettings, genericsettings
from .pproc import DataSet

bestAlgorithmEntries = {}

algs2009 = ("ALPS", "AMALGAM", "BAYEDA", "BFGS", "Cauchy-EDA", "BIPOP-CMA-ES",
            "CMA-ESPLUSSEL", "DASA", "DE-PSO", "DIRECT", "EDA-PSO",
            "FULLNEWUOA", "G3PCX", "GA", "GLOBAL", "iAMALGAM",
            "IPOP-SEP-CMA-ES", "LSfminbnd", "LSstep", "MA-LS-CHAIN", "MCS",
            "NELDER", "NELDERDOERR", "NEWUOA", "ONEFIFTH", "POEMS", "PSO",
            "PSO_Bounds", "RANDOMSEARCH", "Rosenbrock", "SNOBFIT", "VNS")

# Warning: NEWUOA is there twice: NEWUOA noiseless is a 2009 entry, NEWUOA
# noisy is a 2010 entry
algs2010 = ("1komma2", "1komma2mir", "1komma2mirser", "1komma2ser", "1komma4",
            "1komma4mir", "1komma4mirser", "1komma4ser", "1plus1",
            "1plus2mirser", "ABC", "AVGNEWUOA", "CMAEGS", "DE-F-AUC",
            "DEuniform", "IPOP-ACTCMA-ES", "BIPOP-CMA-ES", "MOS", "NBC-CMA",
            "NEWUOA", "PM-AdapSS-DE", "RCGA", "SPSA", "oPOEMS", "pPOEMS")

algs2012 = ("ACOR", "BIPOPaCMA", "BIPOPsaACM", "aCMA", "CMAES", "aCMAa",
            "aCMAm", "aCMAma", "aCMAmah", "aCMAmh", "DBRCGA", "DE", "DEAE",
            "DEb", "DEctpb", "IPOPsaACM", "JADE", "JADEb", "JADEctpb",
            "NBIPOPaCMA", "NIPOPaCMA", "DE-AUTO", "DE-BFGS", "DE-ROLL",
            "DE-SIMPLEX", "MVDE", "PSO-BFGS", "xNES", "xNESas", "SNES")


# TODO: this should be reimplemented:
#  o a best algorithm and an algorithm portfolio are almost the same,
#    they should derive from a CombinedAlgorithmDataSet?

# CLASS DEFINITIONS


class BestAlgSet(DataSet):
    """Unit element of best algorithm data set.

    Here unit element means for one function and one dimension.
    This class is derived from :py:class:`DataSet` but it does not
    inherit from it.

    Class attributes:
        - funcId -- function Id (integer)
        - dim -- dimension (integer)
        - comment -- comment for the setting (string)
        - algId -- algorithm name (string)
        - evals -- collected data aligned by function values (array)
        - maxevals -- maximum number of function evaluations (array)

    evals and funvals are arrays of data collected from N data sets.
    Both have the same format: zero-th column is the value on which the
    data of a row is aligned, the N subsequent columns are either the
    numbers of function evaluations for evals or function values for
    funvals.

    Known bug: algorithms where the ERT is NaN or Inf are not taken into
    account!?

    """

    def __init__(self, dict_alg, algId='Virtual Best Algorithm'):
        """Instantiate one best algorithm data set with name algId.

        :keyword dict_alg: dictionary of datasets, keys are algorithm
                          names, values are 1-element
                          :py:class:`DataSetList`.
        :keyword algId: name of the to-be-constructed algorithm as string

        """

        # values of dict dictAlg are DataSetList which should have only one
        # element which will be assigned as values in the following lines.
        d = set()
        f = set()
        pr = 0
        for i in dict_alg.values():
            d |= set(j.dim for j in i)
            f |= set(j.funcId for j in i)
            if len(i) > 0 and hasattr(i[0], 'precision'):
                pr = max(pr, max(j.precision for j in i))

        if len(f) > 1 or len(d) > 1:
            Usage('Expect the data of algorithms for only one function and '
                  'one dimension.')

        f = f.pop()
        d = d.pop()

        dictMaxEvals = {}
        dictFinalFunVals = {}
        tmpdictAlg = {}
        best_algorithms = []
        self.success_ratio = []

        for alg, i in dict_alg.items():
            if len(i) == 0:
                warnings.warn('Algorithm %s was not tested on f%d %d-D.'
                              % (alg, f, d))
                continue
            elif len(i) > 1:
                warnings.warn('Algorithm %s has a problem on f%d %d-D.'
                              % (alg, f, d))
                continue

            tmpdictAlg[alg] = i[0]  # Assign ONLY the first element as value
            dictMaxEvals[alg] = i[0]._maxevals
            dictFinalFunVals[alg] = i[0].finalfunvals
            best_algorithms = i[0].algs
            self.success_ratio = i[0].success_ratio

        dict_alg = tmpdictAlg

        sortedAlgs = list(dict_alg.keys())
        # algorithms will be sorted along sortedAlgs which is now a fixed list

        # Align ERT
        erts = list(np.transpose(np.vstack([dict_alg[i].target, dict_alg[i].ert]))
                    for i in sortedAlgs)
        res = readalign.alignArrayData(readalign.HArrayMultiReader(erts))

        resalgs = []
        reserts = []
        instance_numbers = []
        # For each function value
        for i in res:
            # Find best algorithm
            curerts = i[1:]
            assert len((np.isnan(curerts) == False)) > 0
            currentbestert = np.inf
            currentbestalg = ''
            for j, tmpert in enumerate(curerts):
                if np.isnan(tmpert):
                    continue  # TODO: don't disregard these entries
                if tmpert == currentbestert:
                    # TODO: what do we do in case of ties?
                    # look at function values corresponding to the ERT?
                    # Look at the function evaluations? the success ratio?
                    pass
                elif tmpert < currentbestert:
                    currentbestert = tmpert
                    currentbestalg = sortedAlgs[j]
            reserts.append(currentbestert)
            resalgs.append(currentbestalg)
            sorted_instance_numbers = list(set(dict_alg[currentbestalg].instancenumbers))
            sorted_instance_numbers.sort()
            instance_numbers.append(sorted_instance_numbers)

        dictiter = {}
        dictcurLine = {}
        resDataSet = []

        # write down the #fevals to reach the function value.
        for funval, alg in zip(res[:, 0], resalgs):
            it = dictiter.setdefault(alg, iter(dict_alg[alg].evals))  # TODO: do we want evals_appended here?
            curLine = dictcurLine.setdefault(alg, np.array([np.inf, 0]))
            while curLine[0] > funval:
                try:
                    curLine = advance_iterator(it)
                except StopIteration:
                    break
            dictcurLine[alg] = curLine.copy()
            tmp = curLine.copy()
            tmp[0] = funval
            resDataSet.append(tmp)

        setalgs = set(resalgs)
        dictFunValsNoFail = {}
        for alg in setalgs:
            for curline in dict_alg[alg].funvals:
                if (curline[1:] == dict_alg[alg].finalfunvals).any():
                    # only works because the funvals are monotonous
                    break
            dictFunValsNoFail[alg] = curline.copy()

        self._evals = resDataSet
        # evals is not a np array but a list of arrays because they may not
        # all be of the same size.
        self._maxevals = dict((i, dictMaxEvals[i]) for i in setalgs)
        self.finalfunvals = dict((i, dictFinalFunVals[i]) for i in setalgs)
        self.funvalsnofail = dictFunValsNoFail
        self.dim = d
        self.funcId = f
        if pr > 0:
            self.precision = pr
        self.algs = best_algorithms if best_algorithms else resalgs
        self.instances = instance_numbers
        self.best_algorithm_data = resalgs
        self.algId = algId
        if len(sortedAlgs) > 1:
            self.comment = 'Combination of ' + ', '.join(sortedAlgs)
        else:
            self.comment = dict_alg[sortedAlgs[0]].comment.lstrip('%% ')
        self._ert = np.array(reserts)
        self._target = res[:, 0]
        self.suite = dict_alg[sortedAlgs[0]].suite_name
        self.used_algorithms = sortedAlgs
        bestfinalfunvals = np.array([np.inf])
        for alg in sortedAlgs:
            if np.median(dict_alg[alg].finalfunvals) < np.median(bestfinalfunvals):
                bestfinalfunvals = dict_alg[alg].finalfunvals
                algbestfinalfunvals = alg
        self.bestfinalfunvals = bestfinalfunvals
        self.algbestfinalfunvals = algbestfinalfunvals

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.funcId == other.funcId and
                self.dim == other.dim and
                # self.precision == other.precision and
                self.algId == other.algId and
                self.comment == other.comment)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return ('{alg: %s, F%d, dim: %d}'
                % (self.algId, self.funcId, self.dim))

    def pickle(self, outputdir=None):
        """Save instance to a pickle file.

        Saves the instance to a pickle file. If not specified
        by argument outputdir, the location of the pickle is given by
        the location of the first index file associated.

        """

        # the associated pickle file does not exist
        if not getattr(self, 'pickleFile', False):
            if outputdir is None:
                outputdir = os.path.split(self.indexFiles[0])[0] + '-pickle'
                if not os.path.isdir(outputdir):
                    try:
                        os.mkdir(outputdir)
                    except OSError:
                        print('Could not create output directory % for pickle files'
                              % outputdir)
                        raise

            self.pickleFile = os.path.join(outputdir,
                                           'bestalg_f%03d_%02d.pickle'
                                           % (self.funcId, self.dim))

        if getattr(self, 'modsFromPickleVersion', True):
            try:
                f = open(self.pickleFile, 'w')  # TODO: what if file already exist?
                pickle.dump(self, f)
                f.close()
                if genericsettings.verbose:
                    print('Saved pickle in %s.' % self.pickleFile)
            except IOError as e:
                print("I/O error(%s): %s" % (e.errno, e.strerror))
            except pickle.PicklingError:
                print("Could not pickle %s" % self)
                # else: #What?
                # if genericsettings.verbose:
                # print('Skipped update of pickle file %s: no new data.'
                # % self.pickleFile)

    def createDictInstance(self):
        """Returns a dictionary of the instances

        The key is the instance id, the value is a list of index.

        """

        dictinstance = {}
        for i in range(len(self.instancenumbers)):
            dictinstance.setdefault(self.instancenumbers[i], []).append(i)

        return dictinstance

    def detERT(self, targets):
        """Determine the expected running time to reach target values.

        :keyword list targets: target function values of interest

        :returns: list of expected running times corresponding to the
                  targets.

        """
        res = []
        for f in targets:
            idx = (self.target <= f)
            try:
                res.append(self.ert[idx][0])
            except IndexError:
                res.append(np.inf)
        return res

    # TODO: return the algorithm here as well.

    def detEvals(self, targets):
        """Determine the number of evaluations to reach target values.

        :keyword seq targets: target precisions
        :returns: list of arrays each corresponding to one value in
                  targets and the list of the corresponding algorithms

        """
        res = []
        res2 = []
        res3 = []
        for f in targets:
            tmp = np.array([np.nan] * len(self.bestfinalfunvals))
            tmp2 = None
            tmp3 = None
            for i, line in enumerate(self.evals):
                if len(self.success_ratio) > i:
                    tmp3 = [0, self.success_ratio[i][1]]
                if line[0] <= f:
                    tmp = line[1:]
                    tmp2 = self.best_algorithm_data[i]
                    if len(self.success_ratio) > i:
                        tmp3 = self.success_ratio[i]
                    break
            res.append(tmp)
            res2.append(tmp2)
            if tmp3 is not None:
                res3.append(tmp3)
        return res, res2, res3

    def get_success_ratio(self, target):
        det_evaluations = self.detEvals([target, ])
        success_ratios = det_evaluations[2]
        if len(success_ratios) > 0:
            successful_runs = success_ratios[0][0]
            all_runs = success_ratios[0][1]
        else:
            successful_runs = np.sum(np.isnan(det_evaluations[0][0]) == False)  # count the nb of success
            all_runs = len(det_evaluations[0][0])

        return successful_runs, all_runs


# FUNCTION DEFINITIONS
def reset_reference_algorithm():
    global bestAlgorithmEntries
    bestAlgorithmEntries = {}


def load_reference_algorithm(best_algo_filename, force=False, relative_load=True):
    """Assigns :py:data:`bestAlgorithmEntries`.

    This function is needed to set the global variable
    :py:data:`bestAlgorithmEntries`. It reads in the data, specified by
    the string best_algo_filename which can
    either be a pickled file (deprecated), generated by
    deprecated_customgenerate or any standard data set (i.e. a zipped or
    unzipped folder with .info, .dat, and .tdat files such as the ones
    generated by custom_generate). This function will also set
    the testbedsettings.current_testbed.reference_algorithm_displayname
    according to the read data if not already present.

    :py:data:`bestAlgorithmEntries` is a dictionary accessed by providing
    a tuple :py:data:`(dimension, function)`. This returns an instance
    of :py:class:`BestAlgSet`.
    The data is that of specific algorithms (depending on the Testbed used).

    """
    global bestAlgorithmEntries
    # global statement necessary to change the variable bestalg.bestAlgorithmEntries

    if not force and bestAlgorithmEntries:
        return bestAlgorithmEntries

    # If the file or folder name is not specified then we skip the load.
    if not best_algo_filename:
        bestAlgorithmEntries = None
        return bestAlgorithmEntries

    print("Loading best algorithm data from %s ..." % best_algo_filename)
    sys.stdout.flush()

    if relative_load:
        best_alg_file_path = toolsdivers.path_in_package()
        pickleFilename = os.path.join(best_alg_file_path, best_algo_filename)
    else:
        best_alg_file_path = ''
        pickleFilename = best_algo_filename

    if pickleFilename.endswith('pickle.gz'):
        warnings.warn("   !Loading reference algorithms from pickle files deprecated!")
        
        # TODO: for backwards compatibility: check whether algorithm is 
        # actually in pickle format (and not just based on the file ending)

        fid = gzip.open(pickleFilename, 'r')
        try:
            bestAlgorithmEntries = pickle.load(fid)
            # in the deprecated case of loading pickle files, we use the
            # filename to set reference_algorithm_displayname in testbedsettings
            # if not present:
            if testbedsettings.current_testbed:            
                if hasattr(testbedsettings.current_testbed, 'reference_algorithm_displayname'):
                    testbedsettings.current_testbed.reference_algorithm_displayname = pickleFilename
        except:
            warnings.warn("failed to open file %s" % pickleFilename)
            # raise  # outcomment to diagnose
            bestAlgorithmEntries = None
        fid.close()
    else:
        algList = [os.path.join(best_alg_file_path, best_algo_filename)]
        dsList, sortedAlgs, dictAlg = pproc.processInputArgs(algList)
        bestAlgorithmEntries = generate(dictAlg, dsList[0].algId)
        # set reference_algorithm_displayname in testbedsetting if not present:
        if testbedsettings.current_testbed:
            if testbedsettings.current_testbed.reference_algorithm_displayname is None:
                testbedsettings.current_testbed.reference_algorithm_displayname = dsList[0].algId

    print_done()

    return bestAlgorithmEntries



def usage():
    print(__doc__)  # same as: sys.modules[__name__].__doc__, was: main.__doc__


def generate(dict_alg, algId):
    """Generates dictionary of best algorithm data set.
    """

    # dsList, sortedAlgs, dictAlg = processInputArgs(args)
    res = {}
    for f, i in pproc.dictAlgByFun(dict_alg).items():
        for d, j in pproc.dictAlgByDim(i).items():
            tmp = BestAlgSet(j, algId)
            res[(d, f)] = tmp
    return res


def deprecated_customgenerate(args=algs2009):
    """Generates best algorithm data set.

    This functionality is deprecated because it writes
    unsupported pickle files.

    It will create a folder bestAlg in the current working directory
    with a pickle file corresponding to the bestalg dataSet of the
    algorithms listed in variable args.

    This method is called from the python command line from a directory
    containing all necessary data folders.

    """

    outputdir = 'bestCustomAlg'

    genericsettings.verbose = True
    dsList, sortedAlgs, dictAlg = pproc.processInputArgs(args)

    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
        if genericsettings.verbose:
            print('Folder %s was created.' % outputdir)

    res = generate(dictAlg, outputdir)
    picklefilename = os.path.join(outputdir, 'bestalg.pickle')
    fid = gzip.open(picklefilename + ".gz", 'w')
    pickle.dump(res, fid)
    fid.close()

    print('done with writing pickle...')


def custom_generate(args=algs2009, algId='bestCustomAlg', suite=None):
    """Generates best algorithm data set from a given set of algorithms.

    It will create a folder named as algId in the current working directory
    corresponding to the bestalg dataSet of the algorithms listed in
    variable args. This folder is furthermore added to a `.tar.gz` file
    of the same name.

    This method is called from the python command line from a directory
    containing all necessary data folders::

    >>> import cocopp
    >>> def print_(*args, **kwargs): pass
    >>> cocopp.archives.bbob._print = print_  # prevent download message
    >>> filename = cocopp.archives.bbob.get('2009/ALPS_hornby_noiseless')
    >>> with cocopp.toolsdivers.InfolderGoneWithTheWind(cocopp.archives.bbob.local_data_path):
    ...     print('ESC'); cocopp.bestalg.custom_generate((filename, ),
    ...                           '_doctest_refAlgFromALPS') # doctest: +ELLIPSIS
    ESC...

    """

    output_dir = algId

    # genericsettings.verbose = True
    testbedsettings.reset_reference_values()
    dsList, sortedAlgs, dictAlg = pproc.processInputArgs(args)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        if genericsettings.verbose:
            print('Folder %s was created.' % output_dir)

    result = generate(dictAlg, algId)

    create_data_files(output_dir, result, suite)

    tar = tarfile.open(output_dir + ".tar.gz", "w:gz")
    tar.add(output_dir)
    tar.close()

    print('Best algorithm files were written to %s.tar.gz' % output_dir)
    print('...using instances ', end='')
    print(dsList[0].instancenumbers)


def create_data_files(output_dir, result, suite):
    from ._version import __version__ as coco_version

    if not suite:
        suite = result[list(result.keys())[0]].suite_name

    print('create_data_files: %s ' % suite)


    info_filename = '%s-bestalg' % suite
    filename_template = info_filename + '_f%02d_d%02d.%s'
    info_lines = []
    all_instances_used = []
    algorithms_used = []
    for key, value in sorted(result.items()):

        # TODO: throw an error
        # if not len(value.target) == len(value.ert):

        dict_evaluation = {}
        for index in range(len(value.target)):
            evaluation_value = value.ert[index]
            target = value.target[index]
            dict_evaluation[np.round(evaluation_value)] = target

        lines = list()
        lines.append("% Artificial instance")
        lines.append("% algorithm type = best")
        target_list = value.target.tolist()
        instances_used = []
        for key_target, value_target in sorted(dict_evaluation.items()):
            successful_runs, all_runs = result[(key[0], key[1])].get_success_ratio(value_target)
            target_index = target_list.index(value_target)
            alg_for_target = os.path.basename(value.algs[target_index])
            instances_used.append(value.instances[target_index])
            lines.append("%d %10.15e %10.15e %s %d %d" %
                         (key_target, value_target, value_target, alg_for_target, successful_runs, all_runs))
            last_evaluation = key_target
            last_value = value_target

        instance_data = "%d:%d|%10.15e" % (0, last_evaluation, last_value)
        all_instances_used.extend(instances_used)
        instances_list = get_used_instance_list(instances_used)

        algorithm_id = value.algId
        if suite in ['bbob-biobj', 'bbob-biobj-ext', testbedsettings.default_suite_bi]:
            info_lines.append("function = %d, dim = %d, %s, %s"
                              % (key[1], key[0], filename_template % (key[1], key[0], 'dat'), instance_data))
        else:
            header = "funcId = %d, DIM = %d, Precision = %10.15e, algId = '%s'" \
                     % (key[1], key[0], value.precision, algorithm_id)
            if suite is not None:
                header += ", suite = '%s'" % suite
            info_lines.append(header)
            info_lines.append("%% %s; instance_numbers: %s" % (value.comment, instances_list))
            info_lines.append("%s, %s" % (filename_template % (key[1], key[0], 'dat'), instance_data))

        filename = os.path.join(output_dir, filename_template % (key[1], key[0], 'dat'))
        write_to_file(filename, lines)
        filename = os.path.join(output_dir, filename_template % (key[1], key[0], 'tdat'))
        write_to_file(filename, lines)

        for algorithm in value.used_algorithms:
            if algorithm not in algorithms_used:
                algorithms_used.append(algorithm)

    if result[list(result.keys())[0]].suite_name == testbedsettings.default_testbed_bi:
        header = "algorithm = '%s', indicator = 'hyp'" % algorithm_id
        if suite is not None:
            header += ", suite = '%s'" % suite
        reference_values = testbedsettings.get_first_reference_values()
        if reference_values is not None:
            header += ", reference_values_hash = '%s'" % reference_values
        info_lines.insert(0, header)

        instances_list = get_used_instance_list(all_instances_used)
        if len(algorithms_used) > 1:
            comment = 'Combination of ' + ', '.join(algorithms_used)
        else:
            comment = value.comment
        comment += '; coco_version: ' + coco_version

        info_lines.insert(1, "%% %s; instance_numbers: %s" % (comment, instances_list))

    filename = os.path.join(output_dir, '%s.info' % info_filename)
    write_to_file(filename, info_lines)


def write_to_file(filename, lines):
    fid = open(filename, 'w')
    for line in lines:
        fid.write("%s\n" % line)
    fid.close()


def getAllContributingAlgorithmsToBest(algnamelist, target_lb=1e-8,
                                       target_ub=1e2):
    """Computes first the artificial best algorithm from given algorithm list
       algnamelist, constructed by extracting for each target/function pair
       thalgorithm with best ERT among the given ones. Returns then the list
       of algorithms that are contributing to the definition of the best
       algorithm, separated by dimension, and sorted by importance (i.e. with
       respect to the number of target/function pairs where each algorithm is
       best). Only target/function pairs are taken into account where the target
       is in between target_lb and target_ub.
       This method should be called from the python command line from a directory
       containing all necessary data folders::

        >>> import os, cocopp
        >>> import cocopp.toolsdivers
        >>> def print_(*args, **kwargs): pass
        >>> cocopp.archives.bbob._print = print_  # prevent downloading... message
        >>> filenames = (cocopp.archives.bbob.get('2009/BIPOP-CMA-ES'),  # first match will stay the same forever
        ...              cocopp.archives.bbob.get('2009/MCS_huyer_noiseless'))
        >>> with cocopp.toolsdivers.InfolderGoneWithTheWind(cocopp.archives.bbob.local_data_path):
        ...     cocopp.bestalg.getAllContributingAlgorithmsToBest(filenames)
        ...     assert os.path.exists('bestCustomAlg.tar.gz')  # doctest:+ELLIPSIS
        Generating best algorithm data...

    """

    print("Generating best algorithm data from given algorithm list...")
    custom_generate(algnamelist, algId='bestCustomAlg')
    
#    dsList, sortedAlgs, dictAlg = pproc.processInputArgs(('bestCustomAlg', ''))
#    bestalgentries = generate(dictAlg, dsList[0].algId)
    
    path_to_bestalg_file = os.getcwd()
    path_to_bestalg_file = os.path.join(path_to_bestalg_file, 'bestCustomAlg.tar.gz')
    
    bestalgentries = load_reference_algorithm(path_to_bestalg_file, relative_load=False, force=True)
    print('loading of best algorithm data done.')

    countsperalgorithm = {}
    for (d, f) in bestalgentries:
        setofalgs = set(bestalgentries[d, f].algs)
        # pre-processing data to only look at targets >= target_lb:
        correctedbestalgentries = []
        for i in range(0, len(bestalgentries[d, f].target)):
            if ((bestalgentries[d, f].target[i] >= target_lb) and
                    (bestalgentries[d, f].target[i] <= target_ub)):
                correctedbestalgentries.append(bestalgentries[d, f].algs[i])
        # now count how often algorithm a is best for the extracted targets
        for a in setofalgs:
            # use setdefault to initialize with zero if a entry not existant:
            countsperalgorithm.setdefault((d, a), 0)
            countsperalgorithm[(d, a)] += correctedbestalgentries.count(a)

    selectedalgsperdimension = {}
    for (d, a) in sorted(countsperalgorithm):
        if not d in selectedalgsperdimension:
            selectedalgsperdimension[d] = []
        selectedalgsperdimension[d].append((countsperalgorithm[(d, a)], a))

    for d in sorted(selectedalgsperdimension):
        print('%dD:' % d)
        for (count, alg) in sorted(selectedalgsperdimension[d], reverse=True):
            print(count, alg)
        print('\n')

    print(" done.")


def extractBestAlgorithms(args=algs2009, f_factor=2,
                          target_lb=1e-8, target_ub=1e22):
    """Returns (and prints) per dimension a list of algorithms within
    algorithm list args that contains an algorithm if for any
        dimension/target/function pair this algorithm:
        - is the best algorithm wrt ERT
        - its own ERT lies within a factor f_factor of the best ERT
        - there is no algorithm within a factor of f_factor of the best ERT
          and the current algorithm is the second best.

    """

    # TODO: use pproc.TargetValues class as input target values
    # default target values:
    targets = pproc.TargetValues(
        10 ** np.arange(np.log10(max((1e-8, target_lb))),
                        np.log10(target_ub) + 1e-9, 0.2))
    # there should be a simpler way to express this to become the
    # interface of this function

    print('Loading algorithm data from given algorithm list...\n')

    genericsettings.verbose = True
    dsList, sortedAlgs, dictAlg = pproc.processInputArgs(args)

    print('This may take a while (depending on the number of algorithms)')

    selectedAlgsPerProblem = {}
    for f, i in pproc.dictAlgByFun(dictAlg).items():
        for d, j in pproc.dictAlgByDim(i).items():

            best = BestAlgSet(j)

            selectedAlgsPerProblemDF = []
            for i in range(0, len(best.target)):
                t = best.target[i]
                # if ((t <= target_ub) and (t >= target_lb)):
                if toolsstats.in_approximately(t,
                                               targets((f, d), discretize=True)):
                    # add best for this target:
                    selectedAlgsPerProblemDF.append(best.algs[i])

                    # add second best or all algorithms that have an ERT
                    # within a factor of f_factor of the best:
                    secondbest_ERT = np.inf
                    secondbest_str = ''
                    secondbest_included = False
                    for astring in j:
                        currdictalg = dictAlg[astring].dictByDim()
                        if d in currdictalg:
                            curralgdata = currdictalg[d][f - 1]
                            currERT = curralgdata.detERT([t])[0]
                            if (astring != best.algs[i]):
                                if (currERT < secondbest_ERT):
                                    secondbest_ERT = currERT
                                    secondbest_str = astring
                                if (currERT <= best.detERT([t])[0] * f_factor):
                                    selectedAlgsPerProblemDF.append(astring)
                                    secondbest_included = True
                    if not (secondbest_included) and (secondbest_str != ''):
                        selectedAlgsPerProblemDF.append(secondbest_str)

            if len(selectedAlgsPerProblemDF) > 0:
                selectedAlgsPerProblem[(d, f)] = selectedAlgsPerProblemDF

        print('pre-processing of function %d done.' % f)

    print('loading of best algorithm(s) data done.')

    countsperalgorithm = {}
    for (d, f) in selectedAlgsPerProblem:
        print('dimension: %d, function: %d' % (d, f))
        setofalgs = set(selectedAlgsPerProblem[d, f])

        # now count how often algorithm a is best for the extracted targets
        for a in setofalgs:
            # use setdefault to initialize with zero if a entry not existant:
            countsperalgorithm.setdefault((d, a), 0)
            countsperalgorithm[(d, a)] += selectedAlgsPerProblem[d, f].count(a)

    selectedalgsperdimension = {}
    for (d, a) in sorted(countsperalgorithm):
        if not d in selectedalgsperdimension:
            selectedalgsperdimension[d] = []
        selectedalgsperdimension[d].append((countsperalgorithm[(d, a)], a))

    for d in sorted(selectedalgsperdimension):
        print('%dD:' % d)
        for (count, alg) in sorted(selectedalgsperdimension[d], reverse=True):
            print(count, alg)
        print('\n')

    print(" done.")

    return selectedalgsperdimension


def get_used_instance_list(instance_number_list):

    different_instances = []
    for instance_numbers in instance_number_list:
        if instance_numbers not in different_instances:
            different_instances.append(instance_numbers)

    if len(different_instances) == 0:
        return None
    elif len(different_instances) == 1:
        return ','.join(str(i) for i in different_instances[0])
    else:
        instance_summary = []
        for instance_list in different_instances:
            instance_summary.append(','.join(str(i) for i in instance_list))
        return '[' + '],['.join(str(i) for i in instance_summary) + ']'


