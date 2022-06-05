#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module for post-processing the data of one algorithm.

Calls the function main with arguments from the command line. Executes
the postprocessing on the given files and folders arguments, using the
:file:`.info` files found recursively.

Synopsis:
    ``python -m cocopp.rungeneric1 [OPTIONS] FOLDER``

Help:
    ``python -m cocopp.rungeneric1 --help``

"""

from __future__ import absolute_import
from __future__ import print_function

import os, sys
from pdb import set_trace
import matplotlib

import warnings, getopt, numpy as np

from . import genericsettings, testbedsettings, config, ppfig, pptable, pprldistr, ppfigdim, ppfigcons1, pplogloss, findfiles
from .pproc import DataSetList, store_reference_values, dictAlgByDim
from .ppfig import Usage
from .toolsdivers import print_done, prepend_to_file, strip_pathname1, str_to_latex, get_version_label, replace_in_file
from . import ppconverrorbars
from .compall import pprldmany, ppfigs

__all__ = ['main']


def usage():
    print(main.__doc__)

def main(alg, outputdir, argv=None):
    r"""Post-processing COCO data of a single algorithm.

    Provided with some data for an algorithm alg, this routine outputs
    figure and TeX files
    in a folder needed for the compilation of the provided LaTeX templates
    for one algorithm (``*article.tex`` or ``*1*.tex``).
    The used template file needs to be edited so that the commands
    ``\bbobdatapath`` and ``\algfolder`` point to the output folder created
    by this routine.

    These output files will contain performance tables, performance
    scaling figures and empirical cumulative distribution figures. On
    subsequent executions, new files will be added to the output folder,
    overwriting existing older files in the process.

    Previously possible to be called from the system shell, this
    routine is called via rungeneric.py only and hence any
    system shell arguments are handled there.
    """

    if (not genericsettings.verbose):
        warnings.simplefilter('module')
        # warnings.simplefilter('ignore')

    # Gets directory name if outputdir is a archive file.
    algfolder = findfiles.get_output_directory_sub_folder(alg)
    algoutputdir = os.path.join(outputdir, algfolder)

    print("\nPost-processing (1)")
    print("  loading data...")

    dsList = DataSetList(alg)

    if not dsList:
        raise Usage("Nothing to do: post-processing stopped. For more information check the messages above.")

    print("  Will generate output data in folder %s" % algoutputdir)
    print("    this might take several minutes.")

    if genericsettings.isNoisy and not genericsettings.isNoiseFree:
        dsList = dsList.dictByNoise().get('nzall', DataSetList())
    if genericsettings.isNoiseFree and not genericsettings.isNoisy:
        dsList = dsList.dictByNoise().get('noiselessall', DataSetList())

    # filter to allow postprocessing data from different test suites:
    dsList = testbedsettings.current_testbed.filter(dsList)

    store_reference_values(dsList)

    # compute maxfuneval values
    dict_max_fun_evals = {}
    for ds in dsList:
        dict_max_fun_evals[ds.dim] = np.max((dict_max_fun_evals.setdefault(ds.dim, 0), float(np.max(ds.maxevals))))

    config.config(dsList[0].suite_name)
    if genericsettings.verbose:
        for i in dsList:
            # check whether current set of instances correspond to correct
            # setting of a BBOB workshop and issue a warning otherwise:
            curr_instances = (dict((j, i.instancenumbers.count(j)) for j in set(i.instancenumbers)))
            correct = False
            for instance_set_of_interest in inset.instancesOfInterest:
                if curr_instances == instance_set_of_interest:
                    correct = True
            if not correct:
                warnings.warn('The data of %s do not list ' % i +
                              'the correct instances ' +
                              'of function F%d.' % i.funcId)

    dictAlg = dsList.dictByAlg()

    if len(dictAlg) > 1:
        warnings.warn('Data with multiple algId %s ' % str(dictAlg.keys()) +
                      'will be processed together.')
        # TODO: in this case, all is well as long as for a given problem
        # (given dimension and function) there is a single instance of
        # DataSet associated. If there are more than one, the first one only
        # will be considered... which is probably not what one would expect.

    if genericsettings.isFig or genericsettings.isTab or genericsettings.isRLDistr or genericsettings.isLogLoss:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            if genericsettings.verbose:
                print('Folder %s was created.' % (outputdir))
        if not os.path.exists(algoutputdir):
            os.makedirs(algoutputdir)
            if genericsettings.verbose:
                print('Folder %s was created.' % (algoutputdir))

    latex_commands_file = os.path.join(outputdir, 'cocopp_commands.tex')

    if genericsettings.isPickled:
        dsList.pickle()

    dictFunc = dsList.dictByFunc()
    if dictFunc[list(dictFunc.keys())[0]][0].algId not in ("", "ALG"):
        algorithm_string = " for Algorithm %s" % dictFunc[list(dictFunc.keys())[0]][0].algId
    else:
        algorithm_string = ""
    page_title = 'Results%s on the <TT>%s</TT> Benchmark Suite' % \
                 (algorithm_string, dictFunc[list(dictFunc.keys())[0]][0].suite_name)
    ppfig.save_single_functions_html(os.path.join(algoutputdir, genericsettings.single_algorithm_file_name),
                                     page_title,
                                     htmlPage=ppfig.HtmlPage.ONE,
                                     function_groups=dsList.getFuncGroups())

    values_of_interest = testbedsettings.current_testbed.ppfigdim_target_values
    if genericsettings.isFig:
        print("Scaling figures...")
        # ERT/dim vs dim.
        ppfigdim.main(dsList, values_of_interest, algoutputdir)

        print_done()

    if testbedsettings.current_testbed.has_constraints:
        print("Scaling wrt constraints...")
        ppfigcons1.main(dsList, values_of_interest, algoutputdir)
        print_done()

    if genericsettings.isConv:
        print("Generating convergence plots...")
        ppconverrorbars.main(dictAlg,
                             algoutputdir,
                             genericsettings.single_algorithm_file_name)
        print_done()

    if genericsettings.isTab:
        print("Generating LaTeX tables...")
        dictNoise = dsList.dictByNoise()
        dict_dim_list = dictAlgByDim(dictAlg)
        dims = sorted(dict_dim_list)

        ppfig.save_single_functions_html(
            os.path.join(algoutputdir, 'pptable'),
            dimensions=dims,
            htmlPage=ppfig.HtmlPage.PPTABLE,
            parentFileName=genericsettings.single_algorithm_file_name)
        replace_in_file(os.path.join(algoutputdir, 'pptable.html'), '??COCOVERSION??',
                        '<br />Data produced with COCO %s' % (get_version_label(None)))

        for noise, sliceNoise in dictNoise.items():
            pptable.main(sliceNoise, dims, algoutputdir, latex_commands_file)
        print_done()

    if genericsettings.isRLDistr:
        print("ECDF graphs...")
        dictNoise = dsList.dictByNoise()
        if len(dictNoise) > 1:
            warnings.warn('Data for functions from both the noisy and '
                          'non-noisy testbeds have been found. Their '
                          'results will be mixed in the "all functions" '
                          'ECDF figures.')
        dictDim = dsList.dictByDim()
        for dim in testbedsettings.current_testbed.rldDimsOfInterest:
            try:
                sliceDim = dictDim[dim]
            except KeyError:
                continue

            dictNoise = sliceDim.dictByNoise()

            # If there is only one noise type then we don't need the all graphs.
            if len(dictNoise) > 1:
                pprldistr.main(sliceDim, True, algoutputdir, 'all')

            for noise, sliceNoise in dictNoise.items():
                pprldistr.main(sliceNoise, True, algoutputdir, '%s' % noise)

            dictFG = sliceDim.dictByFuncGroup()
            for fGroup, sliceFuncGroup in sorted(dictFG.items()):
                pprldistr.main(sliceFuncGroup, True,
                               algoutputdir,
                               '%s' % fGroup)

            pprldistr.fmax = None  # Resetting the max final value
            pprldistr.evalfmax = None  # Resetting the max #fevalsfactor
        print_done()

        if genericsettings.isRldOnSingleFcts: # copy-paste from above, here for each function instead of function groups
            # ECDFs for each function
            print("ECDF graphs per function...")
            pprldmany.all_single_functions(dictAlg,
                                           True,
                                           None,
                                           algoutputdir,
                                           genericsettings.single_algorithm_file_name,
                                           settings=genericsettings)
            print_done()

    if genericsettings.isLogLoss:
        print("ERT loss ratio figures and tables...")
        for ng, sliceNoise in dsList.dictByNoise().items():
            if ng == 'noiselessall':
                testbed = 'noiseless'
            elif ng == 'nzall':
                testbed = 'noisy'
            txt = ("Please input crafting effort value "
                   + "for %s testbed:\n  CrE = " % testbed)
            CrE = genericsettings.inputCrE
            while CrE is None:
                try:
                    CrE = float(raw_input(txt))
                except (SyntaxError, NameError, ValueError):
                    print("Float value required.")
            dictDim = sliceNoise.dictByDim()
            for d in testbedsettings.current_testbed.rldDimsOfInterest:
                try:
                    sliceDim = dictDim[d]
                except KeyError:
                    continue
                info = '%s' % ng
                pplogloss.main(sliceDim, CrE, True, algoutputdir, info)
                pplogloss.generateTable(sliceDim, CrE, algoutputdir, info)
                for fGroup, sliceFuncGroup in sliceDim.dictByFuncGroup().items():
                    info = '%s' % fGroup
                    pplogloss.main(sliceFuncGroup, CrE, True,
                                   algoutputdir, info)
        print_done()

    prepend_to_file(latex_commands_file,
                    ['\\providecommand{\\bbobloglosstablecaption}[1]{',
                     pplogloss.table_caption(), '}'])
    prepend_to_file(latex_commands_file,
                    ['\\providecommand{\\bbobloglossfigurecaption}[1]{',
                     pplogloss.figure_caption(), '}'])
    prepend_to_file(latex_commands_file,
                    ['\\providecommand{\\bbobpprldistrlegend}[1]{',
                     pprldistr.caption_single(),  # depends on the config setting, should depend on maxfevals
                     '}'])
    # html_file = os.path.join(outputdir, 'pprldistr.html') # remove this line???
    prepend_to_file(latex_commands_file,
                    ['\\providecommand{\\bbobppfigdimlegend}[1]{',
                     ppfigdim.scaling_figure_caption(),
                     '}'])
    prepend_to_file(latex_commands_file,
                    ['\\providecommand{\\bbobpptablecaption}[1]{',
                     pptable.get_table_caption(),
                     '}'])
    prepend_to_file(latex_commands_file,
                    ['\\providecommand{\\bbobecdfcaptionsinglefcts}[2]{',
                     ppfigs.get_ecdfs_single_fcts_caption(),
                     '}'])
    prepend_to_file(latex_commands_file,
                    ['\\providecommand{\\bbobecdfcaptionallgroups}[1]{',
                     ppfigs.get_ecdfs_all_groups_caption(),
                     '}'])
    prepend_to_file(latex_commands_file,
                    ['\\providecommand{\\algfolder}{' + algfolder + '/}'])
    prepend_to_file(latex_commands_file,
                    ['\\providecommand{\\algname}{' +
                     (str_to_latex(strip_pathname1(alg[0])) if len(alg) == 1 else str_to_latex(dsList[0].algId)) + '{}}'])
    print("Output data written to folder %s" %
          os.path.join(os.getcwd(), algoutputdir))

    return dsList.dictByAlg()
