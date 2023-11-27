#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process data to be included in a generic template.

Only called from rungeneric.py.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import getopt
import warnings

from . import genericsettings, config, ppfig, testbedsettings, findfiles
from . import pproc, pptex, pprldistr
from .pproc import DataSetList, processInputArgs
from .ppfig import Usage
from .toolsdivers import prepend_to_file, strip_pathname1, str_to_latex, replace_in_file
from .compall import pprldmany, pptables, ppfigs, ppfigcons
from .comp2 import pprldistr2, ppscatter

import matplotlib.pyplot as plt
from .toolsdivers import print_done, get_version_label

__all__ = ['main']


def usage():
    print(main.__doc__)


def grouped_ecdf_graphs(alg_dict, order, output_dir, function_groups, settings, parent_file_name):
    """ Generates ecdf graphs, aggregated over groups as
        indicated via algdict
    """
    for gr, tmpdictAlg in alg_dict.items():
        dictDim = pproc.dictAlgByDim(tmpdictAlg)
        dims = sorted(dictDim)

        ppfig.save_single_functions_html(
            os.path.join(output_dir, genericsettings.pprldmany_file_name),
            '',  # algorithms names are clearly visible in the figure
            dimensions=dims,
            htmlPage=ppfig.HtmlPage.PPRLDMANY_BY_GROUP_MANY,
            function_groups=function_groups,
            parentFileName=parent_file_name
        )

        for i, d in enumerate(dims):
            entries = dictDim[d]

            pprldmany.main(entries,  # pass expensive flag here?
                           order=order,
                           outputdir=output_dir,
                           info=('%02dD_%s' % (d, gr)),
                           settings=settings
                           )

            file_name = os.path.join(output_dir, '%s.html' % genericsettings.pprldmany_file_name)
            replace_in_file(file_name, '##bbobECDFslegend##', ppfigs.ecdfs_figure_caption(True, d))
            replace_in_file(file_name, '??COCOVERSION??', '<br />Data produced with COCO %s' % (get_version_label(None)))


def main(args, outputdir):
    r"""Main routine for post-processing the data of multiple algorithms.

    Provided with some data in argument args, this routine outputs figure and TeX files
    in a folder needed for the compilation of the provided LaTeX templates
    for comparing multiple algorithms (``*many.tex`` or ``*3*.tex``).
    The used template file needs to be edited so that the commands
    ``\bbobdatapath`` points to the output folder created by this routine.

    These output files will contain performance tables, performance
    scaling figures and empirical cumulative distribution figures. On
    subsequent executions, new files will be added to the output folder,
    overwriting existing older files in the process.

    Previously possible to be called from the system shell, this
    routine is called via rungeneric.py only and hence any
    system shell arguments are handled there.

    """

    print("\nPost-processing (2+)");
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
        if genericsettings.verbose:
            print('Folder %s was created.' % outputdir)

    latex_commands_file = os.path.join(outputdir, 'cocopp_commands.tex')

    # prepend the algorithm name command to the tex-command file
    lines = []
    # first prepare list of sorted algorithm names as displayed
    algs = []
    for alg in args:
        algs.append(str_to_latex(strip_pathname1(alg)))
    algs.sort()
    # now ready for writing the sorted algorithms as \providecommand in tex-command file
    for i, alg in enumerate(algs):
        lines.append('\\providecommand{\\algorithm' + pptex.numtotext(i) +
                     '}{' + str_to_latex(strip_pathname1(alg)) + '}')
    prepend_to_file(latex_commands_file, lines, 5000,
                    'bbob_proc_commands.tex truncated, consider removing '
                    + 'the file before the text run'
                    )

    print("  loading data...")
    dsList, sortedAlgs, dictAlg = processInputArgs(args, True)
    # TODO: dictAlg not really needed here anymore as we filter
    #       dsList and then get dictAlg from there...

    # filter data set lists to be compliant with all suites
    dsList = DataSetList(testbedsettings.current_testbed.filter(dsList))
    dictAlg = dsList.dictByAlgName()

    config.config() # make sure that the filtered settings are taken into account

    if not dsList:
        sys.exit()

    algorithm_folder = findfiles.get_output_directory_sub_folder(genericsettings.foreground_algorithm_list)
    prepend_to_file(latex_commands_file, ['\\providecommand{\\algsfolder}{' + algorithm_folder + '/}'])
    many_algorithms_output = os.path.join(outputdir, algorithm_folder)

    print("  Will generate output data in folder %s" % many_algorithms_output)
    print("    this might take several minutes.")

    if not os.path.exists(many_algorithms_output):
        os.makedirs(many_algorithms_output)
        if genericsettings.verbose:
            print('Folder %s was created.' % many_algorithms_output)

    for i in dictAlg:
        if genericsettings.isNoisy and not genericsettings.isNoiseFree:
            dictAlg[i] = dictAlg[i].dictByNoise().get('nzall', DataSetList())
        if genericsettings.isNoiseFree and not genericsettings.isNoisy:
            dictAlg[i] = dictAlg[i].dictByNoise().get('noiselessall', DataSetList())

    config.config(dsList[0].suite_name)

    for i in dsList:
        if i.dim not in testbedsettings.current_testbed.dimensions_to_display:
            continue
        # check whether current set of instances correspond to correct
        # setting of a BBOB workshop and issue a warning otherwise:
        curr_instances = (dict((j, i.instancenumbers.count(j)) for j in set(i.instancenumbers)))
        correct = False
        for instance_set_of_interest in genericsettings.instancesOfInterest:
            if curr_instances == instance_set_of_interest:
                correct = True
        try: last_incorrect_instances
        except: last_incorrect_instances = None
        if not correct and curr_instances != last_incorrect_instances:
            warnings.warn('The data of %s ' % i +
                          '(and possibly further data) do not ' +
                          'list the correct instances ' +
                          'of function F%d.' % i.funcId)
            last_incorrect_instances = curr_instances

    ppfig.copy_js_files(many_algorithms_output)

    ppfig.save_single_functions_html(
        os.path.join(many_algorithms_output, genericsettings.many_algorithm_file_name),
        '',  # algorithms names are clearly visible in the figure
        htmlPage=ppfig.HtmlPage.MANY,
        function_groups=dictAlg[sortedAlgs[0]].getFuncGroups()
    )

    ppfig.save_single_functions_html(
        os.path.join(many_algorithms_output, genericsettings.ppfigs_file_name),
        '',  # algorithms names are clearly visible in the figure
        htmlPage=ppfig.HtmlPage.PPFIGS,
        function_groups=dictAlg[sortedAlgs[0]].getFuncGroups(),
        parentFileName=genericsettings.many_algorithm_file_name
    )

    dimensions = sorted(pproc.dictAlgByDim(dictAlg))

    if testbedsettings.current_testbed.has_constraints:
        ppfig.save_single_functions_html(
            os.path.join(many_algorithms_output, genericsettings.ppfigcons_file_name),
            '',  # algorithms names are clearly visible in the figure
            dimensions=dimensions,
            htmlPage=ppfig.HtmlPage.PPFIGCONS,
            function_groups=dictAlg[sortedAlgs[0]].getFuncGroups(),
            parentFileName=genericsettings.many_algorithm_file_name
        )

    ppfig.save_single_functions_html(
        os.path.join(many_algorithms_output, genericsettings.pptables_file_name),
        '',  # algorithms names are clearly visible in the figure
        dimensions=dimensions,
        htmlPage=ppfig.HtmlPage.PPTABLES,
        function_groups=dictAlg[sortedAlgs[0]].getFuncGroups(),
        parentFileName=genericsettings.many_algorithm_file_name
    )

    # empirical cumulative distribution functions (ECDFs) aka Data profiles
    if genericsettings.isRLDistr:
        config.config(dsList[0].suite_name)

        if len(genericsettings.foreground_algorithm_list) == 2:
            print("ECDF runlength ratio graphs...")

            ds_list0 = dictAlg[sortedAlgs[0]]
            dict_fun0 = ds_list0.dictByNoise()
            ds_list1 = dictAlg[sortedAlgs[1]]
            dict_fun1 = ds_list1.dictByNoise()

            if len(dict_fun0) > 1 or len(dict_fun1) > 1:
                warnings.warn('Data for functions from both the noisy and ' +
                              'non-noisy testbeds have been found. Their ' +
                              'results will be mixed in the "all functions" ' +
                              'ECDF figures.')

            algorithm_name0 = str_to_latex(strip_pathname1(sortedAlgs[0]))
            algorithm_name1 = str_to_latex(strip_pathname1(sortedAlgs[1]))

            algorithm_name = "%s vs %s" % (algorithm_name1, algorithm_name0)
            ppfig.save_single_functions_html(
                os.path.join(many_algorithms_output, genericsettings.pprldistr2_file_name),
                algname=algorithm_name,
                htmlPage=ppfig.HtmlPage.PPRLDISTR2,
                function_groups=ds_list0.getFuncGroups(),
                parentFileName=genericsettings.many_algorithm_file_name
            )

            # ECDFs of ERT ratios
            dic_dim0 = ds_list0.dictByDim()
            dic_dim1 = ds_list1.dictByDim()
            for dim in set(dic_dim0.keys()) & set(dic_dim1.keys()):
                if dim in testbedsettings.current_testbed.rldDimsOfInterest:
                    # ECDF for all functions altogether
                    try:
                        pprldistr2.main(dic_dim0[dim], dic_dim1[dim], dim,
                                        testbedsettings.current_testbed.rldValsOfInterest,
                                        many_algorithms_output,
                                        '%02dD_all' % dim)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.' % dim)
                        continue

                    # ECDFs per function groups
                    dict_fun_group0 = dic_dim0[dim].dictByFuncGroup()
                    dict_fun_group1 = dic_dim1[dim].dictByFuncGroup()

                    for fGroup in set(dict_fun_group0.keys()) & set(dict_fun_group1.keys()):
                        pprldistr2.main(dict_fun_group1[fGroup], dict_fun_group0[fGroup], dim,
                                        testbedsettings.current_testbed.rldValsOfInterest,
                                        many_algorithms_output,
                                        '%02dD_%s' % (dim, fGroup))

                    # ECDFs per noise groups
                    dict_fun0 = dic_dim0[dim].dictByNoise()
                    dict_fun1 = dic_dim1[dim].dictByNoise()

                    for fGroup in set(dict_fun0.keys()) & set(dict_fun1.keys()):
                        pprldistr2.main(dict_fun1[fGroup], dict_fun0[fGroup], dim,
                                        testbedsettings.current_testbed.rldValsOfInterest,
                                        many_algorithms_output,
                                        '%02dD_%s' % (dim, fGroup))

            prepend_to_file(latex_commands_file,
                            ['\\providecommand{\\bbobpprldistrlegendtwo}[1]{',
                             pprldistr.caption_two(),  # depends on the config
                             # setting, should depend
                             # on maxfevals
                             '}'
                             ])
            print_done()

            if testbedsettings.current_testbed not in [testbedsettings.GECCOBiObjBBOBTestbed,
                                                       testbedsettings.GECCOBiObjExtBBOBTestbed]:
                print("ECDF runlength graphs...")
                for dim in set(dic_dim0.keys()) & set(dic_dim1.keys()):
                    pprldistr.fmax = None  # Resetting the max final value
                    pprldistr.evalfmax = None  # Resetting the max #fevalsfactor
                    # ECDFs of all functions altogether
                    if dim in testbedsettings.current_testbed.rldDimsOfInterest:
                        try:
                            pprldistr.comp(dic_dim1[dim], dic_dim0[dim],
                                           testbedsettings.current_testbed.rldValsOfInterest,
                                           # TODO: let rldVals... possibly be RL-based targets
                                           True,
                                           many_algorithms_output, 'all')
                        except KeyError:
                            warnings.warn('Could not find some data in %d-D.' % dim)
                            continue

                        # ECDFs per function groups
                        dict_fun_group0 = dic_dim0[dim].dictByFuncGroup()
                        dict_fun_group1 = dic_dim1[dim].dictByFuncGroup()

                        for fGroup in set(dict_fun_group0.keys()) & set(dict_fun_group1.keys()):
                            pprldistr.comp(dict_fun_group1[fGroup], dict_fun_group0[fGroup],
                                           testbedsettings.current_testbed.rldValsOfInterest, True,
                                           many_algorithms_output,
                                           '%s' % fGroup)

                        # ECDFs per noise groups
                        dict_fun0 = dic_dim0[dim].dictByNoise()
                        dict_fun1 = dic_dim1[dim].dictByNoise()
                        for fGroup in set(dict_fun0.keys()) & set(dict_fun1.keys()):
                            pprldistr.comp(dict_fun1[fGroup], dict_fun0[fGroup],
                                           testbedsettings.current_testbed.rldValsOfInterest, True,
                                           many_algorithms_output,
                                           '%s' % fGroup)
                print_done()  # of "ECDF runlength graphs..."

        # ECDFs per noise groups
        print("ECDF graphs per noise group...")
        grouped_ecdf_graphs(pproc.dictAlgByNoi(dictAlg),
                            sortedAlgs,
                            many_algorithms_output,
                            dictAlg[sortedAlgs[0]].getFuncGroups(),
                            genericsettings,
                            genericsettings.many_algorithm_file_name)
        print_done()

        # ECDFs per function groups
        print("ECDF graphs per function group...")
        grouped_ecdf_graphs(pproc.dictAlgByFuncGroup(dictAlg),
                            sortedAlgs,
                            many_algorithms_output,
                            dictAlg[sortedAlgs[0]].getFuncGroups(),
                            genericsettings,
                            genericsettings.many_algorithm_file_name)
        print_done()

        # copy-paste from above, here for each function instead of function groups:
        print("ECDF graphs per function...")
        if genericsettings.isRldOnSingleFcts:
            # ECDFs for each function
            if 1 < 3:
                pprldmany.all_single_functions(dictAlg,
                                               False,
                                               sortedAlgs,
                                               many_algorithms_output,
                                               genericsettings.many_algorithm_file_name,
                                               settings=genericsettings)
            else:  # subject to removal
                dictFG = pproc.dictAlgByFun(dictAlg)
                for fg, tmpdictAlg in dictFG.items():
                    dictDim = pproc.dictAlgByDim(tmpdictAlg)
                    dims = sorted(dictDim)
                    for i, d in enumerate(dims):
                        entries = dictDim[d]
                        single_fct_output_dir = (many_algorithms_output.rstrip(os.sep) + os.sep +
                                                 'pprldmany-single-functions'
                                                 # + os.sep + ('f%03d' % fg)
                                                 )
                        if not os.path.exists(single_fct_output_dir):
                            os.makedirs(single_fct_output_dir)
                        pprldmany.main(entries,
                                       order=sortedAlgs,
                                       outputdir=single_fct_output_dir,
                                       info=('f%03d_%02dD' % (fg, d)),
                                       settings=genericsettings
                                       )

                    ppfig.save_single_functions_html(
                        os.path.join(single_fct_output_dir, genericsettings.pprldmany_file_name),
                        '',  # algorithms names are clearly visible in the figure
                        dimensions=dims,
                        htmlPage=ppfig.HtmlPage.NON_SPECIFIED,
                        header=ppfig.pprldmany_per_func_dim_header)
        print_done()

    if genericsettings.isTab:
        print("Generating comparison tables...")
        prepend_to_file(latex_commands_file,
                        [r'\providecommand{\bbobpptablesmanylegend}[1]{' +
                         pptables.get_table_caption() + '}'])
        dictNoi = pproc.dictAlgByNoi(dictAlg)
        for ng, tmpdictng in dictNoi.items():
            dictDim = pproc.dictAlgByDim(tmpdictng)
            for d, tmpdictdim in sorted(dictDim.items()):
                pptables.main(
                    tmpdictdim,
                    sortedAlgs,
                    many_algorithms_output,
                    ([1, 20, 38] if (testbedsettings.current_testbed.name ==
                                     testbedsettings.suite_name_bi) else True),
                    latex_commands_file)
        print_done()

    if genericsettings.isScatter and len(genericsettings.foreground_algorithm_list) == 2:
        print("Scatter plots...")

        ds_list0 = dictAlg[sortedAlgs[0]]
        algorithm_name0 = str_to_latex(strip_pathname1(sortedAlgs[0]))
        ds_list1 = dictAlg[sortedAlgs[1]]
        algorithm_name1 = str_to_latex(strip_pathname1(sortedAlgs[1]))

        algorithm_name = "%s vs %s" % (algorithm_name1, algorithm_name0)
        ppfig.save_single_functions_html(
            os.path.join(many_algorithms_output, genericsettings.ppscatter_file_name),
            algname=algorithm_name,
            htmlPage=ppfig.HtmlPage.PPSCATTER,
            function_groups=ds_list0.getFuncGroups(),
            parentFileName=genericsettings.many_algorithm_file_name
        )

        html_file_name = os.path.join(many_algorithms_output, genericsettings.ppscatter_file_name + '.html')

        ppscatter.main(ds_list1, ds_list0, many_algorithms_output, genericsettings)
        prepend_to_file(latex_commands_file,
                        ['\\providecommand{\\bbobppscatterlegend}[1]{',
                         ppscatter.figure_caption(),
                         '}'
                         ])

        replace_in_file(html_file_name, '##bbobppscatterlegend##', ppscatter.figure_caption(for_html=True))
        for i, alg in enumerate(args):
            replace_in_file(html_file_name, 'algorithm' + pptex.numtotext(i), str_to_latex(strip_pathname1(alg)))

        print_done()

    if genericsettings.isFig:
        print("Scaling figures...")
        ppfigs.main(dictAlg,
                    genericsettings.ppfigs_file_name,
                    sortedAlgs,
                    many_algorithms_output,
                    latex_commands_file)
        print_done()

    if testbedsettings.current_testbed.has_constraints:
        print("Scaling wrt constraints...")
        ppfigcons.main(dictAlg,
                       genericsettings.ppfigcons_file_name,
                       sortedAlgs,
                       many_algorithms_output,
                       latex_commands_file)
        print_done()

    print("Output data written to folder %s" %
          os.path.join(os.getcwd(), many_algorithms_output))

    return DataSetList(dsList).dictByAlg()



