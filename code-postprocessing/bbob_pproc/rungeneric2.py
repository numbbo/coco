#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for the comparison of 2 algorithms.

Synopsis:
    ``python path_to_folder/bbob_pproc/rungeneric2.py [OPTIONS] FOLDER_NAME1 FOLDER_NAME2...``

Help:
    ``python path_to_folder/bbob_pproc/rungeneric2.py -h``

"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import warnings
import getopt
import numpy
import matplotlib

if __name__ == "__main__":
    matplotlib.use('Agg')  # To avoid window popup and use without X forwarding
    filepath = os.path.split(sys.argv[0])[0]
    # Add the path to bbob_pproc/.. folder
    sys.path.append(os.path.join(filepath, os.path.pardir))
    try:
        import bbob_pproc as cocopp
    except ImportError:
        import cocopp
    res = cocopp.rungeneric2.main(sys.argv[1:])
    sys.exit(res)

from . import genericsettings, ppfig, toolsdivers, rungenericmany
from .toolsdivers import print_done
from .compall import pptables

# genericsettings.summarized_target_function_values[0] might be another option

if __name__ == "__main__":
    # matplotlib.use('pdf')
    matplotlib.use('Agg')  # To avoid window popup and use without X forwarding
    filepath = os.path.split(sys.argv[0])[0]
    # Add the path to bbob_pproc/.. folder
    sys.path.append(os.path.join(filepath, os.path.pardir))
    try:
        import bbob_pproc as cocopp
    except ImportError:
        import cocopp
    res = cocopp.rungeneric2.main(sys.argv[1:])
    sys.exit(res)

from . import pproc
from . import config
from . import testbedsettings
from . import pprldistr
from . import htmldesc
from .pproc import DataSetList, processInputArgs
from .ppfig import Usage
from .toolsdivers import prepend_to_file, replace_in_file, strip_pathname1, str_to_latex
from .comp2 import ppfig2, pprldistr2, pptable2, ppscatter
from .compall import ppfigs, pprldmany
from . import ppconverrorbars
import matplotlib.pyplot as plt

__all__ = ['main']


def usage():
    print(main.__doc__)


def main(argv=None):
    r"""Routine for post-processing COCO data from two algorithms.

    Provided with some data, this routine outputs figure and TeX files
    in a folder needed for the compilation of the provided LaTeX templates
    for comparing two algorithms (``*cmp.tex`` or ``*2*.tex``).

    The used template file needs to be edited so that the command
    ``\bbobdatapath`` points to the output folder created by this routine.

    The output files will contain performance tables, performance
    scaling figures and empirical cumulative distribution figures. On
    subsequent executions, new files will be added to the output folder,
    overwriting existing older files in the process.

    Keyword arguments:

    *argv* -- list of strings containing options and arguments. If not
    given, sys.argv is accessed.

    *argv* must list folders containing BBOB data files. Each of these
    folders should correspond to the data of ONE algorithm.

    Furthermore, argv can begin with, in any order, facultative option
    flags listed below.

        -h, --help
            displays this message.
        -v, --verbose
            verbose mode, prints out operations.
        -o OUTPUTDIR, --output-dir=OUTPUTDIR
            changes the default output directory (:file:`ppdata`) to
            :file:`OUTPUTDIR`
        --noise-free, --noisy
            processes only part of the data.
        --settings=SETTING
            changes the style of the output figures and tables. At the
            moment only the only differences are in the colors of the
            output figures. SETTING can be either "grayscale", "color"
            or "black-white". The default setting is "color".
        --fig-only, --rld-only, --tab-only, --sca-only
            these options can be used to output respectively the aRT
            graphs figures, run length distribution figures or the
            comparison tables scatter plot figures only. Any combination
            of these options results in no output.
        --conv
            if this option is chosen, additionally convergence
            plots for each function and algorithm are generated.
        --no-rld-single-fcts
            do not generate runlength distribution figures for each
            single function.
        --expensive
            runlength-based f-target values and fixed display limits,
            useful with comparatively small budgets.
        --no-svg
            do not generate the svg figures which are used in html files

    Exceptions raised:

    *Usage* -- Gives back a usage message.

    Examples:

    * Calling the rungeneric2.py interface from the command line::

        $ python bbob_pproc/rungeneric2.py -v Alg0-baseline Alg1-of-interest

      will post-process the data from folders :file:`Alg0-baseline` and
      :file:`Alg1-of-interest`, the former containing data for the
      reference algorithm (zero-th) and the latter data for the
      algorithm of concern (first). The results will be output in the
      default output folder. The ``-v`` option adds verbosity.

    * From the python interpreter (requires that the path to this
      package is in python search path)::

        >> import bbob_pproc as bb
        >> bb.rungeneric2.main('-o outputfolder PSO DEPSO'.split())

    This will execute the post-processing on the data found in folder
    :file:`PSO` and :file:`DEPSO`. The ``-o`` option changes the output
    folder from the default to :file:`outputfolder`.

    """

    if argv is None:
        argv = sys.argv[1:]
        # The zero-th input argument which is the name of the calling script is
        # disregarded.

    try:

        try:
            opts, args = getopt.getopt(argv, genericsettings.shortoptlist,
                                       genericsettings.longoptlist)
        except getopt.error, msg:
            raise Usage(msg)

        if not args:
            usage()
            sys.exit()

        # Process options
        outputdir = genericsettings.outputdir
        for o, a in opts:
            if o in ("-v", "--verbose"):
                genericsettings.verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-o", "--output-dir"):
                outputdir = a
            elif o == "--fig-only":
                genericsettings.isRLDistr = False
                genericsettings.isTab = False
                genericsettings.isScatter = False
            elif o == "--rld-only":
                genericsettings.isFig = False
                genericsettings.isTab = False
                genericsettings.isScatter = False
            elif o == "--tab-only":
                genericsettings.isFig = False
                genericsettings.isRLDistr = False
                genericsettings.isScatter = False
            elif o == "--sca-only":
                genericsettings.isFig = False
                genericsettings.isRLDistr = False
                genericsettings.isTab = False
            elif o == "--noisy":
                genericsettings.isNoisy = True
            elif o == "--noise-free":
                genericsettings.isNoiseFree = True
            elif o == "--settings":
                genericsettings.inputsettings = a
            elif o == "--conv":
                genericsettings.isConv = True
            elif o == "--no-rld-single-fcts":
                genericsettings.isRldOnSingleFcts = False
            elif o == "--runlength-based":
                genericsettings.runlength_based_targets = True
            elif o == "--expensive":
                genericsettings.isExpensive = True  # comprises runlength-based
            elif o == "--no-svg":
                genericsettings.generate_svg_files = False
            elif o == "--los-only":
                warnings.warn("option --los-only will have no effect with rungeneric2.py")
            elif o == "--crafting-effort=":
                warnings.warn("option --crafting-effort will have no effect with rungeneric2.py")
            elif o in ("-p", "--pickle"):
                warnings.warn("option --pickle will have no effect with rungeneric2.py")
            else:
                assert False, "unhandled option"

        # from bbob_pproc import bbob2010 as inset # input settings
        if genericsettings.inputsettings == "color":
            from bbob_pproc import genericsettings as inset  # input settings
            config.config()
        elif genericsettings.inputsettings == "grayscale":  # probably very much obsolete
            from bbob_pproc import grayscalesettings as inset  # input settings
        elif genericsettings.inputsettings == "black-white":  # probably very much obsolete
            from bbob_pproc import bwsettings as inset  # input settings
        else:
            txt = ('Settings: %s is not an appropriate ' % genericsettings.inputsettings
                   + 'argument for input flag "--settings".')
            raise Usage(txt)

        if not genericsettings.verbose:
            warnings.simplefilter('module')
            warnings.simplefilter('ignore')

        print("\nPost-processing will generate comparison " +
              "data in folder %s" % outputdir)
        print("  this might take several minutes.")

        dsList, sortedAlgs, dictAlg = processInputArgs(args, verbose=genericsettings.verbose)

        if 1 < 3 and len(sortedAlgs) != 2:
            raise ValueError('rungeneric2.py needs exactly two algorithms to '
                             + 'compare, found: ' + str(sortedAlgs)
                             + '\n use rungeneric.py (or rungenericmany.py) to '
                             + 'compare more algorithms. ')

        if not dsList:
            sys.exit()

        if (any(ds.isBiobjective() for ds in dsList)
            and any(not ds.isBiobjective() for ds in dsList)):
            sys.exit()

        for i in dictAlg:
            if genericsettings.isNoisy and not genericsettings.isNoiseFree:
                dictAlg[i] = dictAlg[i].dictByNoise().get('nzall', DataSetList())
            if genericsettings.isNoiseFree and not genericsettings.isNoisy:
                dictAlg[i] = dictAlg[i].dictByNoise().get('noiselessall', DataSetList())

        for i in dsList:
            if i.dim not in genericsettings.dimensions_to_display:
                continue
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

        if len(sortedAlgs) < 2:
            raise Usage('Expect data from two different algorithms, could ' +
                        'only find one.')
        elif len(sortedAlgs) > 2:
            warnings.warn('Data from folders: %s ' % (sortedAlgs) +
                          'were found, the first two will be processed.')

        # Group by algorithm
        dsList0 = dictAlg[sortedAlgs[0]]
        if not dsList0:
            raise Usage('Could not find data for algorithm %s.' % (sortedAlgs[0]))

        dsList1 = dictAlg[sortedAlgs[1]]
        if not dsList1:
            raise Usage('Could not find data for algorithm %s.' % (sortedAlgs[0]))

        # get the name of each algorithm from the input arguments
        tmppath0, alg0name = os.path.split(sortedAlgs[0].rstrip(os.sep))
        tmppath1, alg1name = os.path.split(sortedAlgs[1].rstrip(os.sep))

        for i in dsList0:
            i.algId = alg0name
        for i in dsList1:
            i.algId = alg1name

        config.target_values(genericsettings.isExpensive)
        config.config(dsList[0].testbed_name())

        ######################### Post-processing #############################
        if (genericsettings.isFig or genericsettings.isRLDistr
            or genericsettings.isTab or genericsettings.isScatter):
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if genericsettings.verbose:
                    print('Folder %s was created.' % (outputdir))

            # prepend the algorithm name command to the tex-command file
            abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            lines = []
            for i, alg in enumerate(args):
                lines.append('\\providecommand{\\algorithm' + abc[i] + '}{' +
                             str_to_latex(strip_pathname1(alg)) + '}')
            prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'),
                            lines, 1000, 'bbob_proc_commands.tex truncated, '
                            + 'consider removing the file before the text run'
                            )

        # Check whether both input arguments list noisy and noise-free data
        dictFN0 = dsList0.dictByNoise()
        dictFN1 = dsList1.dictByNoise()
        k0 = set(dictFN0.keys())
        k1 = set(dictFN1.keys())
        symdiff = k1 ^ k0  # symmetric difference
        if symdiff:
            tmpdict = {}
            for i, noisegrp in enumerate(symdiff):
                if noisegrp == 'nzall':
                    tmp = 'noisy'
                elif noisegrp == 'noiselessall':
                    tmp = 'noiseless'

                if dictFN0.has_key(noisegrp):
                    tmp2 = sortedAlgs[0]
                elif dictFN1.has_key(noisegrp):
                    tmp2 = sortedAlgs[1]

                tmpdict.setdefault(tmp2, []).append(tmp)

            txt = []
            for i, j in tmpdict.iteritems():
                txt.append('Only input folder %s lists %s data.'
                           % (i, ' and '.join(j)))
            raise Usage('Data Mismatch: \n  ' + ' '.join(txt) + '\nTry using --noise-free or --noisy flags.')

        algName0 = toolsdivers.str_to_latex(
            set(i[0] for i in dsList0.dictByAlg().keys()).pop().replace(genericsettings.extraction_folder_prefix, ''))
        algName1 = toolsdivers.str_to_latex(
            set(i[0] for i in dsList1.dictByAlg().keys()).pop().replace(genericsettings.extraction_folder_prefix, ''))

        algorithm_name = "%s vs %s" % (algName1, algName0)
        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.ppfigs_file_name),
            algname=algorithm_name,
            htmlPage=ppfig.HtmlPage.PPFIGS,
            isBiobjective=dsList0.isBiobjective(),
            functionGroups=dsList0.getFuncGroups(),
            parentFileName=genericsettings.two_algorithm_file_name
        )

        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.ppscatter_file_name),
            algname=algorithm_name,
            htmlPage=ppfig.HtmlPage.PPSCATTER,
            isBiobjective=dsList0.isBiobjective(),
            functionGroups=dsList0.getFuncGroups(),
            parentFileName=genericsettings.two_algorithm_file_name
        )

        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.pprldistr2_file_name),
            algname=algorithm_name,
            htmlPage=ppfig.HtmlPage.PPRLDISTR2,
            isBiobjective=dsList0.isBiobjective(),
            functionGroups=dsList0.getFuncGroups(),
            parentFileName=genericsettings.two_algorithm_file_name
        )

        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.pptable2_file_name),
            algname=algorithm_name,
            htmlPage=ppfig.HtmlPage.PPTABLE2,
            isBiobjective=dsList0.isBiobjective(),
            functionGroups=dsList0.getFuncGroups(),
            parentFileName=genericsettings.two_algorithm_file_name
        )

        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.pptables_file_name),
            '',  # algorithms names are clearly visible in the figure
            htmlPage=ppfig.HtmlPage.PPTABLES,
            isBiobjective=dsList[0].isBiobjective(),
            functionGroups=dsList0.getFuncGroups(),
            parentFileName=genericsettings.many_algorithm_file_name
        )

        if genericsettings.isFig:
            print("log aRT1/aRT0 vs target function values...")
            plt.rc("axes", **inset.rcaxeslarger)
            plt.rc("xtick", **inset.rcticklarger)
            plt.rc("ytick", **inset.rcticklarger)
            plt.rc("font", **inset.rcfontlarger)
            plt.rc("legend", **inset.rclegendlarger)
            plt.rc('pdf', fonttype=42)
            ppfig2.main(dsList0, dsList1, testbedsettings.current_testbed.ppfig2_ftarget,
                        outputdir, genericsettings.verbose)
            print_done()

        plt.rc("axes", **inset.rcaxes)
        plt.rc("xtick", **inset.rctick)
        plt.rc("ytick", **inset.rctick)
        plt.rc("font", **inset.rcfont)
        plt.rc("legend", **inset.rclegend)
        plt.rc('pdf', fonttype=42)

        if genericsettings.isRLDistr:
            print("ECDF runlength ratio graphs...")
            if len(dictFN0) > 1 or len(dictFN1) > 1:
                warnings.warn('Data for functions from both the noisy and ' +
                              'non-noisy testbeds have been found. Their ' +
                              'results will be mixed in the "all functions" ' +
                              'ECDF figures.')
            dictDim0 = dsList0.dictByDim()
            dictDim1 = dsList1.dictByDim()

            # ECDFs of aRT ratios
            for dim in set(dictDim0.keys()) & set(dictDim1.keys()):
                if dim in inset.rldDimsOfInterest:
                    # ECDF for all functions altogether
                    try:
                        pprldistr2.main(dictDim0[dim], dictDim1[dim], dim,
                                        testbedsettings.current_testbed.rldValsOfInterest,
                                        outputdir,
                                        '%02dD_all' % dim,
                                        genericsettings.verbose)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.' % dim)
                        continue

                    # ECDFs per function groups
                    dictFG0 = dictDim0[dim].dictByFuncGroup()
                    dictFG1 = dictDim1[dim].dictByFuncGroup()

                    for fGroup in set(dictFG0.keys()) & set(dictFG1.keys()):
                        pprldistr2.main(dictFG1[fGroup], dictFG0[fGroup], dim,
                                        testbedsettings.current_testbed.rldValsOfInterest,
                                        outputdir,
                                        '%02dD_%s' % (dim, fGroup),
                                        genericsettings.verbose)

                    # ECDFs per noise groups
                    dictFN0 = dictDim0[dim].dictByNoise()
                    dictFN1 = dictDim1[dim].dictByNoise()

                    for fGroup in set(dictFN0.keys()) & set(dictFN1.keys()):
                        pprldistr2.main(dictFN1[fGroup], dictFN0[fGroup], dim,
                                        testbedsettings.current_testbed.rldValsOfInterest,
                                        outputdir,
                                        '%02dD_%s' % (dim, fGroup),
                                        genericsettings.verbose)

            prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'),
                            ['\\providecommand{\\bbobpprldistrlegendtwo}[1]{',
                             pprldistr.caption_two(),  # depends on the config
                             # setting, should depend
                             # on maxfevals
                             '}'
                             ])
            print_done()

            # ECDFs per noise groups, code copied from rungenericmany.py
            # (needed for bbob-biobj multiple algo template)
            print("ECDF graphs per noise group...")
            rungenericmany.grouped_ecdf_graphs(
                pproc.dictAlgByNoi(dictAlg),
                dsList[0].isBiobjective(),
                sortedAlgs,
                outputdir,
                dictAlg[sortedAlgs[0]].getFuncGroups())
            print_done()

            # ECDFs per function groups, code copied from rungenericmany.py
            # (needed for bbob-biobj multiple algo template)
            print("ECDF runlength graphs per function group...")
            rungenericmany.grouped_ecdf_graphs(
                pproc.dictAlgByFuncGroup(dictAlg),
                dsList[0].isBiobjective(),
                sortedAlgs,
                outputdir,
                dictAlg[sortedAlgs[0]].getFuncGroups())
            print_done()

            print("ECDF runlength graphs...")
            for dim in set(dictDim0.keys()) & set(dictDim1.keys()):
                pprldistr.fmax = None  # Resetting the max final value
                pprldistr.evalfmax = None  # Resetting the max #fevalsfactor
                # ECDFs of all functions altogether
                if dim in inset.rldDimsOfInterest:
                    try:
                        pprldistr.comp(dictDim1[dim], dictDim0[dim],
                                       testbedsettings.current_testbed.rldValsOfInterest,
                                       # TODO: let rldVals... possibly be RL-based targets
                                       True,
                                       outputdir, 'all', genericsettings.verbose)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.'
                                      % (dim))
                        continue

                    # ECDFs per function groups
                    dictFG0 = dictDim0[dim].dictByFuncGroup()
                    dictFG1 = dictDim1[dim].dictByFuncGroup()

                    for fGroup in set(dictFG0.keys()) & set(dictFG1.keys()):
                        pprldistr.comp(dictFG1[fGroup], dictFG0[fGroup],
                                       testbedsettings.current_testbed.rldValsOfInterest, True,
                                       outputdir,
                                       '%s' % fGroup, genericsettings.verbose)

                    # ECDFs per noise groups
                    dictFN0 = dictDim0[dim].dictByNoise()
                    dictFN1 = dictDim1[dim].dictByNoise()
                    for fGroup in set(dictFN0.keys()) & set(dictFN1.keys()):
                        pprldistr.comp(dictFN1[fGroup], dictFN0[fGroup],
                                       testbedsettings.current_testbed.rldValsOfInterest, True,
                                       outputdir,
                                       '%s' % fGroup, genericsettings.verbose)
            print_done()

            # copy-paste from above, here for each function instead of function groups
            if genericsettings.isRldOnSingleFcts:
                print("ECDF graphs per function...")
                # ECDFs for each function
                pprldmany.all_single_functions(dictAlg,
                                               dsList[0].isBiobjective(),
                                               False,
                                               sortedAlgs,
                                               outputdir,
                                               genericsettings.verbose,
                                               genericsettings.two_algorithm_file_name)
                print_done()

        if genericsettings.isConv:
            print("Convergence plots...")
            ppconverrorbars.main(dictAlg,
                                 dsList[0].isBiobjective(),
                                 outputdir,
                                 genericsettings.verbose,
                                 genericsettings.two_algorithm_file_name)
            print_done()

        htmlFileName = os.path.join(outputdir, genericsettings.ppscatter_file_name + '.html')

        if genericsettings.isScatter:
            print("Scatter plots...")
            ppscatter.main(dsList1, dsList0, outputdir,
                           verbose=genericsettings.verbose)
            prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'),
                            ['\\providecommand{\\bbobppscatterlegend}[1]{',
                             ppscatter.figure_caption(),
                             '}'
                             ])

            replace_in_file(htmlFileName, '##bbobppscatterlegend##', ppscatter.figure_caption(True))
            for i, alg in enumerate(args):
                replace_in_file(htmlFileName, 'algorithm'
                                + abc[i], str_to_latex(strip_pathname1(alg)))

            print_done()

        if genericsettings.isTab:
            print("Generating old tables (pptable2.py)...")
            dictNG0 = dsList0.dictByNoise()
            dictNG1 = dsList1.dictByNoise()

            for nGroup in set(dictNG0.keys()) & set(dictNG1.keys()):
                # split table in as many as necessary
                dictFunc0 = dictNG0[nGroup].dictByFunc()
                dictFunc1 = dictNG1[nGroup].dictByFunc()
                funcs = list(set(dictFunc0.keys()) & set(dictFunc1.keys()))
                if len(funcs) > 24:
                    funcs.sort()
                    nbgroups = int(numpy.ceil(len(funcs) / 24.))

                    def split_seq(seq, nbgroups):
                        newseq = []
                        splitsize = 1.0 / nbgroups * len(seq)
                        for i in range(nbgroups):
                            newseq.append(seq[int(round(i * splitsize)):int(round((i + 1) * splitsize))])
                        return newseq

                    groups = split_seq(funcs, nbgroups)
                    # merge
                    group0 = []
                    group1 = []
                    for i, g in enumerate(groups):
                        tmp0 = DataSetList()
                        tmp1 = DataSetList()
                        for f in g:
                            tmp0.extend(dictFunc0[f])
                            tmp1.extend(dictFunc1[f])
                        group0.append(tmp0)
                        group1.append(tmp1)
                    for i, g in enumerate(zip(group0, group1)):
                        pptable2.main(g[0], g[1], inset.tabDimsOfInterest,
                                      outputdir,
                                      '%s%d' % (nGroup, i), genericsettings.verbose)
                else:
                    if 11 < 3:  # future handling:
                        dictFunc0 = dsList0.dictByFunc()
                        dictFunc1 = dsList1.dictByFunc()
                        funcs = list(set(dictFunc0.keys()) & set(dictFunc1.keys()))
                        funcs.sort()
                    # nbgroups = int(numpy.ceil(len(funcs)/testbedsettings.numberOfFunctions))
                    #                        pptable2.main(dsList0, dsList1,
                    #                                      testbedsettings.tabDimsOfInterest, outputdir,
                    #                                      '%s' % (testbedsettings.testbedshortname), genericsettings.verbose)
                    else:
                        pptable2.main(dictNG0[nGroup], dictNG1[nGroup],
                                      inset.tabDimsOfInterest,
                                      outputdir,
                                      '%s' % (nGroup), genericsettings.verbose)

            prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'),
                            ['\\providecommand{\\bbobpptablestwolegend}[1]{',
                             pptable2.get_table_caption(),
                             '}'
                             ])

            htmlFileName = os.path.join(outputdir, genericsettings.pptable2_file_name + '.html')
            key = '##bbobpptablestwolegend%s##' % (testbedsettings.current_testbed.scenario)
            replace_in_file(htmlFileName, '##bbobpptablestwolegend##', htmldesc.getValue(key))

            replace_in_file(htmlFileName, 'algorithmAshort', algName0[0:3])
            replace_in_file(htmlFileName, 'algorithmBshort', algName1[0:3])

            for htmlFileName in (genericsettings.pprldistr2_file_name,
                                 genericsettings.pptable2_file_name):
                for i, alg in enumerate(args):
                    replace_in_file(os.path.join(outputdir, htmlFileName + '.html'),
                                    'algorithm' + abc[i], str_to_latex(strip_pathname1(alg)))

            print_done()

            # The following is copied from rungenericmany.py to comply
            # with the bi-objective many-algorithm LaTeX template
            print("Generating new tables (pptables.py)...")
            prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'),
                            ['\providecommand{\\bbobpptablesmanylegend}[2]{' +
                             pptables.get_table_caption() + '}'])
            dictNoi = pproc.dictAlgByNoi(dictAlg)
            for ng, tmpdictng in dictNoi.iteritems():
                dictDim = pproc.dictAlgByDim(tmpdictng)
                for d, tmpdictdim in sorted(dictDim.iteritems()):
                    pptables.main(
                        tmpdictdim,
                        sortedAlgs,
                        outputdir,
                        genericsettings.verbose,
                        ([1, 20, 38] if (testbedsettings.current_testbed.name ==
                                         testbedsettings.testbed_name_bi) else True))
            print_done()

        if genericsettings.isScaleUp:
            print("Scaling figures...")
            plt.rc("axes", labelsize=20, titlesize=24)
            plt.rc("xtick", labelsize=20)
            plt.rc("ytick", labelsize=20)
            plt.rc("font", size=20)
            plt.rc("legend", fontsize=20)
            plt.rc('pdf', fonttype=42)

            ppfigs.main(dictAlg,
                        genericsettings.ppfigs_file_name,
                        dsList[0].isBiobjective(),
                        sortedAlgs,
                        outputdir,
                        genericsettings.verbose)
            plt.rcdefaults()
            print_done()

        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.two_algorithm_file_name),
            algname=algorithm_name,
            htmlPage=ppfig.HtmlPage.TWO,
            isBiobjective=dsList0.isBiobjective(),
            functionGroups=dsList0.getFuncGroups())

        if (genericsettings.isFig or genericsettings.isRLDistr
            or genericsettings.isTab or genericsettings.isScatter
            or genericsettings.isScaleUp):
            print("Output data written to folder %s" % outputdir)

        plt.rcdefaults()

    except Usage, err:
        print(err.msg, file=sys.stderr)
        print("For help use -h or --help", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
