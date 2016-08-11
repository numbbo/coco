#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process data to be included in a generic template.

Synopsis:
    ``python path_to_folder/bbob_pproc/rungenericmany.py [OPTIONS] FOLDER``
Help:
    ``python path_to_folder/bbob_pproc/rungenericmany.py -h``

"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import getopt
import warnings
import matplotlib

if __name__ == "__main__":
    matplotlib.use('Agg')  # To avoid window popup and use without X forwarding
    # matplotlib.use('pdf')
    filepath = os.path.split(sys.argv[0])[0]
    # Add the path to bbob_pproc/.. folder
    sys.path.append(os.path.join(filepath, os.path.pardir))
    try:
        import bbob_pproc as cocopp
    except ImportError:
        import cocopp
    res = cocopp.rungenericmany.main(sys.argv[1:])
    sys.exit(res)

from . import genericsettings, ppfig, testbedsettings
from . import pproc, pptex
from .pproc import DataSetList, processInputArgs
from .ppfig import Usage
from .toolsdivers import prepend_to_file, strip_pathname1, str_to_latex, replace_in_file
from .compall import pprldmany, pptables, ppfigs
from . import ppconverrorbars

import matplotlib.pyplot as plt
from .toolsdivers import print_done

__all__ = ['main']


def usage():
    print(main.__doc__)


def grouped_ecdf_graphs(alg_dict, is_biobjective, order, output_dir, function_groups):
    """ Generates ecdf graphs, aggregated over groups as
        indicated via algdict
    """
    for gr, tmpdictAlg in alg_dict.iteritems():
        dictDim = pproc.dictAlgByDim(tmpdictAlg)
        dims = sorted(dictDim)
        for i, d in enumerate(dims):
            entries = dictDim[d]
            next_dim = dims[i+1] if i + 1 < len(dims) else dims[0]

            ppfig.save_single_functions_html(
                os.path.join(output_dir, genericsettings.pprldmany_file_name),
                '',  # algorithms names are clearly visible in the figure
                add_to_names='_%02dD' % d,
                next_html_page_suffix='_%02dD' % next_dim,
                htmlPage=ppfig.HtmlPage.PPRLDMANY_BY_GROUP_MANY,
                isBiobjective=is_biobjective,
                functionGroups=function_groups,
                parentFileName=genericsettings.many_algorithm_file_name
            )

            pprldmany.main(entries,  # pass expensive flag here?
                           is_biobjective,
                           order=order,
                           outputdir=output_dir,
                           info=('%02dD_%s' % (d, gr)),
                           verbose=genericsettings.verbose,
                           add_to_html_file_name='_%02dD' % d,
                           next_html_page_suffix='_%02dD' % next_dim
                           )

            file_name = os.path.join(output_dir, '%s_%02dD.html' % (genericsettings.pprldmany_file_name, d))
            replace_in_file(file_name, '##bbobECDFslegend##', ppfigs.ecdfs_figure_caption(True, d))


def main(argv=None):
    r"""Main routine for post-processing the data of multiple algorithms.

    Provided with some data, this routine outputs figure and TeX files
    in a folder needed for the compilation of the provided LaTeX templates
    for comparing multiple algorithms (``*many.tex`` or ``*3*.tex``).
    The used template file needs to be edited so that the commands
    ``\bbobdatapath`` points to the output folder created by this routine.

    These output files will contain performance tables, performance
    scaling figures and empirical cumulative distribution figures. On
    subsequent executions, new files will be added to the output folder,
    overwriting existing older files in the process.

    Keyword arguments:

    *argv* -- list of strings containing options and arguments. If not
    provided, sys.argv is accessed.

    *argv* must list folders containing BBOB data files.
    The name of these folders will be used in the output figures and
    tables to designate the algorithms. Therefore you should name the
    folders with differentiating names.

    Furthermore, argv can begin with facultative option flags listed
    below.

        -h, --help
            displays this message.
        -v, --verbose
            verbose mode, prints out operations, warnings.
        -o OUTPUTDIR, --output-dir=OUTPUTDIR
            changes the default output directory (:file:`ppdatamany`) to
            :file:`OUTPUTDIR`.
        --noise-free, --noisy
            processes only part of the data.
        --settings=SETTINGS
            changes the style of the output figures and tables. At the
            moment, the only differences are in the colors of the
            output figures. SETTINGS can be either "grayscale", "color"
            or "black-white". The default setting is "color".
        --tab-only, --rld-only, --fig-only
            these options can be used to output respectively the
            comparison TeX tables, the run lengths distributions or the
            figures of aRT/dim vs dim only. A combination of any two or
            more of these options results in no output.
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
        -

    Exceptions raised:

    *Usage* -- Gives back a usage message.

    Examples:

    * Calling the rungenericmany.py interface from the command line::

        $ python bbob_pproc/rungenericmany.py -v AMALGAM BFGS BIPOP-CMA-ES


    * Loading this package and calling the main from the command line
      (requires that the path to this package is in python search path)::

        $ python -m bbob_pproc.rungenericmany -h

      This will print out this help message.

    * From the python interpreter (requires that the path to this
      package is in python search path)::

        >> import bbob_pproc as bb
        >> bb.rungenericmany.main('-o outputfolder folder1 folder2'.split())

      This will execute the post-processing on the data found in
      :file:`folder1` and :file:`folder2`.
      The ``-o`` option changes the output folder from the default to
      :file:`outputfolder`.

    * Generate post-processing data for some algorithms with figures in
      shades of gray::

        $ python rungenericmany.py --settings grayscale NEWUOA NELDER LSSTEP

    """

    if argv is None:
        argv = sys.argv[1:]

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
            elif o == "--noisy":
                genericsettings.isNoisy = True
            elif o == "--noise-free":
                genericsettings.isNoiseFree = True
            # The next 3 are for testing purpose
            elif o == "--tab-only":
                genericsettings.isRLDistr = False
                genericsettings.isFig = False
            elif o == "--no-rld-single-fcts":
                genericsettings.isRldOnSingleFcts = False
            elif o == "--rld-only":
                genericsettings.isTab = False
                genericsettings.isFig = False
            elif o == "--fig-only":
                genericsettings.isRLDistr = False
                genericsettings.isTab = False
            elif o == "--settings":
                genericsettings.inputsettings = a
            elif o == "--conv":
                genericsettings.isConv = True
            elif o == "--runlength-based":
                genericsettings.runlength_based_targets = True
            elif o == "--expensive":
                genericsettings.isExpensive = True  # comprises runlength-based
            elif o == "--no-svg":
                genericsettings.generate_svg_files = False
            elif o == "--sca-only":
                warnings.warn("option --sca-only will have no effect with rungenericmany.py")
            elif o == "--los-only":
                warnings.warn("option --los-only will have no effect with rungenericmany.py")
            elif o == "--crafting-effort=":
                warnings.warn("option --crafting-effort will have no effect with rungenericmany.py")
            elif o in ("-p", "--pickle"):
                warnings.warn("option --pickle will have no effect with rungenericmany.py")
            else:
                assert False, "unhandled option"

        # from bbob_pproc import bbob2010 as inset # input settings
        # TODO: conditional imports are NOT the way to go here
        if genericsettings.inputsettings == "color":
            from . import config, genericsettings as inset  # input settings
            config.config()
        elif genericsettings.inputsettings == "grayscale":
            # this settings strategy (by proving different settings files) is problematic,
            # because it means copy-paste of the settings
            # file and future changes have a great chance to make the pasted files incompatible
            # as has most likely happened with grayscalesettings:
            from . import config, grayscalesettings as inset  # input settings
            # better would be just adjust the previous settings, as config is doing it,
            # so a config_grayscalesettings.py module seems the better approach to go
        elif genericsettings.inputsettings == "black-white":
            from . import config, bwsettings as inset  # input settings
        else:
            txt = ('Settings: %s is not an appropriate ' % genericsettings.inputsettings
                   + 'argument for input flag "--settings".')
            raise Usage(txt)

        if not genericsettings.verbose:
            warnings.filterwarnings('module', '.*', Warning, '.*')  # same warning just once
            #warnings.simplefilter('ignore')  # that is bad, but otherwise to many warnings appear

        config.target_values(genericsettings.isExpensive)

    except Usage, err:
        print(err.msg, file=sys.stderr)
        print("for help use -h or --help", file=sys.stderr)
        return 2

    if 1 < 3:
        print("\nPost-processing: will generate output " +
              "data in folder %s" % outputdir)
        print("  this might take several minutes.")

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            if genericsettings.verbose:
                print('Folder %s was created.' % outputdir)

        # prepend the algorithm name command to the tex-command file
        lines = []
        for i, alg in enumerate(args):
            lines.append('\\providecommand{\\algorithm' + pptex.numtotext(i) +
                         '}{' + str_to_latex(strip_pathname1(alg)) + '}')
        prepend_to_file(os.path.join(outputdir,
                                     'bbob_pproc_commands.tex'), lines, 5000,
                        'bbob_proc_commands.tex truncated, consider removing '
                        + 'the file before the text run'
                        )

        dsList, sortedAlgs, dictAlg = processInputArgs(args, verbose=genericsettings.verbose)

        if not dsList:
            sys.exit()

        if any(ds.isBiobjective() for ds in dsList) and any(not ds.isBiobjective() for ds in dsList):
            sys.exit()

        for i in dictAlg:
            if genericsettings.isNoisy and not genericsettings.isNoiseFree:
                dictAlg[i] = dictAlg[i].dictByNoise().get('nzall', DataSetList())
            if genericsettings.isNoiseFree and not genericsettings.isNoisy:
                dictAlg[i] = dictAlg[i].dictByNoise().get('noiselessall', DataSetList())

        # set target values
        from . import config
        config.target_values(genericsettings.isExpensive)
        config.config(dsList[0].testbed_name())

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

        plt.rc("axes", **inset.rcaxes)
        plt.rc("xtick", **inset.rctick)
        plt.rc("ytick", **inset.rctick)
        plt.rc("font", **inset.rcfont)
        plt.rc("legend", **inset.rclegend)
        plt.rc('pdf', fonttype=42)

        ppfig.copy_js_files(outputdir)

        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.ppfigs_file_name),
            '',  # algorithms names are clearly visible in the figure
            htmlPage=ppfig.HtmlPage.PPFIGS,
            isBiobjective=dsList[0].isBiobjective(),
            functionGroups=dictAlg[sortedAlgs[0]].getFuncGroups(),
            parentFileName=genericsettings.many_algorithm_file_name
        )

        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.pptables_file_name),
            '',  # algorithms names are clearly visible in the figure
            htmlPage=ppfig.HtmlPage.PPTABLES,
            isBiobjective=dsList[0].isBiobjective(),
            functionGroups=dictAlg[sortedAlgs[0]].getFuncGroups(),
            parentFileName=genericsettings.many_algorithm_file_name
        )


        # convergence plots
        print("Generating convergence plots...")
        if genericsettings.isConv:
            ppconverrorbars.main(dictAlg,
                                 dsList[0].isBiobjective(),
                                 outputdir,
                                 genericsettings.verbose,
                                 genericsettings.many_algorithm_file_name)
        print_done()

        # empirical cumulative distribution functions (ECDFs) aka Data profiles
        if genericsettings.isRLDistr:
            config.config(dsList[0].testbed_name())

            # ECDFs per noise groups
            print("ECDF graphs per noise group...")
            grouped_ecdf_graphs(pproc.dictAlgByNoi(dictAlg),
                                dsList[0].isBiobjective(),
                                sortedAlgs,
                                outputdir,
                                dictAlg[sortedAlgs[0]].getFuncGroups())
            print_done()

            # ECDFs per function groups
            print("ECDF graphs per function group...")
            grouped_ecdf_graphs(pproc.dictAlgByFuncGroup(dictAlg),
                                dsList[0].isBiobjective(),
                                sortedAlgs,
                                outputdir,
                                dictAlg[sortedAlgs[0]].getFuncGroups())
            print_done()

            # copy-paste from above, here for each function instead of function groups:
            print("ECDF graphs per function...")
            if genericsettings.isRldOnSingleFcts:
                # ECDFs for each function
                if 1 < 3:
                    pprldmany.all_single_functions(dictAlg,
                                                   dsList[0].isBiobjective(),
                                                   False,
                                                   sortedAlgs,
                                                   outputdir,
                                                   genericsettings.verbose,
                                                   genericsettings.many_algorithm_file_name)
                else:  # subject to removal
                    dictFG = pproc.dictAlgByFun(dictAlg)
                    for fg, tmpdictAlg in dictFG.iteritems():
                        dictDim = pproc.dictAlgByDim(tmpdictAlg)
                        dims = sorted(dictDim)
                        for i, d in enumerate(dims):
                            entries = dictDim[d]
                            next_dim = dims[i + 1] if i + 1 < len(dims) else dims[0]
                            single_fct_output_dir = (outputdir.rstrip(os.sep) + os.sep +
                                                     'pprldmany-single-functions',
                                                     # + os.sep + ('f%03d' % fg),
                                                     dsList[0].isBiobjective()
                                                     )
                            if not os.path.exists(single_fct_output_dir):
                                os.makedirs(single_fct_output_dir)
                            pprldmany.main(entries,
                                           order=sortedAlgs,
                                           outputdir=single_fct_output_dir,
                                           info=('f%03d_%02dD' % (fg, d)),
                                           verbose=genericsettings.verbose,
                                           add_to_html_file_name='_%02dD' % d,
                                           next_html_page_suffix='_%02dD' % next_dim
                                           )
            print_done()

        if genericsettings.isTab:
            print("Generating comparison tables...")
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

        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.many_algorithm_file_name),
            '',  # algorithms names are clearly visible in the figure
            htmlPage=ppfig.HtmlPage.MANY,
            isBiobjective=dsList[0].isBiobjective(),
            functionGroups=dictAlg[sortedAlgs[0]].getFuncGroups()
        )

        if genericsettings.isFig:
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

        plt.rcdefaults()


if __name__ == "__main__":
    sys.exit(main())
