#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process data to be included in a generic template.

Synopsis:
    ``python path_to_folder/bbob_pproc/rungenericmany.py [OPTIONS] FOLDER``
Help:
    ``python path_to_folder/bbob_pproc/rungenericmany.py -h``

"""

from __future__ import absolute_import

import os
import sys
import glob
import getopt
import pickle
import tarfile
from pdb import set_trace
import warnings
import numpy

ftarget = 1e-8
target_runlength = 10 # used for ppfigs.main

# Add the path to bbob_pproc
if __name__ == "__main__":
    (filepath, filename) = os.path.split(sys.argv[0])
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import genericsettings, ppfig
from bbob_pproc import dataoutput, pproc, pptex
from bbob_pproc.pproc import DataSetList, processInputArgs
from bbob_pproc.toolsdivers import prepend_to_file, strip_pathname1, str_to_latex
from bbob_pproc.compall import pprldmany, pptables, ppfigs
from bbob_pproc import ppconverrorbars

import matplotlib.pyplot as plt

__all__ = ['main']

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

#FUNCTION DEFINITIONS

def usage():
    print main.__doc__

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
            figures of ERT/dim vs dim only. A combination of any two or
            more of these options results in no output.
        --conv
            if this option is chosen, additionally convergence
            plots for each function and algorithm are generated.
        --rld-single-fcts
            generate also runlength distribution figures for each
            single function. 
        --expensive
            runlength-based f-target values and fixed display limits,
            useful with comparatively small budgets. By default the
            setting is based on the budget used in the data.
        --not-expensive
            expensive setting off. 
        --svg
            generate also the svg figures which are used in html files 
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
            opts, args = getopt.getopt(argv, genericsettings.shortoptlist, genericsettings.longoptlist)
        except getopt.error, msg:
            raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        #Process options
        outputdir = genericsettings.outputdir
        for o, a in opts:
            if o in ("-v","--verbose"):
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
            #The next 3 are for testing purpose
            elif o == "--tab-only":
                genericsettings.isRLDistr = False
                genericsettings.isFig = False
            elif o == "--rld-single-fcts":
                genericsettings.isRldOnSingleFcts = True
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
            elif o == "--not-expensive":
                genericsettings.isExpensive = False  
            elif o == "--svg":
                genericsettings.generate_svg_files = True
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
            from bbob_pproc import config, genericsettings as inset # input settings
            config.config()
        elif genericsettings.inputsettings == "grayscale":
            # this settings strategy (by proving different settings files) is problematic, 
            # because it means copy-paste of the settings
            # file and future changes have a great chance to make the pasted files incompatible
            # as has most likely happened with grayscalesettings:
            from bbob_pproc import config, grayscalesettings as inset # input settings
            # better would be just adjust the previous settings, as config is doing it, 
            # so a config_grayscalesettings.py module seems the better approach to go 
        elif genericsettings.inputsettings == "black-white":
            from bbob_pproc import config, bwsettings as inset # input settings
        else:
            txt = ('Settings: %s is not an appropriate ' % genericsettings.inputsettings
                   + 'argument for input flag "--settings".')
            raise Usage(txt)

        if (not genericsettings.verbose):
            warnings.filterwarnings('module', '.*', Warning, '.*')  # same warning just once
            warnings.simplefilter('ignore')  # that is bad, but otherwise to many warnings appear 

        config.target_values(genericsettings.isExpensive)
        
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

    if 1 < 3:
        print ("Post-processing: will generate output " +
               "data in folder %s" % outputdir)
        print "  this might take several minutes."

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            if genericsettings.verbose:
                print 'Folder %s was created.' % (outputdir)

        # prepend the algorithm name command to the tex-command file
        lines = []
        for i, alg in enumerate(args):
            lines.append('\\providecommand{\\algorithm' + pptex.numtotext(i) + 
                    '}{' +  str_to_latex(strip_pathname1(alg)) + '}')
        prepend_to_file(os.path.join(outputdir,
                    'bbob_pproc_commands.tex'), 
                    lines, 5000, 
                    'bbob_proc_commands.tex truncated, consider removing the file before the text run'
                    )

        dsList, sortedAlgs, dictAlg = processInputArgs(args, verbose=genericsettings.verbose)

        if not dsList:
            sys.exit()

        for i in dictAlg:
            if genericsettings.isNoisy and not genericsettings.isNoiseFree:
                dictAlg[i] = dictAlg[i].dictByNoise().get('nzall', DataSetList())
            if genericsettings.isNoiseFree and not genericsettings.isNoisy:
                dictAlg[i] = dictAlg[i].dictByNoise().get('noiselessall', DataSetList())



        # compute maxfuneval values
        # TODO: we should rather take min_algorithm max_evals
        dict_max_fun_evals = {}
        for ds in dsList:
            dict_max_fun_evals[ds.dim] = numpy.max((dict_max_fun_evals.setdefault(ds.dim, 0), float(numpy.max(ds.maxevals))))
            
        # set target values
        from bbob_pproc import config
        config.target_values(genericsettings.isExpensive, dict_max_fun_evals)
        config.config()


        for i in dsList:
            if i.dim not in genericsettings.dimensions_to_display:
                continue

            if (dict((j, i.instancenumbers.count(j)) for j in set(i.instancenumbers)) <
                inset.instancesOfInterest):
                warnings.warn('The data of %s do not list ' %(i) +
                              'the correct instances ' +
                              'of function F%d.' %(i.funcId))

        plt.rc("axes", **inset.rcaxes)
        plt.rc("xtick", **inset.rctick)
        plt.rc("ytick", **inset.rctick)
        plt.rc("font", **inset.rcfont)
        plt.rc("legend", **inset.rclegend)
        plt.rc('pdf', fonttype = 42)

        ppfig.save_single_functions_html(
            os.path.join(outputdir, genericsettings.many_algorithm_file_name),
            '', # algorithms names are clearly visible in the figure
            algorithmCount=ppfig.AlgorithmCount.MANY
        )

        ppfig.copy_js_files(outputdir)
        
        # convergence plots
        if genericsettings.isConv:
            ppconverrorbars.main(dictAlg, outputdir, genericsettings.verbose)
        # empirical cumulative distribution functions (ECDFs) aka Data profiles
        if genericsettings.isRLDistr:
            config.config()
            # ECDFs per noise groups
            dictNoi = pproc.dictAlgByNoi(dictAlg)
            for ng, tmpdictAlg in dictNoi.iteritems():
                dictDim = pproc.dictAlgByDim(tmpdictAlg)
                for d, entries in dictDim.iteritems():
                    # pprldmany.main(entries, inset.summarized_target_function_values,
                    # from . import config
                    # config.config()
                    pprldmany.main(entries, # pass expensive flag here? 
                                   order=sortedAlgs,
                                   outputdir=outputdir,
                                   info=('%02dD_%s' % (d, ng)),
                                   verbose=genericsettings.verbose)
            # ECDFs per function groups
            dictFG = pproc.dictAlgByFuncGroup(dictAlg)
            for fg, tmpdictAlg in dictFG.iteritems():
                dictDim = pproc.dictAlgByDim(tmpdictAlg)
                for d, entries in dictDim.iteritems():
                    pprldmany.main(entries,
                                   order=sortedAlgs,
                                   outputdir=outputdir,
                                   info=('%02dD_%s' % (d, fg)),
                                   verbose=genericsettings.verbose)
            if genericsettings.isRldOnSingleFcts: # copy-paste from above, here for each function instead of function groups
                # ECDFs for each function
                if 1 < 3:
                    pprldmany.all_single_functions(dictAlg, sortedAlgs,
                            outputdir, genericsettings.verbose)
                else:  # subject to removal
                    dictFG = pproc.dictAlgByFun(dictAlg)
                    for fg, tmpdictAlg in dictFG.iteritems():
                        dictDim = pproc.dictAlgByDim(tmpdictAlg)
                        for d, entries in dictDim.iteritems():
                            single_fct_output_dir = (outputdir.rstrip(os.sep) + os.sep +
                                                     'pprldmany-single-functions'
                                                     # + os.sep + ('f%03d' % fg)
                                                     )
                            if not os.path.exists(single_fct_output_dir):
                                os.makedirs(single_fct_output_dir)
                            pprldmany.main(entries,
                                           order=sortedAlgs,
                                           outputdir=single_fct_output_dir,
                                           info=('f%03d_%02dD' % (fg, d)),
                                           verbose=genericsettings.verbose)
            print "ECDFs of run lengths figures done."

        if genericsettings.isTab:
            if genericsettings.isExpensive:
                prepend_to_file(os.path.join(outputdir,
                            'bbob_pproc_commands.tex'), 
                            ['\providecommand{\\bbobpptablesmanylegend}[1]{' + 
                             pptables.tables_many_expensive_legend + '}'])
            else:
                prepend_to_file(os.path.join(outputdir,
                            'bbob_pproc_commands.tex'), 
                            ['\providecommand{\\bbobpptablesmanylegend}[1]{' + 
                             pptables.tables_many_legend + '}'])
            dictNoi = pproc.dictAlgByNoi(dictAlg)
            for ng, tmpdictng in dictNoi.iteritems():
                dictDim = pproc.dictAlgByDim(tmpdictng)
                for d, tmpdictdim in dictDim.iteritems():
                    pptables.main(tmpdictdim, sortedAlgs,
                                  outputdir, genericsettings.verbose)
            print "Comparison tables done."

        global ftarget  # not nice
        if genericsettings.isFig:
            plt.rc("axes", labelsize=20, titlesize=24)
            plt.rc("xtick", labelsize=20)
            plt.rc("ytick", labelsize=20)
            plt.rc("font", size=20)
            plt.rc("legend", fontsize=20)
            plt.rc('pdf', fonttype = 42)
            if genericsettings.runlength_based_targets:
                ftarget = pproc.RunlengthBasedTargetValues([target_runlength])  # TODO: make this more variable but also consistent
            ppfigs.main(dictAlg, genericsettings.many_algorithm_file_name, sortedAlgs, ftarget,
                        outputdir, genericsettings.verbose)
            plt.rcdefaults()
            print "Scaling figures done."

        plt.rcdefaults()

if __name__ == "__main__":
    sys.exit(main())

