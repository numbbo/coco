#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Routines for the comparison of 2 algorithms.

Synopsis:
    ``python path_to_folder/bbob_pproc/rungeneric2.py [OPTIONS] FOLDER_NAME1 FOLDER_NAME2...``

Help:
    ``python path_to_folder/bbob_pproc/rungeneric2.py -h``

"""

from __future__ import absolute_import

import os
import sys
import glob
import warnings
import getopt
from pdb import set_trace
import numpy
import numpy as np
from bbob_pproc import pproc

ftarget = 1e-8  # used for ppfigs.main 
ppfig2_ftarget = 1e-8  # a hack, used in ppfig2.main 
target_runlength = 10 # used for ppfigs.main

# genericsettings.summarized_target_function_values[0] might be another option

# Add the path to bbob_pproc
if __name__ == "__main__":
    # append path without trailing '/bbob_pproc', using os.sep fails in mingw32
    #sys.path.append(filepath.replace('\\', '/').rsplit('/', 1)[0])
    (filepath, filename) = os.path.split(sys.argv[0])
    #Test system independent method:
    sys.path.append(os.path.join(filepath, os.path.pardir))
    import matplotlib
    # matplotlib.use('pdf')
    matplotlib.use('Agg') # To avoid window popup and use without X forwarding

from bbob_pproc import genericsettings, config
from bbob_pproc import pprldistr
from bbob_pproc.pproc import DataSetList, processInputArgs, TargetValues, RunlengthBasedTargetValues
from bbob_pproc.toolsdivers import prepend_to_file, strip_pathname, str_to_latex
from bbob_pproc.comp2 import ppfig2, pprldistr2, pptable2, ppscatter
from bbob_pproc.compall import ppfigs
from bbob_pproc import ppconverrorbars
import matplotlib.pyplot as plt

__all__ = ['main']

# Used by getopt:
shortoptlist = "hvo:"
longoptlist = ["help", "output-dir=", "noisy", "noise-free", "fig-only",
               "rld-only", "tab-only", "sca-only", "verbose",
               "settings=", "conv",
               "runlength-based", "expensive", "not-expensive"]

#CLASS DEFINITIONS

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


#FUNCTION DEFINITIONS

def usage():
    print main.__doc__

def main(argv=None):
    r"""Routine for post-processing COCO data from two algorithms.

    Provided with some data, this routine outputs figure and TeX files
    in a folder needed for the compilation of latex document
    :file:`template2XXX.tex` or :file:`noisytemplate2XXX.tex`, where
    :file:`XXX` is either :file:`ecj` or :file:`generic`. The template
    file needs to be edited so that the command ``\bbobdatapath`` points
    to the output folder.

    These output files will contain performance tables, performance
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
            these options can be used to output respectively the ERT
            graphs figures, run length distribution figures or the
            comparison tables scatter plot figures only. Any combination
            of these options results in no output.
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

    global ftarget
    try:

        try:
            opts, args = getopt.getopt(argv, shortoptlist, longoptlist)
        except getopt.error, msg:
             raise Usage(msg)

        if not (args):
            usage()
            sys.exit()

        isfigure = True
        isrldistr = True
        istable = True
        isscatter = True
        isscaleup = True
        isNoisy = False
        isNoiseFree = False
        verbose = False
        outputdir = 'ppdata'
        inputsettings = 'color'
        isConv= False
        isRLbased = None  # allows automatic choice
        isExpensive = None 

        #Process options
        for o, a in opts:
            if o in ("-v","--verbose"):
                verbose = True
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-o", "--output-dir"):
                outputdir = a
            #elif o in ("-s", "--style"):
            #    inputsettings = a
            elif o == "--fig-only":
                isrldistr = False
                istable = False
                isscatter = False
            elif o == "--rld-only":
                isfigure = False
                istable = False
                isscatter = False
            elif o == "--tab-only":
                isfigure = False
                isrldistr = False
                isscatter = False
            elif o == "--sca-only":
                isfigure = False
                isrldistr = False
                istable = False
            elif o == "--noisy":
                isNoisy = True
            elif o == "--noise-free":
                isNoiseFree = True
            elif o == "--settings":
                inputsettings = a
            elif o == "--conv":
                isConv = True
            elif o == "--runlength-based":
                isRLbased = True
            elif o == "--expensive":
                isExpensive = True  # comprises runlength-based
            elif o == "--not-expensive":
                isExpensive = False  
            else:
                assert False, "unhandled option"

        # from bbob_pproc import bbob2010 as inset # input settings
        if inputsettings == "color":
            from bbob_pproc import genericsettings as inset # input settings
            config.config()
        elif inputsettings == "grayscale": # probably very much obsolete
            from bbob_pproc import grayscalesettings as inset # input settings
        elif inputsettings == "black-white": # probably very much obsolete
            from bbob_pproc import bwsettings as inset # input settings
        else:
            txt = ('Settings: %s is not an appropriate ' % inputsettings
                   + 'argument for input flag "--settings".')
            raise Usage(txt)

        if (not verbose):
            warnings.simplefilter('module')
            warnings.simplefilter('ignore')            

        print ("Post-processing will generate comparison " +
               "data in folder %s" % outputdir)
        print "  this might take several minutes."

        dsList, sortedAlgs, dictAlg = processInputArgs(args, verbose=verbose)

        if 1 < 3 and len(sortedAlgs) != 2:
            raise ValueError('rungeneric2.py needs exactly two algorithms to compare, found: ' 
                             + str(sortedAlgs)
                             + '\n use rungeneric.py (or rungenericmany.py) to compare more algorithms. ')
 
        if not dsList:
            sys.exit()

        for i in dictAlg:
            if isNoisy and not isNoiseFree:
                dictAlg[i] = dictAlg[i].dictByNoise().get('nzall', DataSetList())
            if isNoiseFree and not isNoisy:
                dictAlg[i] = dictAlg[i].dictByNoise().get('noiselessall', DataSetList())

        for i in dsList:
            if i.dim not in genericsettings.dimensions_to_display:
                continue

            if (dict((j, i.instancenumbers.count(j)) for j in set(i.instancenumbers)) <
                inset.instancesOfInterest):
                warnings.warn('The data of %s do not list ' %(i) +
                              'the correct instances ' +
                              'of function F%d.' %(i.funcId))

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

        # compute maxfuneval values
        dict_max_fun_evals1 = {}
        dict_max_fun_evals2 = {}
        for ds in dsList0:
            dict_max_fun_evals1[ds.dim] = np.max((dict_max_fun_evals1.setdefault(ds.dim, 0), float(np.max(ds.maxevals))))
        for ds in dsList1:
            dict_max_fun_evals2[ds.dim] = np.max((dict_max_fun_evals2.setdefault(ds.dim, 0), float(np.max(ds.maxevals))))
        if isRLbased is not None:
            genericsettings.runlength_based_targets = isRLbased
        config.target_values(isExpensive, {1: min([max([val/dim for dim, val in dict_max_fun_evals1.iteritems()]), 
                                                   max([val/dim for dim, val in dict_max_fun_evals2.iteritems()])]
                                                  )})
        config.config()
        
        ######################### Post-processing #############################
        if isfigure or isrldistr or istable or isscatter:
            if not os.path.exists(outputdir):
                os.mkdir(outputdir)
                if verbose:
                    print 'Folder %s was created.' % (outputdir)
            
            # prepend the algorithm name command to the tex-command file
            abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            lines = []
            for i, alg in enumerate(args):
                lines.append('\\providecommand{\\algorithm' + abc[i] + '}{' + 
                        str_to_latex(strip_pathname(alg)) + '}')
            prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 
                         lines, 1000, 
                         'bbob_proc_commands.tex truncated, consider removing the file before the text run'
                         )

        # Check whether both input arguments list noisy and noise-free data
        dictFN0 = dsList0.dictByNoise()
        dictFN1 = dsList1.dictByNoise()
        k0 = set(dictFN0.keys())
        k1 = set(dictFN1.keys())
        symdiff = k1 ^ k0 # symmetric difference
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
            raise Usage('Data Mismatch: \n  ' + ' '.join(txt)
                        + '\nTry using --noise-free or --noisy flags.')

        if isfigure:
            plt.rc("axes", **inset.rcaxeslarger)
            plt.rc("xtick", **inset.rcticklarger)
            plt.rc("ytick", **inset.rcticklarger)
            plt.rc("font", **inset.rcfontlarger)
            plt.rc("legend", **inset.rclegendlarger)
            ppfig2.main(dsList0, dsList1, ppfig2_ftarget, outputdir, verbose)
            print "log ERT1/ERT0 vs target function values done."

        plt.rc("axes", **inset.rcaxes)
        plt.rc("xtick", **inset.rctick)
        plt.rc("ytick", **inset.rctick)
        plt.rc("font", **inset.rcfont)
        plt.rc("legend", **inset.rclegend)

        if isrldistr:
            if len(dictFN0) > 1 or len(dictFN1) > 1:
                warnings.warn('Data for functions from both the noisy and ' +
                              'non-noisy testbeds have been found. Their ' +
                              'results will be mixed in the "all functions" ' +
                              'ECDF figures.')
            dictDim0 = dsList0.dictByDim()
            dictDim1 = dsList1.dictByDim()

            # ECDFs of ERT ratios
            for dim in set(dictDim0.keys()) & set(dictDim1.keys()):
                if dim in inset.rldDimsOfInterest:
                    # ECDF for all functions altogether
                    try:
                        pprldistr2.main(dictDim0[dim], dictDim1[dim], dim,
                                        inset.rldValsOfInterest,
                                        outputdir, '%02dD_all' % dim, verbose)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.'
                                      % (dim))
                        continue

                    # ECDFs per function groups
                    dictFG0 = dictDim0[dim].dictByFuncGroup()
                    dictFG1 = dictDim1[dim].dictByFuncGroup()

                    for fGroup in set(dictFG0.keys()) & set(dictFG1.keys()):
                        pprldistr2.main(dictFG1[fGroup], dictFG0[fGroup], dim,
                                        inset.rldValsOfInterest,
                                        outputdir, '%02dD_%s' % (dim, fGroup),
                                        verbose)

                    # ECDFs per noise groups
                    dictFN0 = dictDim0[dim].dictByNoise()
                    dictFN1 = dictDim1[dim].dictByNoise()

                    for fGroup in set(dictFN0.keys()) & set(dictFN1.keys()):
                        pprldistr2.main(dictFN1[fGroup], dictFN0[fGroup], dim,
                                        inset.rldValsOfInterest,
                                        outputdir, '%02dD_%s' % (dim, fGroup),
                                        verbose)
                                                
            prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'),
                            ['\\providecommand{\\bbobpprldistrlegendtwo}[1]{',
                             pprldistr.caption_two(),  # depends on the config setting, should depend on maxfevals
                             '}'
                            ])
            
            
            print "ECDF runlength ratio graphs done."

            for dim in set(dictDim0.keys()) & set(dictDim1.keys()):
                pprldistr.fmax = None #Resetting the max final value
                pprldistr.evalfmax = None #Resetting the max #fevalsfactor
                # ECDFs of all functions altogether
                if dim in inset.rldDimsOfInterest:
                    try:
                        pprldistr.comp(dictDim1[dim], dictDim0[dim],
                                       inset.rldValsOfInterest, # TODO: let rldVals... possibly be RL-based targets
                                       True,
                                       outputdir, 'all', verbose)
                    except KeyError:
                        warnings.warn('Could not find some data in %d-D.'
                                      % (dim))
                        continue

                    # ECDFs per function groups
                    dictFG0 = dictDim0[dim].dictByFuncGroup()
                    dictFG1 = dictDim1[dim].dictByFuncGroup()

                    for fGroup in set(dictFG0.keys()) & set(dictFG1.keys()):
                        pprldistr.comp(dictFG1[fGroup], dictFG0[fGroup],
                                       inset.rldValsOfInterest, 
                                       True, outputdir,
                                       '%s' % fGroup, verbose)

                    # ECDFs per noise groups
                    dictFN0 = dictDim0[dim].dictByNoise()
                    dictFN1 = dictDim1[dim].dictByNoise()
                    for fGroup in set(dictFN0.keys()) & set(dictFN1.keys()):
                        pprldistr.comp(dictFN1[fGroup], dictFN0[fGroup],
                                       inset.rldValsOfInterest, 
                                       True, outputdir,
                                       '%s' % fGroup, verbose)

            print "ECDF runlength graphs done."

        if isConv:
            ppconverrorbars.main(dictAlg,outputdir,verbose)

        if istable:
            dictNG0 = dsList0.dictByNoise()
            dictNG1 = dsList1.dictByNoise()

            for nGroup in set(dictNG0.keys()) & set(dictNG1.keys()):
                # split table in as many as necessary
                dictFunc0 = dictNG0[nGroup].dictByFunc()
                dictFunc1 = dictNG1[nGroup].dictByFunc()
                funcs = list(set(dictFunc0.keys()) & set(dictFunc1.keys()))
                if len(funcs) > 24:
                    funcs.sort()
                    nbgroups = int(numpy.ceil(len(funcs)/24.))
                    def split_seq(seq, nbgroups):
                        newseq = []
                        splitsize = 1.0/nbgroups*len(seq)
                        for i in range(nbgroups):
                            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
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
                                      outputdir, '%s%d' % (nGroup, i), verbose)
                else:
                    if 11 < 3:  # future handling: 
                        dictFunc0 = dsList0.dictByFunc()
                        dictFunc1 = dsList1.dictByFunc()
                        funcs = list(set(dictFunc0.keys()) & set(dictFunc1.keys()))
                        funcs.sort()
#                        nbgroups = int(numpy.ceil(len(funcs)/testbedsettings.numberOfFunctions))
#                        pptable2.main(dsList0, dsList1,
#                                      testbedsettings.tabDimsOfInterest, outputdir,
#                                      '%s' % (testbedsettings.testbedshortname), verbose)
                    else:
                        pptable2.main(dictNG0[nGroup], dictNG1[nGroup],
                                      inset.tabDimsOfInterest, outputdir,
                                      '%s' % (nGroup), verbose)

            if isinstance(pptable2.targetsOfInterest, pproc.RunlengthBasedTargetValues):
                prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 
                            ['\\providecommand{\\bbobpptablestwolegend}[1]{', 
                             pptable2.table_caption_expensive, 
                             '}'
                            ])
            else:
                prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 
                            ['\\providecommand{\\bbobpptablestwolegend}[1]{', 
                             pptable2.table_caption, 
                             '}'
                            ])
            print "Tables done."

        if isscatter:
            if genericsettings.runlength_based_targets:
                ppscatter.targets = ppscatter.runlength_based_targets
            ppscatter.main(dsList1, dsList0, outputdir, verbose=verbose)
            prepend_to_file(os.path.join(outputdir, 'bbob_pproc_commands.tex'), 
                            ['\\providecommand{\\bbobppscatterlegend}[1]{', 
                             ppscatter.figure_caption(), 
                             '}'
                            ])
            print "Scatter plots done."

        if isscaleup:
            plt.rc("axes", labelsize=20, titlesize=24)
            plt.rc("xtick", labelsize=20)
            plt.rc("ytick", labelsize=20)
            plt.rc("font", size=20)
            plt.rc("legend", fontsize=20)
            if genericsettings.runlength_based_targets:
                ftarget = RunlengthBasedTargetValues([target_runlength])  # TODO: make this more variable but also consistent
            ppfigs.main(dictAlg, sortedAlgs, ftarget, outputdir, verbose)
            plt.rcdefaults()
            print "Scaling figures done."

        if isfigure or isrldistr or istable or isscatter or isscaleup:
            print "Output data written to folder %s" % outputdir

        plt.rcdefaults()

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "For help use -h or --help"
        return 2
    
if __name__ == "__main__":
    sys.exit(main())
