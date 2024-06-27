#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Process data to be included in a latex template. Called via

python -m cocopp [OPTIONS] DATAFOLDER1 DATAFOLDER2 ...

For a detailed help, simply type

python -m cocopp

"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import getopt
import warnings
import webbrowser
import matplotlib
from . import genericsettings, testbedsettings, rungeneric1, rungenericmany, toolsdivers, bestalg, archiving
from .toolsdivers import truncate_latex_command_file, print_done, diff_attr
from .ppfig import Usage
from .compall import ppfigs

import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')  # To avoid window popup and use without X forwarding

__all__ = ['main']

# Used by getopt:
short_options = "hvo:"
long_options = ["help", "output-dir=", "noisy", "noise-free",
               "tab-only", "fig-only", "rld-only", "no-rld-single-fcts",
               "verbose", "settings=", "conv",
               "expensive", "runlength-based",
               "los-only", "crafting-effort=", "pickle",
               "sca-only", "no-svg",
               "include-fonts"]
# thereby, "los-only", "crafting-effort=", and "pickle" affect only rungeneric1
# and "sca-only" only affects rungenericmany


def _split_short_opt_list(short_opt_list):
    """Split short options list used by getopt.

    Returns a set of the options.

    """
    res = set()
    tmp = short_opt_list[:]
    # split into logical elements: one-letter that could be followed by colon
    while tmp:
        if len(tmp) > 1 and tmp[1] == ':':
            res.add(tmp[0:2])
            tmp = tmp[2:]
        else:
            res.add(tmp[0])
            tmp = tmp[1:]

    return res


def usage():
    print(main.__doc__)


def main(argv=None):
    r"""Main routine for post-processing data from COCO.

    Synopsis::

        python -m cocopp [data_folder [more_data_folders]]

    or::

        python -c "import cocopp; cocopp.main('data_folder [more_data_folders]')"

    For this call to work, the path to this package must be in python
    search path, that is,

    * it can be in the current working directory, or
    * the path to the package was appended to the Python path, or
    * the package was installed (which essentially copies the package
      to a location which is in the path)

    ``data_folder`` may be a name from the known data archive, see e.g.
    `cocopp.bbob`, or a uniquely matching substring of such a name, or a
    matching substring with added "!" in which case the first match is taken, or
    a matching substring with added "*" in which case all matches are taken, or
    a regular expression containing a '*' before the last character, in which
    case, for example, "bbob/.*7.*cma"  matches "bbob/2017/DTS-CMA-ES-Pitra.tgz"
    (among others).

    This routine will:

    * call sub-routine :py:func:`cocopp.rungeneric1.main` for one
      input argument (see also --include-single option); the input
      argument will be used as output sub-folder relative to the main
      output folder,
    * call sub-routine :py:func:`cocopp.rungenericmany.main`
      (2 or more input arguments) for the input arguments altogether.
    * alternatively call sub-routine :py:func:`cocopp.__main__.main` if option
      flag --test is used. In this case it will run through the
      post-processing tests.

    Usecase from a Python shell
    ---------------------------
    To fine-control the behavior of the module, it is highly recommended
    to work from an (I)Python shell. For example::

        import cocopp
        cocopp.genericsettings.background = {None: cocopp.bbob.get_all("2009/")}
        cocopp.main("data_folder " + cocopp.data_archive.get("2009/BFGS_ros_noiseless"))

    compares an experiment given in `"data_folder"` with BFGS and displays
    all archived results from 2009 in the background. `cocopp.bbob` is a
    `cocopp.archiving.COCODataArchive` class.

    This may take 5-15 minutes to complete, because more than 30 algorithm
    datasets are processed.

    Output
    ------

    The output figures and tables written by default to the output folder
    :file:`ppdata` are used in the provided LaTeX templates:

    * :file:`*article.tex` and :file:`*1*.tex`
      for results with a **single** algorithm
    * :file:`*cmp.tex` and :file:`*2*.tex`
      for showing the comparison of **2** algorithms
    * :file:`*many.tex` and :file:`*3*.tex`
      for showing the comparison of **more than 2** algorithms.
    The templates with `noisy` mentioned in the filename have to be used
    for the noisy testbed, the others for the noise-less one.

    These latex templates need to be copied in the current working directory
    and possibly edited so that the LaTeX commands ``\bbobdatapath`` and
    ``\algfolder`` point to the correct output folders of the post-processing.
    Compiling the template file with LaTeX should then produce a document.

    Keyword arguments:

    *argv* -- list of strings containing options and arguments. If not
       provided, sys.argv is accessed.

    *argv* must list folders containing COCO data files. Each of these
    folders should correspond to the data of ONE algorithm.

    Furthermore, argv can begin with facultative option flags.

        -h, --help

            displays this message.

        -v, --verbose

            verbose mode, prints out operations.

        -o, --output-dir=OUTPUTDIR

            changes the default output directory (:file:`ppdata`) to
            :file:`OUTPUTDIR`.

        --include-single

            calls the postprocessing and in particular
            :py:func:`cocopp.rungeneric1.main` on each of the
            single input arguments separately.

        --no-rld-single-fcts

            do not generate runlength distribution figures for each
            single function.

        --input-path=INPUTPATH

            all folder/file arguments are prepended with the given value
            which must be a valid path.

        --in-a-hurry

            takes values between 0 (default) and 1000, fast processing that
            does not write eps files and uses a small number of bootstrap samples

        --no-svg

            do not generate the svg figures which are used in html files

        --conv

            prepares also convergence plots with median function values over time

        --include-fonts

            generated pdfs will have the fonts included (important for ACM style
            LaTeX submissions)


    Exceptions raised:

    *Usage* -- Gives back a usage message.

    Examples:

    Printing out this help message::

        $ python -m cocopp.rungeneric -h

    Post-processing two algorithms in verbose mode::

        $ python -m cocopp -v AMALGAM BIPOP-CMA-ES

    From the python interpreter::

        >> import cocopp
        >> cocopp.main('-o outputfolder folder1 folder2')

    This will execute the post-processing on the data found in
    :file:`folder1` and :file:`folder2` and return the respective
    `DataSetList`. The ``-o`` option changes the output folder from the
    default :file:`ppdata` to :file:`outputfolder`. The arguments can
    also be presented as a list of strings.

    """
    # global shortoptlist
    # global longoptlist

    if argv is None:
        argv = sys.argv[1:]
    if not isinstance(argv, list) and str(argv) == argv:  # get rid of .split in python shell
        argv = argv.split()
    try:
        try:
            opts, args = getopt.getopt(argv, short_options,
                                       long_options +
                                       ['include-single', 'in-a-hurry=', 'input-path='])
        except getopt.error as msg:
            raise Usage(msg)

        if not args:
            usage()
            sys.exit()

        testbedsettings.reset_current_testbed()
        testbedsettings.reset_reference_values()
        bestalg.reset_reference_algorithm()

        inputdir = '.'

        # Process options
        shortoptlist = list("-" + i.rstrip(":")
                            for i in _split_short_opt_list(short_options))
        if "-o" in shortoptlist:
            shortoptlist.remove("-o") # 2020/6/5: TODO: not sure why this is done
        longoptlist = list("--" + i.rstrip("=") for i in long_options)

        plt.rc("axes", **genericsettings.rcaxes)
        plt.rc("xtick", **genericsettings.rctick)
        plt.rc("ytick", **genericsettings.rctick)
        plt.rc("font", **genericsettings.rcfont)
        plt.rc("legend", **genericsettings.rclegend)

        genopts = []
        outputdir = genericsettings.outputdir
        for o, a in opts:
            if o in ("-h", "--help"):
                usage()
                sys.exit()
            elif o in ("-o", "--output-dir"):
                outputdir = a.strip()  # like this ["-o folder"] + ... works as input
            elif o in ("--in-a-hurry", ):
                genericsettings.in_a_hurry = int(a)
                if genericsettings.in_a_hurry:
                    print('in_a_hurry like ', genericsettings.in_a_hurry, ' (should finally be set to zero)')
            elif o in ("--input-path", ):
                inputdir = a
            elif o in "--no-svg":
                genericsettings.generate_svg_files = False
            elif o == "--no-rld-single-fcts":
                genericsettings.isRldOnSingleFcts = False
            elif o == "--conv":
                genericsettings.isConv = True
            elif o == "--noisy":
                genericsettings.isNoisy = True
                warnings.warn('The usage of --noisy is deprecated and will be removed in a later release of COCO.')
            elif o == "--noise-free":
                genericsettings.isNoiseFree = True
                warnings.warn('The usage of --noise-free is deprecated and will be removed in a later release of COCO.')
            elif o in ("-p", "--pickle"):
                genericsettings.isPickled = True
                warnings.warn('The usage of --pickle is deprecated and will be removed in a later release of COCO.')
            # The next 4 are for testing purpose
            elif o in ("--runlength-based", "--budget-based"):
                genericsettings.runlength_based_targets = True
            elif o == "--expensive":
                genericsettings.isExpensive = True  # comprises runlength-based
            elif o == "--crafting-effort":
                try:
                    genericsettings.inputCrE = float(a)
                except ValueError:
                    raise Usage('Expect a valid float for flag crafting-effort.')
            elif o == "--include-fonts":
                plt.rc('pdf', fonttype=42)
                plt.rcParams['pdf.fonttype'] = 42
                matplotlib.rcParams['pdf.fonttype'] = 42
                matplotlib.rcParams['ps.fonttype'] = 42
            elif o == "--tab-only":
                genericsettings.isFig = False
                genericsettings.isRLDistr = False
                genericsettings.isLogLoss = False
            elif o == "--fig-only":
                genericsettings.isTab = False
                genericsettings.isRLDistr = False
                genericsettings.isLogLoss = False
                genericsettings.isScatter = False
            elif o == "--rld-only":
                genericsettings.isTab = False
                genericsettings.isFig = False
                genericsettings.isLogLoss = False
                genericsettings.isScatter = False
            elif o == "--los-only":
                genericsettings.isTab = False
                genericsettings.isFig = False
                genericsettings.isRLDistr = False
            elif o == "--cons-only":
                genericsettings.isTab = False
                genericsettings.isFig = False
                genericsettings.isRLDistr = False
                genericsettings.isLogLoss = False
            else:
                is_assigned = False
                if o in longoptlist or o in shortoptlist:
                    genopts.append(o)
                    # Append o and then a separately otherwise the list of
                    # command line arguments might be incorrect
                    if a:
                        genopts.append(a)
                    if o == '--settings' and a == 'grayscale':  # a hack for test cases
                        genericsettings.interactive_mode = False
                    is_assigned = True
                if o in ("-v", "--verbose"):
                    genericsettings.verbose = True
                    is_assigned = True
                if o == '--include-single':
                    is_assigned = True
                if not is_assigned:
                    assert False, "unhandled option"
        if not genericsettings.verbose:
            warnings.filterwarnings('module', '.*', UserWarning, '.*')
            # warnings.simplefilter('ignore')  # that is bad, but otherwise to many warnings appear

#        print("\nPost-processing: will generate output " +
#               "data in folder %s" % outputdir)
#        print("  this might take several minutes.")

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            if genericsettings.verbose:
                print('Folder %s was created.' % outputdir)

        latex_commands_filename = os.path.join(outputdir, 'cocopp_commands.tex')

        truncate_latex_command_file(latex_commands_filename)

        print('Post-processing (%s)' % ('1' if len(args) == 1 else '2+'))  # to not break doctests

        # manage data paths as given in args
        data_archive = archiving.official_archives.all  # was: archiving.COCODataArchive()
        args = data_archive.get_extended(args)
        if None in args:
            raise ValueError("Data argument %d was not matching any file"
                             " or archive entry." % (args.index(None) + 1))
        if len(args) != len(set(args)):
            warnings.warn("Several data arguments point to the very same location."
                          "This will most likely lead to a rather unexpected outcome.")
            # TODO: we would like the users input with timeout to confirm
            # and otherwise raise a ValueError

        update_background_algorithms(inputdir)

        print('  Using %d data set%s:' % (len(args), 's' if len(args) > 1 else ''))
        for path in args:
            print('    %s' % path)

        # we still need to check that all data come from the same
        # test suite, at least for the data_archive data
        suites = set()
        for path in args:
            if data_archive.contains(path):  # this is the archive of *all* testbeds
                # extract suite name
                suites.add(data_archive._name_with_check(path).split('/')[0])
        if len(suites) > 2:
            raise ValueError("Data from more than two suites %s cannot "
                             "be post-processed together" % str(suites))

        if len(args) == 1 or '--include-single' in dict(opts):
            genericsettings.foreground_algorithm_list = []
            for i, alg in enumerate(args):
                genericsettings.foreground_algorithm_list.append(alg)
                dsld = rungeneric1.main(alg, outputdir, genopts + ["-o", outputdir, alg])

        if len(args) >= 2 or len(genericsettings.background) > 0:
            # Reset foreground algorithm list if cocopp.main() is called.
            # Otherwise the list accumulates arguments passed to cocopp.main().
            # Arguments are still accumulated if rungeneric.main() is bypassed
            # and rungenericmany.main() or lower-level functions are called.
            genericsettings.foreground_algorithm_list = []
            dsld = rungenericmany.main(args, outputdir)
            
        toolsdivers.prepend_to_file(latex_commands_filename,
                                        ['\\providecommand{\\numofalgs}{%d}' % len(args)]
                                        )
        toolsdivers.prepend_to_file(latex_commands_filename,
                                    ['\\providecommand{\\cocoversion}{{\\scriptsize\\sffamily{}' +
                                     '\\color{Gray}Data produced with COCO %s}}' % (toolsdivers.get_version_label(None))]
                                    )
        toolsdivers.prepend_to_file(latex_commands_filename,
                                    ['\\providecommand{\\bbobecdfcaptionsinglefunctionssingledim}[1]{',
                                     ppfigs.get_ecdfs_single_functions_single_dim_caption(), '}']
                                    )
            
        open(os.path.join(outputdir,
                          'cocopp_commands.tex'), 'a').close()


        # print changed genericsettings attributes
        def as_str(s, clip=25):
            """return ``str(s)``, only surround by '"' if `s` is a string
            """
            put_quotes = True if s is str(s) else False
            s = str(s)
            if len(s) > clip:
                s = s[:clip-3] + '...'
            return '"%s"' % s if put_quotes else s
        mess = ''
        for key, v1, v2 in diff_attr(genericsettings.default_settings,
                                     genericsettings):
            mess = mess + '    %s: from %s to %s\n' % (
                key, as_str(v1), as_str(v2))
        if mess:
            print('Setting changes in `cocopp.genericsettings` compared to default:')
            print(mess, end='')

        plt.rcdefaults()

        print_done('ALL done')
        if genericsettings.interactive_mode:
            try:
                webbrowser.open("file://" + os.getcwd() + '/' + outputdir + "/index.html")
            except:
                pass
        return dsld

    # TODO prevent loading the data every time...
        
    except Usage as err:
        print(err.msg, file=sys.stderr)
        print("For help use -h or --help", file=sys.stderr)
        return 2


def update_background_algorithms(input_dir):
    for format, names in genericsettings.background.items():
        if not isinstance(names, (tuple, list, set)):
            raise ValueError(
                "`genericsettings.background` has the wrongly formatted entry\n"
                "%s\n"
                "Expected is ``(format, names)``, where"
                " names is a `list` of one or more pathnames (not a"
                " single pathname as `str`)"
                % str((format, names)))
        genericsettings.background[format] = [os.path.join(input_dir, filename) for filename in names]
