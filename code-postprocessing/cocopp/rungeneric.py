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

import os
import sys
import getopt
import warnings
import matplotlib
from . import genericsettings, testbedsettings, rungeneric1, rungenericmany, toolsdivers, bestalg, findfiles
from .toolsdivers import truncate_latex_command_file, print_done, diff_attr
from .ppfig import Usage
from .compall import ppfigs

matplotlib.use('Agg')  # To avoid window popup and use without X forwarding

__all__ = ['main']


def _split_short_opt_list(short_opt_list):
    """Split short options list used by getopt.

    Returns a set of the options.

    """
    res = set()
    tmp = short_opt_list[:]
    # split into logical elements: one-letter that could be followed by colon
    while tmp:
        if len(tmp) > 1 and tmp[1] is ':':
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
    `cocopp.bbob`, or a uniquely matching substring of such a name,
    or a matching substring with added "!" in which case the first
    match is taken, or a matching substring with added "*" in which
    case all matches are taken.

    This routine will:

    * call sub-routine :py:func:`cocopp.rungeneric1.main` for each
      input argument; each input argument will be used as output
      sub-folder relative to the main output folder,
    * call sub-routine :py:func:`cocopp.rungenericmany.main`
      (2 or more input arguments) for the input arguments altogether.
    * alternatively call sub-routine :py:func:`cocopp.__main__.main` if option
      flag --test is used. In this case it will run through the
      post-processing tests.

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

    if argv is None:
        argv = sys.argv[1:]
    if not isinstance(argv, list) and str(argv) == argv:  # get rid of .split in python shell
        argv = argv.split()
    try:
        try:
            opts, args = getopt.getopt(argv, genericsettings.shortoptlist,
                                       genericsettings.longoptlist +
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
                            for i in _split_short_opt_list(genericsettings.shortoptlist))
        shortoptlist.remove("-o")
        longoptlist = list("--" + i.rstrip("=") for i in genericsettings.longoptlist)

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
            else:
                is_assigned = False
                if o in longoptlist or o in shortoptlist:
                    genopts.append(o)
                    # Append o and then a separately otherwise the list of
                    # command line arguments might be incorrect
                    if a:
                        genopts.append(a)
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
        data_archive = findfiles.COCODataArchive()
        clean_extended_args = []
        for i, name in enumerate(args):
            # prepend common path inputdir to path names
            path = os.path.join(inputdir, args[i].replace('/', os.sep))
            if os.path.exists(path):
                clean_extended_args.append(path)
            elif name.endswith('!'):  # take first match
                data_archive.find(name[:-1])
                clean_extended_args.append(data_archive.get())
            elif name.endswith('*'):  # take all matches
                clean_extended_args.extend(data_archive.get_all(name[:-1]))  # download if necessary
            elif data_archive.find(name):  # get will bail out if there is not exactly one match
                clean_extended_args.append(data_archive.get(name))  # download if necessary
            else:
                warnings.warn('"%s" seems not to be an existing file or match any archived data' % name)
                # TODO: with option --include-single we may have to wait some time until this leads to
                # an error. Hence we should raise the error here?
        if len(args) != len(set(args)):
            warnings.warn("Several data arguments point to the very same location."
                          "This will most likely lead to a rather unexpected outcome.")
            # TODO: we would like the users input with timeout to confirm
            # and otherwise raise a ValueError

        args = clean_extended_args

        update_background_algorithms(inputdir)

        print('  Using:')
        for path in args:
            print('    %s' % path)

        # we still need to check that all data come from the same
        # test suite, at least for the data_archive data
        suites = set()
        for path in clean_extended_args:
            if data_archive.contains(path):  # this is the archive of *all* testbeds
                # extract suite name
                suites.add(data_archive.name(path).split('/')[0])
        if len(suites) > 1:
            raise ValueError("Data from more than one suites %s cannot "
                             "be post-processed together" % str(suites))

        if len(args) == 1 or '--include-single' in dict(opts):
            for i, alg in enumerate(args):
                dsld = rungeneric1.main(genopts + ["-o", outputdir, alg])

        if len(args) >= 2 or len(genericsettings.background) > 0:
            # Reset foreground algorithm list if cocopp.main() is called.
            # Otherwise the list accumulates arguments passed to cocopp.main().
            # Arguments are still accumulated if rungeneric.main() is bypassed
            # and rungenericmany.main() or lower-level functions are called.
            genericsettings.foreground_algorithm_list = []
            dsld = rungenericmany.main(genopts + ["-o", outputdir] + args)
            toolsdivers.prepend_to_file(latex_commands_filename,
                                        ['\\providecommand{\\numofalgs}{2+}']
                                        )

        toolsdivers.prepend_to_file(latex_commands_filename,
                                    ['\\providecommand{\\cocoversion}{\\hspace{\\textwidth}\\scriptsize\\sffamily{}' +
                                     '\\color{Gray}Data produced with COCO %s}' % (toolsdivers.get_version_label(None))]
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

        print_done('ALL done')

        return dsld

    # TODO prevent loading the data every time...
        
    except Usage as err:
        print(err.msg, file=sys.stderr)
        print("For help use -h or --help", file=sys.stderr)
        return 2


def update_background_algorithms(input_dir):
    for key, value in genericsettings.background.items():
        # why can't we use different variable names than value and item, please?
        genericsettings.background[key] = [os.path.join(input_dir, item) for item in value]
