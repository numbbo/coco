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
import imp  # import default genericsettings
import warnings
import matplotlib
from . import genericsettings, rungeneric1, rungeneric2, rungenericmany, ppfig, toolsdivers #, __main__
from .toolsdivers import truncate_latex_command_file, print_done
from .ppfig import Usage
from .compall import ppfigs
from . import __path__  # import path for genericsettings

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

    This routine will:

    * call sub-routine :py:func:`cocopp.rungeneric1.main` for each
      input argument; each input argument will be used as output
      sub-folder relative to the main output folder,
    * call either sub-routines :py:func:`cocopp.rungeneric2.main`
      (2 input arguments) or :py:func:`cocopp.rungenericmany.main`
      (more than 2) for the input arguments altogether.
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

        >> import cocopp as pp
        >> pp.main('-o outputfolder folder1 folder2')

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

    stored_settings = imp.load_module('_genericsettings',
                                      *imp.find_module('genericsettings',
                                                       __path__))
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
                outputdir = a
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

        for i in range(len(args)):  # prepend common path inputdir to all names
            args[i] = os.path.join(inputdir, args[i])
        update_background_algorithms(inputdir)

        if len(args) == 1 or '--include-single' in dict(opts):
            for i, alg in enumerate(args):
                dsld = rungeneric1.main(genopts + ["-o", outputdir, alg])

        if len(args) >= 2 or len(genericsettings.background) > 0:
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

        ppfig.save_index_html_file(os.path.join(outputdir, genericsettings.index_html_file_name))

        # print changed genericsettings attributes
        mess = ''
        def as_str(s):
            return '"%s"' % s if s is str(s) else str(s)
        for key in stored_settings.__dict__:
            if key.startswith('__'):
                continue
            v1, v2 = getattr(stored_settings, key), getattr(genericsettings, key)
            if v1 != v2 and not str(v1).startswith('<function '):
                mess = mess + '    %s: from %s to %s\n' % (
                    key, as_str(v1), as_str(v2))
        if mess:
            print('Changed settings in `genericsettings` (compared to default):')
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
        genericsettings.background[key] = [os.path.join(input_dir, item) for item in value]
