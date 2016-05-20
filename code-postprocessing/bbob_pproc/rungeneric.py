#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Process data to be included in a latex template.

Synopsis:
    ``python path_to_folder/bbob_pproc/rungeneric.py [OPTIONS] FOLDERS``

Help:
    ``python path_to_folder/bbob_pproc/rungeneric.py -h``

"""

from __future__ import absolute_import

import os
import sys
import getopt
import warnings
import matplotlib
matplotlib.use('Agg')  # To avoid window popup and use without X forwarding

# numpy.seterr(all='raise')
if __name__ == "__main__":
    if 11 < 3:
        print(matplotlib.rcsetup.all_backends)
        # [u'GTK', u'GTKAgg', u'GTKCairo', u'MacOSX', u'Qt4Agg', u'Qt5Agg',
        #  u'TkAgg', u'WX', u'WXAgg', u'CocoaAgg', u'GTK3Cairo', u'GTK3Agg',
        #  u'WebAgg', u'nbAgg', u'agg', u'cairo', u'emf', u'gdk', u'pdf',
        #  u'pgf', u'ps', u'svg', u'template']
        matplotlib.use('Agg')  # To avoid window popup and use without X forwarding
        matplotlib.rc('pdf', fonttype = 42)
        # add ".." to the Python search path, import the module to which
        # this script belongs to and call the main of this script from imported
        # module. Like this all relative imports will work smoothly.
        (filepath, filename) = os.path.split(sys.argv[0])
        sys.path.append(os.path.join(filepath, os.path.pardir))
        try:
            import bbob_pproc as cocopp
        except ImportError:
            import cocopp
        res = cocopp.rungeneric.main(sys.argv[1:])
        sys.exit(res)

from . import genericsettings, testbedsettings, rungeneric1, rungeneric2, rungenericmany, ppfig
from .toolsdivers import truncate_latex_command_file, print_done
from .ppfig import Usage

__all__ = ['main']

def _splitshortoptlist(shortoptlist):
    """Split short options list used by getopt.

    Returns a set of the options.

    """
    res = set()
    tmp = shortoptlist[:]
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
    print main.__doc__

def main(argv=None):
    r"""Main routine for post-processing data from COCO.

    Synopsis::

        python -m bbob_pproc [data_folder [more_data_folders]]

    For this call to work, the path to this package must be in python
    search path, that is,

    * it can be in the current working directory, or
    * the path to the package was appended to the Python path, or
    * the package was installed (which essentially copies the package
      to a location which is in the path)

    This routine will:

    * call sub-routine :py:func:`bbob_pproc.rungeneric1.main` for each
      input argument; each input argument will be used as output
      sub-folder relative to the main output folder,
    * call either sub-routines :py:func:`bbob_pproc.rungeneric2.main`
      (2 input arguments) or :py:func:`bbob_pproc.rungenericmany.main`
      (more than 2) for the input arguments altogether.

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

        --omit-single

            omit calling :py:func:`bbob_pproc.rungeneric1.main`, if
            more than one data path argument is provided.

        --no-rld-single-fcts

            do not generate runlength distribution figures for each
            single function. These figures are not (yet) used
            in the LaTeX templates.

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

        $ python -m bbob_pproc.rungeneric -h

    Post-processing two algorithms in verbose mode::

        $ python -m bbob_pproc -v AMALGAM BIPOP-CMA-ES

    From the python interpreter::

        >> import bbob_pproc as pp
        >> pp.main('-o outputfolder folder1 folder2')

      This will execute the post-processing on the data found in
      :file:`folder1` and :file:`folder2`. The ``-o`` option changes the
      output folder from the default :file:`ppdata` to
      :file:`outputfolder`. The arguments can also be presented as
      a list of strings.

    """

    if argv is None:
        argv = sys.argv[1:]
    if not isinstance(argv, list) and str(argv) == argv:  # get rid of .split in python shell
        argv = argv.split()
    try:
        try:
            opts, args = getopt.getopt(argv, genericsettings.shortoptlist,
                                       genericsettings.longoptlist +
                                       ['omit-single', 'in-a-hurry=', 'input-path='])
        except getopt.error, msg:
            raise Usage(msg)

        if not args:
            usage()
            sys.exit()

        inputdir = '.'

        #Process options
        shortoptlist = list("-" + i.rstrip(":")
                            for i in _splitshortoptlist(genericsettings.shortoptlist))
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
                    print 'in_a_hurry like', genericsettings.in_a_hurry, '(should finally be set to zero)'
            elif o in ("--input-path", ):
                inputdir = a
            elif o in ("--no-svg"):
                genericsettings.generate_svg_files = False
            elif o in ("--constrained"):
                testbedsettings.default_testbed_single = 'CONSBBOBTestbed'
            else:
                isAssigned = False
                if o in longoptlist or o in shortoptlist:
                    genopts.append(o)
                    # Append o and then a separately otherwise the list of
                    # command line arguments might be incorrect
                    if a:
                        genopts.append(a)
                    isAssigned = True
                if o in ("-v", "--verbose"):
                    genericsettings.verbose = True
                    isAssigned = True
                if o == '--omit-single':
                    isAssigned = True
                if not isAssigned:
                    assert False, "unhandled option"


        if (not genericsettings.verbose):
            warnings.filterwarnings('module', '.*', UserWarning, '.*')
            #warnings.simplefilter('ignore')  # that is bad, but otherwise to many warnings appear

        print ("Post-processing: will generate output " +
               "data in folder %s" % outputdir)
        print "  this might take several minutes."

        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
            if genericsettings.verbose:
                print 'Folder %s was created.' % (outputdir)

        truncate_latex_command_file(os.path.join(outputdir,
                                                 'bbob_pproc_commands.tex'))

        for i in range(len(args)):  # prepend common path inputdir to all names
            args[i] = os.path.join(inputdir, args[i])

        for i, alg in enumerate(args):
            # remove '../' from algorithm output folder
            if len(args) == 1 or '--omit-single' not in dict(opts):
                rungeneric1.main(genopts
                                 + ["-o", outputdir, alg])

        if len(args) == 2:
            rungeneric2.main(genopts + ["-o", outputdir] + args)
        elif len(args) > 2:
            rungenericmany.main(genopts + ["-o", outputdir] + args)

        open(os.path.join(outputdir,
                          'bbob_pproc_commands.tex'), 'a').close() 

        ppfig.save_index_html_file(os.path.join(outputdir, genericsettings.index_html_file_name))
        # ppdata file is now deprecated.
        ppfig.save_index_html_file(os.path.join(outputdir, 'ppdata'))
        print_done()

    #TODO prevent loading the data every time...

    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use -h or --help"
        return 2

if __name__ == "__main__":
    res = main()
    if genericsettings.test:
        print res
    # sys.exit(res)
