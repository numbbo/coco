#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COmparing Continuous Optimisers (COCO) post-processing package

This package (`cocopp`) generates output figures and tables in html format
and for including into LaTeX-documents.

The `cocopp.Interface` class contains the most basic commands and data of
the package, sufficient for most use cases.

>>> import cocopp
>>> sorted(cocopp.Interface.dir())
['archives', 'config', 'genericsettings', 'load', 'main']
>>> all(hasattr(cocopp, name) for name in cocopp.Interface.dir())
True

The main method of the `cocopp` package is `main` (currently aliased to
`cocopp.rungeneric.main`). The `main` method also allows basic use of the
post-processing through a command-line interface. The recommended use
is however from an IPython/Jupyter shell:

>>> import cocopp
>>> cocopp.main('exdata/my_output another_folder yet_another_or_not')  # doctest:+SKIP

postprocesses data from one or several folders, for example data
generated with the help from the `cocoex` module. Each folder should
contain data of a full experiment with a single algorithm. (Within the
folder the data can be distributed over subfolders).

Results can be explored from the ``ppdata/index.html`` file, unless a
a different output folder is specified with the ``-o`` option.

**Comparative data** from over 200 full experiments are archived online and
can be listed, filtered, and retrieved from `cocopp.archives` (of type
`OfficialArchives`) and processed alone or together with local data.

For example

>>> cocopp.archives.bbob('bfgs')  # doctest:+ELLIPSIS,+SKIP,
['2009/BFGS_...

lists all data sets containing ``'bfgs'`` in their name. The search can
also use regular expressions where '.' matches any single
character and '*' means one or more repetitions:

>>> cocopp.archives.bbob('.*bfgs')  # doctest:+ELLIPSIS,+SKIP,
['2009/BFGS_...

gives the same data sets as above and

>>> cocopp.archives.all('bbob/.*bfgs')  # doctest:+ELLIPSIS,+SKIP,
['bbob/2009/BFGS_...

gives also the same data sets, however extracted from the archive of all
suites, which is the search domain when using `cocopp.main`.

When calling the `cocopp.main` routine, a single trailing '!' or '*'
have the special meaning of take-the-first-only and take-all, respectively.
Hence, the first entry of the above selecting list can be postprocessed with

>>> cocopp.main('bfgs!')  # doctest:+SKIP

All `'bfgs'` matches from the `'bbob'` suite can be processed like

>>> cocopp.main('bbob/.*bfgs')  # doctest:+SKIP

(``cocopp.main('bfgs*')`` raises an error as data from incompatible suites
cannot be processed together.)

The postprocessing result of

>>> cocopp.main('bbob/2009/*')  # doctest:+SKIP

can be browsed at http://numbbo.github.io/ppdata-archive/bbob/2009 (or 2009-all).

To display algorithms in the background, the ``genericsettings.background``
variable needs to be set:

>>> cocopp.genericsettings.background = {None: cocopp.archives.bbob.get_all('bfgs')}  # doctest:+SKIP

where `None` invokes the default color (gray) and line style (solid)
``genericsettings.background_default_style``.

Now we could compare our own data with the first ``'bfgs'``-matching
archived algorithm where all other archived BFGS data are shown in the
background with

>>> cocopp.main('exdata/my_output bfgs!')  # doctest:+SKIP

"""

from __future__ import absolute_import
import sys as _sys

import matplotlib  # just to make sure the following is actually done first
matplotlib.use('Agg')  # To avoid window popup and use without X forwarding
del matplotlib

from numpy.random import seed as set_seed

from .cococommands import *  # outdated
from . import config
from . import archiving
from . import rungeneric
from . import genericsettings

from .rungeneric import main

from ._version import __version__

__all__ = []

archives = archiving.official_archives  # just an alias
if archives is not None:
    data_archive = archives.all  # another alias, only for historical reasons
    archives.link_as_attributes_in(_sys.modules['cocopp'],  # more individual aliases
                                   except_for=['all', 'test'])

# data_archive = 'use `archives.all` instead'
# bbob = 'use `archives.bbob` instead'
# bbob_noisy = 'use `archives.bbob_noisy` instead'
# bbob_biobj = 'use `archives.bbob_biobj` instead'

class Interface:
    """collection of the most user-relevant modules, methods and data.

    `archives`: online data archives of type `OfficialArchives`

    `archiving`: methods to archive data and retrieve archived data put online

    `config`: dynamic configuration tool (advanced)

    `genericsettings`: basic settings

    `load`: loading data from disk

    `main`: post-processing data from disk
    """
    @classmethod
    def dir(cls):
        """return `dict` of non-private class attribute names->values"""
        return dict(it for it in cls.__dict__.items()
                    if not it[0].startswith('_') and not it[0] == 'dir')
    archives = archives
    config = config
    genericsettings = genericsettings
    load = load
    main = main

# clean up namespace
del absolute_import
# del bestalg, captions, comp2, compall, htmldesc, pickle, ppconverrorbars
# del ppfig, ppfigdim, ppfigparam, pplogloss, pprldistr, pproc, pptable
# del pptex, readalign, rungeneric1, rungenericmany, toolsdivers, toolsstats

# cococommands, config, data_archive, dataformatsettings, findfiles,
# genericsettings, info, load, main, rungeneric, set_seed, systeminfo, testbedsettings,
