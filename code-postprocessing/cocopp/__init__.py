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

Comparative data from over 200 full experiments are archived online and
can be listed, filtered, and retrieved from the `COCODataArchive` instances
in `cocopp.archives` (of type `KnownArchives`) and processed alone or
together with local data.

For example

>>> cocopp.archives.bbob('bfgs')  # doctest:+ELLIPSIS
['2009/BFGS_...

lists all data sets containing ``'bfgs'`` in their name. The first in
the list can be postprocessed by

>>> cocopp.main('bfgs!')  # doctest:+SKIP

All of them can be processed like

>>> cocopp.main('bfgs*')  # doctest:+SKIP

Only a trailing `*` is accepted and any string containing the
substring is matched. The postprocessing result of

>>> cocopp.main('bbob/2009/*')  # doctest:+SKIP

can be investigated at http://coco.gforge.inria.fr/ppdata-archive/bbob/2009-all.

To display algorithms in the background, the ``genericsettings.background``
variable needs to be set:

>>> cocopp.genericsettings.background = {None: cocopp.archives.bbob.get_all('bfgs')}  # doctest:+SKIP

where `None` invokes the default color (grey) and line style (solid)
``genericsettings.background_default_style``.

Now we could compare our own data with the first ``'bfgs'``-matching
archived algorithm where all other archived BFGS data are shown in the
background with

>>> cocopp.main('exdata/my_output bfgs!')  # doctest:+SKIP

"""

from __future__ import absolute_import

import matplotlib  # just to make sure the following is actually done first
matplotlib.use('Agg')  # To avoid window popup and use without X forwarding
del matplotlib

from numpy.random import seed as set_seed

from .cococommands import *  # outdated
from . import config
from . import findfiles
from . import rungeneric
from . import genericsettings

from .rungeneric import main

import pkg_resources

__all__ = [# 'main',  # import nothing with "from cocopp import *"
           ]

__version__ = pkg_resources.require('cocopp')[0].version

data_archive = findfiles.COCODataArchive()
_data_archive = data_archive  # should go away but some tests rely on this

archives = findfiles.KnownArchives()
bbob = findfiles.COCOBBOBDataArchive()  # should go away
# bbob = 'use `archives.bbob` instead'
bbob_noisy = findfiles.COCOBBOBNoisyDataArchive()
bbob_biobj = findfiles.COCOBBOBBiobjDataArchive()

class Interface:
    """collection of the most user-relevant methods and data.

    `archives`: online data archives of type `KnownArchives`

    `config`: dynamic configuration tool (advanced)

    `genericsettings`: basic settings

    `load`: loading data from disk

    `main`: post-processing data from disk
    """
    @classmethod
    def dir(cls):
        return dict(it for it in cls.__dict__.items()
                    if not it[0].startswith('_') and not it[0] == 'dir')
    archives = archives
    config = config
    genericsettings = genericsettings
    load = load
    main = main

# clean up namespace
del absolute_import, pkg_resources
# del bestalg, captions, comp2, compall, htmldesc, pickle, ppconverrorbars
# del ppfig, ppfigdim, ppfigparam, pplogloss, pprldistr, pproc, pptable
# del pptex, readalign, rungeneric1, rungenericmany, toolsdivers, toolsstats

# cococommands, config, data_archive, dataformatsettings, findfiles,
# genericsettings, info, load, main, rungeneric, set_seed, systeminfo, testbedsettings,
