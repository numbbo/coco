# -*- coding: utf-8 -*-
"""Online and offline archiving of COCO data.

`create` and `get` are the main functions to create and retrieve online and
local offline archives. Local archives can be listed via `ArchivesLocal`
(experimental/beta), already used online archives are listed in `ArchivesKnown`.

An online archive class defines, and is defined by, a source URL containing
an archive definition file and the archived data.

``get('all')`` returns all "officially" archived data as given in a folder
hierarchy (this may be abondoned in future). Derived classes "point" to
subfolders in the folder tree and "contain" all archived data from a single
test suites. For example, ``get('bbob')`` returns the archived data list
for the `bbob` testbed. 

How to Create an Online Archive
-------------------------------

First, we prepare the datasets. A dataset is a (tar-)zipped file containing
a full experiment from a single algorithm. The first ten-or-so characters
of the filename should be readible and informative. Datasets can reside in
an arbitrary subfolder structure, but the folders should contain no further
(ambiguous) files in order to create an archive from the archive root folder.

Second, we create the new archive with `create`,

>>> import cocopp
>>> from cocopp import archiving
>>> local_path = './my-archive-root-folder'
>>> archiving.create(local_path)  # doctest:+SKIP

thereby creating an archive definition file in the given folder. The
created archive can be re-instantiated with `cocopp.archiving.get` and
all data can be processed with `cocopp.main`, like

>>> my_archive = archiving.get(local_path)  # doctest:+SKIP
>>> cocopp.main(my_archive.get_all(''))  # doctest:+SKIP

We may want to check beforehand the archive size like

>>> len(my_archive)  # doctest:+SKIP

as an archive may contain hundreds of data sets. In case, we can choose a
subset to process (see help of `main` and/or of the archive instance).

Third, we put a mirror of the archive online, like::

    rsync -zauv my-archives/unique-name/ http://my-coco-online-archives/a-name

Now, everyone can use the archive on the fly like

>>> remote_def = 'http://my-coco-online-archives/a-name'
>>> remote_archive = cocopp.archiving.get(remote_def)  # doctest:+SKIP

just as a local archive. Archive data are downloaded only on demand.
All data can be made available offline (which might take long) with:

>>> remote_archive.get_all('')  # doctest:+SKIP

Remote archives that have been used once can be listed via `ArchivesKnown`
(experimental/beta).

Details: a definition file contains a list of all contained datasets by
path/filename, a sha256 hash and optionally their approximate size. Datasets
are (tar-)zipped files containing a full experiment from a single algorithm.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
del absolute_import, division, print_function, unicode_literals

__author__ = 'Nikolaus Hansen'

import os
import shutil as _shutil
import time as _time
import warnings
import hashlib
import ast
from . import toolsdivers as _td  # StrList
try:
    from urllib.request import urlretrieve as _urlretrieve
except ImportError:
    from urllib import urlretrieve as _urlretrieve

coco_urls = ["https://coco.gforge.inria.fr/data-archive",  # original location
             "https://numbbo.github.io/gforge/data-archive",  # new backup location
             "https://numbbo.github.io/data-archive/data-archive",  # new location
            ]
coco_url = coco_urls[-1]  # may be reassigned if it doesn't work out

# cocopp needs a directory where it can cache downloaded datasets.
# 
# We use `platformdirs` to find the users cache directory in a platform independent way
# and create a subdirectory within for cocopp.

import platformdirs
cocopp_home = platformdirs.user_cache_dir("cocopp", ensure_exists=True)
default_archive_location = os.path.join(cocopp_home, 'data-archives')
default_definition_filename = 'coco_archive_definition.txt'
cocopp_home_archives = default_archive_location
listing_file_start = 'list_'
listing_file_extension = '.txt'
backup_last_filename = ''  # global variable to see whether and where a backup was made

def _abs_path(path, *args):
    """return a (OS-dependent) user-expanded path.

    `os.path.abspath` takes care of using the right `os.path.sep`.
    """
    return os.path.abspath(os.path.join(os.path.expanduser(path), *args))

def _makedirs(path, error_ok=True):
    try:
        os.makedirs(path)
    except os.error:  # python 2&3 compatible
        if not error_ok:
            raise

def _make_backup(fullname):
    """backup file with added time stamp if it exists, otherwise do nothing."""
    global backup_last_filename
    fullname = _abs_path(fullname)
    if os.path.exists(fullname):
        p, n = os.path.split(fullname)
        dst = os.path.join(p, '._' + n + _time.strftime("_%Y-%m-%dd%Hh%Mm%Ss")
                              + str(_time.time()).split('.')[1])
        _shutil.copy2(fullname, dst)
        backup_last_filename = dst

def _url_to_folder_name(url):
    """return a path within the default archive location"""
    # if not _is_url(url):
    #     warnings.warn('"%s" seems not to be an URL' % url)
    name = url.strip().strip('/').lstrip('http://').lstrip('https://'
               ).lstrip('HTTP://').lstrip('HTTPS://')
    return _abs_path(default_archive_location, *name.split('/'))

def _is_url(s):
    s = s.lower()
    return s.startswith('http:/') or s.startswith('https:/')

def _definition_file_to_read(local_path_or_definition_file):
    """return absolute path for sound definition file name.

    The file or path may or may not exist.
    """
    local_path = _abs_path(local_path_or_definition_file)
    if os.path.isfile(local_path):
        return local_path
    else:  # local_path may not exist
        if '.txt' in os.path.split(local_path)[-1]:
            return local_path  # assume that filename is already part of path
        return os.path.join(local_path, default_definition_filename)

def _definition_file_to_write(local_path_or_filename,
                              filename=None):
    """return absolute path to a possibly non-exisiting definition file name.

    Creates a backup if the file exists. Does not create the file or folders
    when they do not exist.

    Details: if ``filename is None``, tries to guess whether the first
    argument already includes the filename. If it seems necessary,
    `default_definition_filename` is appended.
    """
    if filename:
        local_path_or_filename = os.path.join(local_path_or_filename,
                                              filename)
    else:  # need to decide whether local_path contains the filename
        _p, f = os.path.split(local_path_or_filename)
        # append default filename if...
        if '.' not in f or len(f.rsplit('.', 1)[1]) > 4:
            local_path_or_filename = os.path.join(local_path_or_filename,
                                            default_definition_filename)
        elif f != default_definition_filename:
            warnings.warn(
                'Interpreted "%s" as definition filename. This\n'
                'will fail in the further processing which expects the default\n'
                'filename %s.' % (f, default_definition_filename))
    fullname = _abs_path(local_path_or_filename)
    _make_backup(fullname)
    return fullname

def _hash(file_name, hash_function=hashlib.sha256):
    """compute hash of file `file_name`"""
    with open(file_name, 'rb') as file_:
        return hash_function(file_.read()).hexdigest()

def _str_to_list(str_or_list):
    """try to return a non-string iterable in either case"""
    if isinstance(str_or_list, (tuple, list, set)):
        return str_or_list
    if str(str_or_list) == str_or_list:
        return [str_or_list]
    raise ValueError(str_or_list)

def _old_move_official_local_data():
    """move "official" archives folder to the generic standardized location once and for all"""
    src = os.path.join(cocopp_home, 'data-archive')
    dest = _url_to_folder_name(coco_url)
    if os.path.exists(src):
        if not os.path.exists(os.path.join(dest, 'data-archive')):  # obsolete
            _makedirs(dest)
            print("moving %s to %s" % (src, dest))
            _shutil.move(src, dest)
        else:
            warnings.warn("could not move the official archive location "
                          "\n%s\n because \n%s\n already exists."
                          "\nTo prevent this message in future, remove either one or"
                        " the other folder (preferably the smaller one)" % (src, dest))

def _repr_definitions(list_):
    return repr(sorted(list_, key=lambda t: (t[0].split('/')[0], t))).replace('(', '\n(')

def _url_add(folder, url):
    """add ``('_url_', url),`` to the definition file in `folder`.
    
    This function is idempotent, however different urls may be in the list.
    """
    defs = read_definition_file(folder)
    if ('_url_', url) not in defs:
        with open(_definition_file_to_write(folder), 'wt') as f:
            f.write(_repr_definitions([('_url_', url)] + defs))

def _download_definitions(url, target_folder):
    """download definition file and sync url into it"""
    _urlretrieve(url + '/' + default_definition_filename,
                 _definition_file_to_write(target_folder))
    try:
        read_definition_file(_definition_file_to_read(target_folder))
    except:
        warnings.warn('Downloaded definition file\n  %s\n  -> %s \n'
                      'seems not to have the proper format.\n'
                      'Make sure that the above URL is a valid definition file and\n'
                      'that the machine is connected to the www, and try again.'
                      % (url + '/' + default_definition_filename,
                         _definition_file_to_read(target_folder)))
        raise

def _get_remote(url, target_folder=None, redownload=False):
    """return remote data archive as `COCODataArchive` instance.

    If necessary, the archive is "created" by downloading the definition file
    from `url` to `target_folder` which doesn't need to exist.
    
    Details: The target folder name is by default derived from the `url` and
    created within ``default_archive_location == ~/.cocopp/data-archives``.
    """
    key = url
    url = official_archives.url(url) or url.rstrip('/')
    target_folder = target_folder or _url_to_folder_name(url)
    # if key in official_archives.names:  # old code
    #     _move_official_local_data()  # once and for all
    if redownload or not os.path.exists(_definition_file_to_read(target_folder)):
        _makedirs(target_folder)
        _download_definitions(url, target_folder)
        _url_add(target_folder, url)
        if not official_archives.url(key) and not url in official_archives.urls.values():
            ArchivesKnown.register(url)
        else:  # TODO: check that ArchivesOfficial is in order?
            pass
    # instantiate class the url is not used as it must be now in the definition file
    arch = official_archives.class_(key)(target_folder)
    if key in official_archives.names:
        arch.name = key + ' (official)' # should stay?
    if arch.remote_data_path is None:
        _url_add(target_folder, url)
        warnings.warn("No URL found in %s.\n"
            "'_url_' = '%s' added.\n"
            "This typically happens if the definition file was newly created locally\n"
            "and could mean that remote and local definition files are out of sync.\n"
            "You may want to upload the above local definition file to the above URL\n"
            "or use the `update` method to re-download the remote definition file."
            % (_definition_file_to_read(target_folder), url))
        arch.remote_data_path = url
    assert arch.remote_data_path.replace('https', 'http') == url.replace('https', 'http')  # check that url was in the definition file
    return arch

def get(url_or_folder=None):
    """return a data archive `COCODataArchive`.

    `url_or_folder` must be an URL or a folder, any of which must contain
    an archive definition file of name `coco_archive_definition.txt`. Use
    `create` to create this file if necessary.

    When an URL is given the archive may already exist locally from
    previous calls of `get`. Then, ``get(url).update()`` updates the
    definition file and returns the updated archive. Only the definition
    file is updated, no data are downloaded before they are requested. The
    updated class instance re-downloads requested data when the saved hash
    disagrees with the computed hash. With new instances of the archive, if
    `COCODataArchive.update` is not called on them, an error message
    may be shown when they try to use outdated local data and the data can
    be deleted manually as specified in the shown message.

    Remotely retrieved archive definitions are registered with `ArchivesKnown`
    and ``cocopp.archiving.ArchivesKnown()`` will show a list.

    >>> import cocopp
    >>> url = 'https://cma-es.github.io/lq-cma/data-archives/lq-gecco2019'
    >>> arch = cocopp.archiving.get(url).update()  # downloads a 0.4KB definition file
    >>> len(arch)
    4
    >>> assert arch.remote_data_path.split('//', 1)[1] == url.split('//', 1)[1], (arch.remote_data_path, url)
 
    See `cocopp.archives` for "officially" available archives.

    See also: `get_all`, `get_extended`.
    """
    if url_or_folder in (None, 'help'):
        raise ValueError(
                  '"Officially" available archives are\n    %s\n'
                  "Otherwise available (known) archives are\n    %s\n"
                  "Local archives are \n    %s"
                  % (str(official_archives.names),
                     str(ArchivesKnown()),
                     str(ArchivesLocal())))
    if (url_or_folder.lower().startswith("http")
        or url_or_folder in official_archives.names):
        return _get_remote(url_or_folder)
    return COCODataArchive(url_or_folder)

def read_definition_file(local_path_or_definition_file):
    """return definition triple `list`"""
    filename = _definition_file_to_read(local_path_or_definition_file)
    with open(filename, 'rt') as file_:
        try:
            return ast.literal_eval(file_.read())
        except:
            warnings.warn(
                "Failed to properly read archive definition file"
                "\n  '%s'\nThe file may have the wrong format." % filename)
            raise

def create(local_path):
    """create a definition file for an existing local "archive" of data.

    The archive in `local_path` must have been prepared such that it
    contains only (tar-g-)zipped data set files, one file for each data
    set / algorithm, within an otherwise arbitrary folder structure (it is
    possible and for large archives often desirable to create and maintain
    sub-archives within folders of an archive). Choose the name of the zip
    files carefully as they become the displayed algorithm names.

    If a definition file already exists it is backed up and replaced.

    The "created" archive is registered with `ArchivesLocal` serving as a
    user-owned machine-wide memory. ``cocopp.archiving.ArchivesLocal()``
    shows the list.

    >>> from cocopp import archiving
    >>> # folder containing the data we want to become known in the archive:
    >>> local_path = 'my-archives/my-first-archive'
    >>>
    >>> my_archive = archiving.create(local_path)  # doctest:+SKIP
    >>> same_archive = archiving.get(local_path)  # doctest:+SKIP

    An archive definition file is a list of (relative file name,
    hash and (optionally) filesize) triplets.

    Assumes that `local_path` points to a complete and sane archive or
    a definition file to be generated at the root of this archive.

    In itself this is not particularly useful, because we can also
    directly load or use the zip files instead of archiving them first and
    accessing the data then from the archive class within Python.

    However, if the data are put online together with the definition file,
    everyone can locally re-create this archive via `get` and use the
    returned `COCODataArchive` without downloading any data
    immediately, but only "on demand".

    """
    backup_file = backup_last_filename
    definition_file = _definition_file_to_write(local_path)
    if backup_file != backup_last_filename:
        warnings.warn("previous definition file has been back upped to %s"
                      % backup_last_filename)
    full_local_path = os.path.split(definition_file)[0]
    res = []
    for dirpath, _dirnames, filenames in os.walk(full_local_path):
        for filename in filenames:
            fnlower = filename.lower()
            if ('.extracted' not in dirpath
                and not fnlower.startswith('.')
                and not default_definition_filename in filename
                and not fnlower == 'readme'
                and not fnlower.endswith(('.git', '.dat', '.rdat', '.tdat', '.info',
                                          '.txt', '.md', '.py', '.ipynb', '.pdf'))
                and not '.txt' in fnlower
                ):
                if filename[-1] not in ('2', 'z') and filename[-2:] not in ('ar', ) and '.zip' not in filename:
                    warnings.warn('Trying to archive unusual file "%s".'
                        'Remove the file from %s and call `create` again '
                        'if the file was not meant to be in the archive.'
                        % (filename, dirpath))
                name = dirpath[len(full_local_path) + 1:].replace(os.path.sep, '/')
                # print(dirpath, local_path, name, filename)
                name = '/'.join([name, filename]) if name else filename
                path = os.path.join(dirpath, filename)
                res += [(name,
                         _hash(path),
                         int(os.path.getsize(path) // 1000))] # or os.stat(path).st_size
                if 'L)' in name:
                    raise ValueError("Name '%s' at %s contains 'L)' which"
                                     " is not allowed."
                                     "\nPlease change the filename."
                                     % (name, path))
    if not len(res):
        warnings.warn('cocopp.archiving.create: no data found in %s' % local_path)
        return
    with open(definition_file, 'wt') as file_:
        file_.write(_repr_definitions(res).replace('L)', ')'))
    ArchivesLocal.register(full_local_path)  # to find splattered local archives easily
    return COCODataArchive(full_local_path)


class COCODataArchive(_td.StrList):
    """Data archive based on an archive definition file.

    This class is not meant to be instantiated directly. Instead, use
    `cocopp.archiving.get` to get a class instance. The class needs an
    archive definition file to begin with, as created with
    `cocopp.archiving.create`.

    See `cocopp.archives` or `cocopp.archiving.official_archives` for the
    "official" archives.

    This class "is" a `list` (`StrList`) of names which are relative file
    names separated with slashes "/". Each name represents the zipped data
    from a full archived experiment, benchmarking one algorithm on an
    entire benchmark suite.

    The function `create` serves to create a new user-defined archive from
    experiment data which can be loaded with `get`. Other derived classes define
    other specific (sub)archives.

    Using the class
    ---------------

    Calling the class instance (alias to `find`) helps to extract entries
    matching one or several substrings, e.g. a year or a method.
    `find_indices` returns the respective indices instead of the names.
    `print` displays both. For example:

    >>> import cocopp
    >>> cocopp.archives.bbob.find('bfgs')  # doctest:+SKIP
    ['2009/BFGS_ros_noiseless.tgz',
     '2012/DE-BFGS_voglis_noiseless.tgz',
     '2012/PSO-BFGS_voglis_noiseless.tgz',
     '2014-others/BFGS-scipy-Baudis.tgz',
     '2014-others/L-BFGS-B-scipy-Baudis.tgz'...

    To post-process these data call:

    >>> cocopp.main(cocopp.archives.bbob.get_all('bfgs'))  # doctest:+SKIP

    Method `get` downloads a single "matching" data set if necessary and
    returns the absolute data path which can be used with
    `cocopp.main`.

    Method `index` is inherited from `list` and finds the index of the
    respective name entry in the archive (exact match only).

    `cocopp.archives.all` contains all experimental data for all test
    suites.

    >>> import cocopp
    >>> bbob = cocopp.archives.bbob  # the bbob testbed archive
    >>> len(bbob) > 150
    True

    >>> bbob[:3]  # doctest:+ELLIPSIS,+SKIP,
    ['2009/...
    >>> bbob('2009/bi')[0]  # doctest:+ELLIPSIS,+SKIP,
    '...

    Get a `list` of already downloaded data full pathnames or `None`:

    >>> [bbob.get(i, remote=False) for i in range(len(bbob))] # doctest:+ELLIPSIS
    [...

    Find something more specific:

    >>> bbob('auger')[0]  # == bbob.find('auger')[0]  # doctest:+SKIP,
    '2009/CMA-ESPLUSSEL_auger_noiseless.tgz'

    corresponds to ``cocopp.main('auger!')``.

    >>> bbob.index('2009/CMA-ESPLUSSEL_auger_noiseless.tgz')  # just list.index
    5

    >>> data_path = bbob.get(bbob(['au', '2009'])[0], remote=False)
    >>> assert data_path is None or str(data_path) == data_path

    These commands may download data, to avoid this the option ``remote=False`` is given:

    >>> ' '.join(bbob.get(i, remote=False) or '' for i in [2, 13, 33])  # can serve as argument to cocopp.main  # doctest:+ELLIPSIS,+SKIP,
    '...
    >>> bbob.get_all([2, 13, 33], remote=False).as_string  # is the same  # doctest:+ELLIPSIS,+SKIP,
    ' ...
    >>> ' '.join(bbob.get(name, remote=False) for name in [bbob[2], bbob[13], bbob[33]])  # is the same  # doctest:+ELLIPSIS,+SKIP,
    '...
    >>> ' '.join(bbob.get(name, remote=False) for name in [
    ...         '2009/BAYEDA_gallagher_noiseless.tgz',
    ...         '2009/GA_nicolau_noiseless.tgz',
    ...         '2010/1komma2mirser_brockhoff_noiseless.tar.gz'])  # is the same  # doctest:+ELLIPSIS,+SKIP,
    '...

    DONE: join with COCODataArchive, to get there:
    - DONE upload definition files to official archives
    - DONE? use uploaded definition files (see official_archive_locations in `_get_remote`)
    - DONE? replace usages of derived data classes by `get`
    - DONE remove definition list in code of the root class
    - DONE review and join classes without default for local path

    """
    def __init__(self, local_path):
        """Argument is a local path to the archive.

        This class is not anymore meant to be used directly, rather use
        `cocopp.archiving.get`.

        `local_path` is an archive folder containing a definition file,
        possibly downloaded with `get` calling `_get_remote` from a given `url`.
        ``~`` may refer to the user home folder.

        Set `_all` and `self` from `_all` without `_url_`` entry.
        This init does not deal with remote logic, it only reads in _url_ from
        the definition file into the `remote_data_path` attribute.

        Details: Set `_all_dict` which is a (never used) dictionary
        generated from `_all` and `self` and consists of the keys except
        for ``'_url_'``.
        """
        local_path = _abs_path(local_path)
        if not local_path:
            raise ValueError("local path folder needs to be defined")
        if os.path.isfile(local_path):  # ignore filename
            # TODO: shall become a ValueError!?
            local_path, fn = os.path.split(local_path)
            if fn:
                warnings.warn("COCODataArchive.__init__: filename"
                              " %s in %s ignored" % (fn, local_path))
        if not COCODataArchive.is_archive(local_path):
            raise ValueError('The folder "%s" seems not to "be" a COCO data'
                            " archive as it does not contain a %s file)."
                            "\nUse `create(folder)` or `get(URL)` of"
                            " `cocopp.archiving` to create/get this file."
                            % (local_path, default_definition_filename))
        self.local_data_path = local_path
        self._names_found = []  # names recently found
        self._redownload_if_changed = []
        self._checked_consistency = False
        self._print = print  # like this we can make it quiet for testing
        self._all = self.read_definition_file()
        assert hasattr(self, '_all')
        self.remote_data_path = self._url_(self._all)  # later we could use self._all_dict.get('_url_', None)
        if not self.remote_data_path and len(self) != len(self.downloaded):
                warnings.warn(
                    "defined=%d!=%d=downloaded data sets and no url given"
                    % (len(self), self.downloaded))
        self._all_dict = dict((kv[0], kv[1:]) for kv in self._all)
        if len(self._all_dict) != len(self._all):  # warn on double entries
            keys = [v[0] for v in self._all]
            warnings.warn("definitions contain double entries %s" %
                          str([v for v in self._all if keys.count(v[0]) > 1]))
        if self.remote_data_path and self._all_dict.setdefault("_url_",
                                       (self.remote_data_path, )) != (self.remote_data_path, ):
            warnings.warn("found different remote paths \n    %s\n vs %s"
                          % (self.remote_data_path, self._all_dict["_url_"]))
        _td.StrList.__init__(self, (kv[0] for kv in self._all if kv[0] != '_url_'))
        self.consistency_check_read()
        if 11 < 3:  # this takes too long on importing cocopp
            self.consistency_check_data()

    def get_found(self, remote=True):
        """get full entries of the last `find`"""
        return self.get_all(self.found or None, remote=remote)

    def get_all(self, indices=None, remote=True):
        """Return a `list` (`StrList`) of absolute pathnames,

        by repeatedly calling `get`. Elements of the `indices` list can
        be an index or a (sub)string that matches one or several names
        in the archive. If ``indices is None``, the results from the
        last call to `find` are used. Data are downloaded if necessary.

        See `find` or `cocopp.archiving.OfficialArchives` for how matching
        is determined.

        See also `get`, `get_extended`.
        """
        if indices is not None:  # TODO: just "if indices" should do?
            names = self.find(indices)
        else:
            names = self.found
        return _td.StrList(self.get(name, remote=remote)
                           for name in names)

    def get_first(self, substrs, remote=True):
        """get the first archived data matching all of `substrs`.

        `substrs` is a list of substrings.

        `get_first(substrs, remote)` is a shortcut for::

            self.find(*substrs)
            if self.found:
                return self.get(self.found[0], remote=remote)
            return None

        """
        self.find(*_str_to_list(substrs))
        if self.found:
            return self.get(self.found[0], remote=remote)
        return None

    def get(self, substr=None, remote=True):
        """return the full data pathname of `substr` in the archived data.

        Retrieves the data from remote if necessary.

        `substr` can be a substring that matches one and only one name in
        the data archive or an integer between 0 and `len(self)`, see
        `find` or `cocopp.archiving.OfficialArchives` for how matching is
        determined.

        Raises a `ValueError` if `substr` matches several archive entries
        on none.

        If ``substr is None`` (default), the first match of the last
        call to ``find*`` or ``get*`` is used like `self.found[0]``.

        If ``remote is True`` (default), the respective data are
        downloaded from the remote location if necessary. Otherwise
        return `None` for a match.
        """
        # handle different ways to pass arguments, create names
        if not isinstance(remote, bool):
            raise ValueError(
                "Second argument to `COCODataArchive.get` must be a "
                "`bool`,\n that is ``remote=True`` or `False`."
                "Use a `list` of `str` to define several substrings.")
        if substr is None:
            try:
                names = [self.found[0]]
            except IndexError:
                raise ValueError("nothing specified to `get`, use `find` "
                                 "first or give a name")
        elif isinstance(substr, int):
            names = [self[substr]]
        else:
            names = self.find(substr)

        # check that names has only one match
        if len(names) < 1:
            if len(self) > 0:  # there are known data entries but substr doesn't match
                raise ValueError("'%s' has no match in data archive" % substr)
            # a blank archive, hope for the best, match must be exact
            names = [substr]
        elif len(names) > 1:
            raise ValueError(
                "'%s' has multiple matches in the data archive:\n   %s\n"
                "Either pick a single match, or use the `get_all` or\n"
                "`get_first` method, or use the ! (first) or * (all)\n"
                "marker and try again."
                % (substr, '\n   '.join(names)))
        # create full path
        full_name = self.full_path(names[0])
        if os.path.exists(full_name):
            try:
                self.check_hash(full_name)
            except ValueError:
                if names[0] in self._redownload_if_changed:
                    self._download(names[0])
                    self.check_hash(full_name)
                elif 11 < 3 and not input("\n\n  ** wrong hash, download {} again? [n=no, return=yes] **".format(names[0])).lower().startswith('n'):
                    # seems to break tests?
                    self._download(names[0])
                    self.check_hash(full_name)
                else:
                    raise
            try: self._redownload_if_changed.remove(names[0])
            except ValueError: pass
            return full_name
        if not remote:
            return ''  # like this string operations don't bail out
        self._download(names[0])
        return full_name

    def _download(self, name):
        """create full local path and download single dataset"""
        url = '/'.join((self.remote_data_path, name))
        full_name = self.full_path(name)
        _makedirs(os.path.split(full_name)[0])  # create path if necessary
        self._print("  downloading %s to %s" % (url, full_name))
        try:
            _urlretrieve(url, full_name)
        except BaseException:  # KeyboardInterrupt is a BaseException
            if os.path.exists(full_name):
                os.remove(full_name)  # remove partial download
            raise
        self.check_hash(full_name)

    def get_one(self, *args, **kwargs):
        """deprecated, for backwards compatibility only, use `get` instead
        """
        warnings.warn("use get instead", DeprecationWarning)
        return self.get(*args, **kwargs)

    def get_extended(self, args, remote=True):
        """return a list of valid paths.

        Elements in `args` may be a valid path name or a known name from the
        data archive, or a uniquely matching substring of such a name, or a
        matching substring with added "!" in which case the first match is taken
        only (calling `self.get_first`), or a matching substring with added "*"
        in which case all matches are taken (calling `self.get_all`), or a
        regular expression containing a `*` and not ending with `!` or `*`, in
        which case, for example, "bbob/2017.*cma" matches
        "bbob/2017/DTS-CMA-ES-Pitra.tgz" among others (in a regular expression
        "." matches any single character and ".*" matches any number >= 0 of
        characters).

        """
        res = []
        args = _str_to_list(args)
        for name in args:
            name = name.strip()
            if os.path.exists(name):
                res.append(name)
                continue
            for try_ in range(2):
                more = []
                if name.endswith('!'):  # take first match
                    more.append(self.get_first([name[:-1]], remote=remote))
                elif name.endswith('*'):  # take all matches
                    more.extend(self.get_all(name[:-1], remote=remote))
                elif '*' in name:  # use find which also handles regular expressions
                    more.extend(self.get(found, remote=remote)
                                for found in self.find(name))
                elif self.find(name):  # get will bail out if there is not exactly one match
                    more.append(self.get(name, remote=remote))
                if more and more[-1] is not None:
                    if try_ == 1:
                        print('2nd try succeeded')
                    break
                if not remote or try_ > 0:
                    raise ValueError('"%s" seems not to be an existing file or match any archived data'
                                     % name)
                warnings.warn('COCODataArchive failed to locate "%s".\n'
                              'Will try again after updating from %s'
                              % (name, self.remote_data_path))
                self.update()
            res.extend(more)
        if len(args) != len(set(args)):
            warnings.warn("Several data arguments point to the very same "
                          "location. This will likely lead to \n"
                          "rather unexpected outcomes.")
            # TODO: we would like the users input with timeout to confirm
            # and otherwise raise a ValueError
        return res

    def _name(self, full_path):
        """return supposed name of full_path or name without any checks"""
        assert self.local_data_path
        if full_path.startswith(self.local_data_path):
            name = full_path[len(self.local_data_path) + 1:]  # assuming the path separator has len 1
        else:
            name = full_path
        return name.replace(os.path.sep, '/')  # this may not be a 100% fix

    def contains(self, name):
        """return `True` if (the exact) name or path is in the archive"""
        return self._name(name) in self

    @property
    def downloaded(self):
        """return `list` of data set names of locally available data.

        This is only meaningful for a remote archive.
        """
        return [name for name in self
                if os.path.isfile(self.full_path(name))]

    def full_path(self, name):
        """return full local path of `name` or any path, idempotent
        """
        name = self._name_with_check(name)
        return os.path.join(self.local_data_path,
                            os.path.join(*name.split('/')))

    def _name_with_check(self, full_path):
        """return name of `full_path`, idempotent.

        If `full_path` is not from the data archive a warning is issued
        and path seperators are replaced with `/`.

        Check that all names are only once in the data archive:

        >>> import cocopp
        >>> bbob = cocopp.archives.bbob
        >>> for name in bbob:
        ...     assert bbob.count(name) == 1, "%s counted %d times in data archive" % (name, bbob.count(name))
        ...     assert len(bbob.find(name)) == 1, "%s found %d times" % (name, bbob.find(name))

        """
        name = self._name(full_path)
        if name not in self:
            warnings.warn('name "%s" is not defined as member of this '
                          'COCODataArchive' % name)
        return name

    def consistency_check_data(self):
        """basic quick consistency check of downloaded data.

        return ``(number_of_checked_data, number_of_all_data)``
        """
        for name in self.downloaded:
            self.check_hash(name)
        self._checked_consistency = True
        return len(self.downloaded), len(self)

    def check_hash(self, name):
        """raise Exception when hashes disagree or file is missing.

        raise RunTimeError if hash is unknown
        raise ValueError if hashes disagree
        """
        known_hash = self._known_hash(name)
        if known_hash is None:
            raise RuntimeError(
                'COCODataArchive has no hash checksum for\n  %s\n'
                'The computed checksum was \n  %s\n'
                'To remove this warning, consider to manually insert this hash in `COCODataArchive._all_coco_remote`\n'
                'Or, if this happens for many different data, consider using `create` to\n'
                'compute all hashes of local data and then manually insert the hash in _all.\n'
                'Or consider filing a bug report (issue) at https://github.com/numbbo/coco/issues'
                '' % (name, self._hash(name)))
        elif self._hash(name) != known_hash:
            raise ValueError(
                'wrong checksum for\n\n   %s\n\n in archive\n\n   %s\n   %s\n\n'
                'Consider to (re)move file\n'
                '   %s\n'
                'as it may be a partial or unsuccessful download.\n'
                'A missing file will be downloaded again by `get`.\n'
                'Alternatively, call `cocopp.archives.this-archive.update()`\n'
                'of the respective data archive to update definitions\n'
                'and checksums and allow for automatic re-downloads.\n'
                'If this is not a remote archive consider to re-`create` it.'
                '' % (name, self.local_data_path, str(self.remote_data_path),
                self.full_path(name)))

    def _hash(self, name, hash_function=hashlib.sha256):
        """compute hash of `name` or path"""
        return _hash(self.full_path(name) if name in self else name)

    def _known_hash(self, name):
        """return known hash or `None`
        """
        try:
            return self._all_dict[self._name_with_check(name)][0]
            # was depending on order consistency of self and self._all:
            # return self._all[self.index(self.name(name))][1]
        except KeyError:
            return None

    def update(self):
        """update definition file, either from remote location or from local data.
        
        As remote archives may grow or change, a common usecase may be
        
        >>> import cocopp.archiving as ac
        >>> url = 'https://cma-es.github.io/lq-cma/data-archives/lq-gecco2019'
        >>> arch = ac.get(url).update()  # doctest:+SKIP
        
        For updating a local archive use::
        
            create(self.local_data_path)
        
        Details: for updating the local definition file from the local data
        rather use `create`. This will however remove a remote URL from its
        definition and the remote and the local archive can be different
        now. `create` makes a backup of the existing definition file.
        """
        if self.remote_data_path:
            _get_remote(self.remote_data_path, self.local_data_path,
                        redownload=True)  # redownload definition file
            # allow to re-download data by tagging possibly outdated data
            self._redownload_if_changed = self.downloaded
        else:
            print('This archive has no remote URL. If you intended to update the\n'
                  'definition file from the local data, call\n'
                  '    create("%s")\n'
                  'and\n'
                  '    get("%s")\n'
                  'to get an updated instance.'
                  % (2 * [self.local_data_path]))
        # update the definition list in this class
        # and the URL in the definition file:
        self.__init__(self.local_data_path)
        return self  # allow for ca = get(...).update()

    def consistency_check_read(self):
        """check/compare against definition file on disk"""
        all = self.read_definition_file()
        diff = set(all).symmetric_difference(self._all)
        assert len(diff) == 0 or (len(diff) == 1 and
                                  list(diff)[0][0] == "_url_" and
                                  list(diff)[0][1] == self.remote_data_path)

    def _url_(self, definition_list=None):
        """return value of _url_ entry in `definition_list` file or `None`.
        """
        if definition_list is None:
            definition_list = self.read_definition_file()
        url_in_defs = [entry for entry in definition_list
                       if entry[0] == '_url_']
        if len(url_in_defs) > 1:
            raise ValueError(url_in_defs)
        elif len(url_in_defs) == 1:
            return url_in_defs[0][1]
        return None

    def read_definition_file(self):
        """return definition triple `list`"""
        return read_definition_file(self.local_data_path)

    @staticmethod
    def is_archive(url_or_folder):
        """return `True` if `folder` contains a COCO archive definition file"""
        for folder in (_abs_path(url_or_folder), _url_to_folder_name(url_or_folder)):
            if (os.path.exists(folder) and
                default_definition_filename in os.listdir(folder)):
                return True
        return False

class COCOBBOBDataArchive(COCODataArchive):
    """`list` of archived data for the 'bbob' test suite.

    To see the list of all data from 2009:

    >>> import cocopp
    >>> cocopp.archives.bbob.find("2009/")  # doctest:+ELLIPSIS,+SKIP,
    ['2009/ALPS...

    To use the above list in `main`:

    >>> cocopp.main(cocopp.archives.bbob.get_all("2009/")  # doctest:+SKIP

    `get_all` downloads the data from the online archive if necessary.
    While the data are specific to the `COCOBBOBDataArchive` class, all
    functionality is inherited from the parent `class` `COCODataArchive`:

    """
    __doc__ += COCODataArchive.__doc__

class COCOBBOBNoisyDataArchive(COCODataArchive):
    """This class "contains" archived data for the 'bbob-noisy' suite.

    >>> import cocopp
    >>> cocopp.archives.bbob_noisy  # doctest:+ELLIPSIS,+SKIP,
    ['2009/ALPS_hornby_noisy.tgz',...
    >>> isinstance(cocopp.archives.bbob_noisy, cocopp.archiving.COCOBBOBNoisyDataArchive)
    True

    While the data are specific to `COCOBBOBNoisyDataArchive`, all the
    functionality is inherited from the parent `class` `COCODataArchive`:
    """
    __doc__ += COCODataArchive.__doc__

class COCOBBOBBiobjDataArchive(COCODataArchive):
    """This class "contains" archived data for the 'bbob-biobj' suite.

    >>> import cocopp
    >>> cocopp.archives.bbob_biobj  # doctest:+ELLIPSIS,+SKIP,
    ['2016/DEMO_Tusar_bbob-biobj.tgz', '2016/HMO-CMA-ES_Loshchilov_bbob-biobj.tgz',...
    >>> isinstance(cocopp.archives.bbob_biobj, cocopp.archiving.COCOBBOBBiobjDataArchive)
    True

    While the data are specific to `COCOBBOBBiobjDataArchive`, all the
    functionality is inherited from the parent `class` `COCODataArchive`:
    """
    __doc__ += COCODataArchive.__doc__

class ListOfArchives(_td.StrList):
    """List of URLs or path names to COCO data archives available to this user.

    Elements can be added with `append` and deleted with the respective
    `list` operations. After any of these operations, the current state can
    be made "permanent" with `save`.

    Archives created with `create` are automatically added to
    `ArchivesLocal` to find splattered archives more easily, archives
    retrieved from a remote location with `get` are added to
    `ArchivesKnown`.

    Elements of the list are meant to be used directly as argument to `get`.

    >>> import cocopp.archiving as ar
    >>> ar.ListOfArchives.lists()  # doctest:+ELLIPSIS
    {...

    returns all available archive lists.

    To reassign a given list:

    >>> l = ar.ListOfArchives('test')
    >>> l[:] = ['p2', 'p1']  # re-assign a new value to the list
    >>> l.sort()  # `list` builtin in-place sort method
    >>> l.save()  # doctest:+SKIP

    To save the list, a listing name needs to be given on instantiation. Two
    inherited classes are

    >>> ar.ArchivesKnown() and ar.ArchivesLocal()  # doctest:+ELLIPSIS
    [...

    where the former contains downloaded "inofficial" archives. To add all
    locally created archives within the user home directory tree (CAVEAT: the
    search may take a while):

    >>> al = ar.ArchivesLocal()  # based on user home
    >>> al._walk()  # doctest:+SKIP
    >>> # al._save_walk()  # add found paths (permanently) to the ArchivesLocal class

    Details: duplicates are removed during `save`. Path names are stored as
    absolute path names (OS-dependent).

    TODO: should we be able to generate a list on the fly? Figure out usecase with
    _walk?

    TODO-decide: usecase, when to decide setting search_folder?

    """

    listing_file = None  # abstract base class
    search_folder = None

    def __init__(self, listing_name=None):
        """print available archive lists if no listing file is given
        """
        if listing_name:
            listing_name = self._fullfile(listing_name)
        elif not self.listing_file:
            raise ValueError("Available lists:\n %s" % str(self.lists()))
        # may overwrite/change list_file which should be fine
        self.listing_file = _abs_path(listing_name or type(self).listing_file)
        self._last_walk = []
        if not os.path.exists(self.listing_file):
            _makedirs(os.path.split(self.listing_file)[0])
            if 11 < 3:  # TODO-decide: do we want this?
                print("cocopp.archiving: initializing once and for all "
                      "%s for %s, this may take a while"
                        % (str(self.listing_file), str(type(self))))
                self._walk()
                self._save_walk()
        else:  # file is not generated if it does not exist
            with open(self.listing_file, 'rt') as f:
                list.__init__(self, ast.literal_eval(f.read()))

    @property
    def name(self):
        """name of this list"""
        return self._name(self.listing_file)
    @staticmethod
    def _name(listing_file):
        """"""
        return listing_file.split(listing_file_start)[1].split(
                                  listing_file_extension)[0]
    @staticmethod
    def _file(listing_name):
        return listing_file_start + listing_name + listing_file_extension
    
    @staticmethod
    def _fullfile(listing_name):
        return _abs_path(cocopp_home_archives,
                         ListOfArchives._file(listing_name))

    @staticmethod
    def lists():
        lists = [ListOfArchives._name(n) for n in os.listdir(cocopp_home_archives)
                 if n.startswith(listing_file_start) and
                    not listing_file_extension + "_2" in n]  # backups add the date like "...txt_2019-04-29_17h23m45s"
        d = {}
        for name in lists:
            with open(ListOfArchives._fullfile(name), 'rt') as f:
                d[name] = ast.literal_eval(f.read())  # read list of archive paths
        return d

    @classmethod
    def register(cls, folder):
        """add folder path or url to list of archives.

        Caveat: existing class instances won't be aware of new
        registrations.
        """
        folder = folder.rstrip(os.path.sep).rstrip('/')  # should not be necessary
        ls = cls()
        if folder not in ls:
            ls.append(folder)
            ls.save()  # deals with duplicates

    def _makepathsabsolute(self):
        for i in range(len(self)):
            if not _is_url(self[i]):
                self[i] = _abs_path(self[i])

    def save(self):
        """save current list making changes permanent"""
        for foldername in self:
            if not COCODataArchive.is_archive(foldername):
                raise ValueError('"%s" is not an archive, save aborted' % foldername)
        self._makepathsabsolute()
        self._remove_double_entries()  # can have gotten here from append or extend
        _make_backup(self.listing_file)
        with open(self.listing_file, 'wt') as f:
            f.write(_repr_definitions(self))

    def remote_update(self, name=None):
        """join in the respective list from ``coco_url.rsplit('/')[0] + '/data-archives'``.
        
        Use `save` to save the joined entries.
        """
        for s in RemoteListOfArchives(name or self.name):  # download and read list
            if s not in self:
                print(s, 'appended')
                self.append(s)

    def _remove_double_entries(self):
        """keep the first of duplicated entries"""
        l = []
        l.extend(v for v in self if v not in l)  # depends on l changing in generator
        self[:] = l

    def _walk(self, folder=None):
        """recursive search for COCO data archives in `folder` on disk.

        This may take (many?) seconds, for example if using ``folder='~'``.

        `_last_walk` contains the returned result and _save_walk` makes the
        result "permanent" to the current class instance / listing file.
        """
        if folder is None:
            folder = self.search_folder
        # skip_home = cocopp_home not in _abs_path(folder)  # see below
        res = []
        for dirpath, dnames, fnames in os.walk(_abs_path(folder), followlinks=False):
            dnames[:] = [d for d in dnames
                         if not d.startswith('.')]  # do not descent in any . folder, e.g. .Trash or .cocopp
            if default_definition_filename in fnames:
                res.append(_abs_path(dirpath))
        self._last_walk = res
        return res

    def _save_walk(self):
        """extend `self` by elements of `_last_walk` not in `self` and `save`
        """
        self.extend(v for v in self._last_walk if v not in self)  # also prevents adding elements twice
        self.save()

    def update(self):
        """update self from the listing file that may have changed"""
        with open(self.listing_file, 'rt') as f:
            self[:] = ast.literal_eval(f.read())  # or list.__init__(self, ...)

class OfficialArchives(object):
    """overdesigned class to connect URLs, names, and classes of "official" archives.

    The `all` archive name gets special treatment. The 'all'-archive is
    currently needed in `cocopp.main` but could be dropped otherwise.

    The class collects the "official" online data archives as attributes.
    Each suite is also a different class inheriting from `COCODataArchive`.
    The class may become the same for some or all suites in future.

    `cocopp.archives` is an instance of `OfficialArchives` and contains as
    archive attributes (at this point in time):

    ``all``: `COCODataArchive`, the `list` of all archived data from
    all test suites.

    ``bbob``: `COCOBBOBDataArchive`, the `list` of archived data run on the
    `bbob` test suite.

    ``bbob_noisy``: `COCOBBOBNoisyDataArchive`, ditto on the `bbob_noisy`
    test suite.

    ``bbob_biobj``: `COCOBBOBBiobjDataArchive`, ditto...

    The `names` property is a list of all "official" archive names available.

    A Quick Guide
    -------------

    **1) To list all available data**:

    >>> import cocopp
    >>> cocopp.archives.all  # doctest:+ELLIPSIS,+SKIP,
    ['bbob/2009/AL...

    **2) To see all available data from a given test suite**:

    >>> cocopp.archives.bbob  # doctest:+ELLIPSIS,+SKIP,
    ['2009/ALPS...

    or

    >>> cocopp.archives.bbob_biobj  # doctest:+ELLIPSIS,+SKIP,
    ['2016/DEMO_Tusar...

    or

    >>> cocopp.archives.bbob_noisy  # doctest:+ELLIPSIS,+SKIP,
    ['2009/ALPS...

    **3) We can extract a subset** from any test suite (such as
    `cocopp.archives.all`, `cocopp.archives.bbob`, ...) of the
    archive by sub-string matching:

    >>> cocopp.archives.bbob.find('bfgs')  # doctest:+NORMALIZE_WHITESPACE,+ELLIPSIS,+SKIP
    ['2009/BFGS_ros_noiseless.tgz',
     '2012/DE-BFGS_voglis_noiseless.tgz',
     '2012/PSO-BFGS_voglis_noiseless.tgz',
     '2014-others/BFGS-scipy-Baudis.tgz',
     '2014-others/L-BFGS-B-scipy-Baudis.tgz'...
    
    or by regex pattern matching:

    >>> cocopp.archives.bbob.find('bfgs') == cocopp.archives.bbob.find('.*bfgs')
    True

    but 

    >>> len(cocopp.archives.bbob.find('bfgs')) > len(cocopp.archives.bbob.find('bfgs.*'))
    True

    The `find` method will not download data and is only for inspecting the
    archives. If we want to actually process the data we need to use `get`,
    `get_all`, `get_extended` or `get_found`, or use the same search string
    in `cocopp.main` appending a `*` if more than one algorithm should be
    processed:

    **4) When postprocessing data via `cocopp.main`**, we can use the archive
    like

    >>> cocopp.main(cocopp.archives.bbob.get_all('bfgs').as_string)  # doctest:+SKIP

    or the shortcut
    
    >>> cocopp.main('bfgs*')  # doctest:+SKIP
    
    using the special meaning of the trailing `*` in this case (see below).
    The `get` and `get_extended` methods are called on each argument given in a
    string to `main`. We can do things like

    >>> cocopp.main('bbob/2009/BIPOP DE-BFGS')  # doctest:+SKIP

    When a string has multiple matches, the postprocessing bails out. For
    such cases, we can use the trailing symbols `*` (AKA take all matches)
    and `!` (AKA take the first match) which uses the `get_extended` method
    under the hood:

    >>> cocopp.main('BIPOP! 2012/DE*')  # doctest:+SKIP

    will expand to the following::

        Post-processing (2+)
          Using:
            /.../.cocopp/data-.../bbob/2009/BIPOP-CMA-ES_hansen_noiseless.tgz
            /.../.cocopp/data-.../bbob/2012/DE-AUTO_voglis_noiseless.tgz
            /.../.cocopp/data-.../bbob/2012/DE-BFGS_voglis_noiseless.tgz
            /.../.cocopp/data-.../bbob/2012/DE-ROLL_voglis_noiseless.tgz
            /.../.cocopp/data-.../bbob/2012/DE-SIMPLEX_voglis_noiseless.tgz
            /.../.cocopp/data-.../bbob/2012/DE_posik_noiseless.tgz
            /.../.cocopp/data-.../bbob/2012/DEAE_posik_noiseless.tgz
            /.../.cocopp/data-.../bbob/2012/DEb_posik_noiseless.tgz
            /.../.cocopp/data-.../bbob/2012/DEctpb_posik_noiseless.tgz

        Post-processing (2+)
          loading data...
        [...]

    **5) If we want to also pass other arguments to the postprocessing**
    (e.g. the output folder) in case 3) above, string concatenation does
    the trick:

    >>> cocopp.main('-o myoutputfolder ' + cocopp.archives.bbob.get_all('bfgs').as_string)  # doctest:+SKIP

    For case 4), this works directly:

    >>> cocopp.main('-o myoutputfolder BIPOP! 2012/DE*')  # doctest:+SKIP

    """
    def __init__(self, url=None):
        """all URLs and classes (optional) in one place.
        
        The archive names are identical with the last part of the URL. The only
        exception is made for `'all'`, which is removed to get the URL.
        """
        self.all = None  # only to prevent lint error in cocopp/__init__.py
        if url is None:
            url = coco_url
        self._base = url.rstrip('/') + '/'
        # self._base = coco_url + '/data-archive/'  # old way
        # TODO-decide: should this list better be initialized by a ListOfArchives file?
        #              (the same transition as before with the _all attribute in COCODataArchive)
        #              The code then only "hardcodes" the class name mapping?
        self._list = [
            (self._base + 'all', COCODataArchive),
            (self._base + 'bbob', COCOBBOBDataArchive),
            (self._base + 'bbob-noisy', COCOBBOBNoisyDataArchive),
            (self._base + 'bbob-biobj', COCOBBOBBiobjDataArchive),
            (self._base + 'bbob-largescale', None),  # TODO: introduce a new class
            (self._base + 'bbob-mixint', None),  # TODO: introduce a new class
            (self._base + 'bbob-constrained', None),  # TODO: introduce a new class
            (self._base + 'sbox-cost', None),  # TODO: introduce a new class
            (self._base + 'test', None),  # None resolves to COCODataArchive
        ]

    def add_archive(self, name):
        """Allow to use a new official archive.
        
        The archive must exist as a subfolder of ``coco_url``.
        """
        self._list += [(self._base + name, None),]
        self.set_as_attributes_in()

    def set_as_attributes_in(self, target=None, except_for=('test',),
                             update=False):
        """Assign all archives as attribute of `target` except for ``'test'``.
        
        `target` is by default `self`.
        
        Details: This method can only be called when the class names and
        the module attribute `official_archives: OfficialArchive` (used in
        `get`) are available. It creates new instances of the archives.
        Depending on the implementation of `get`, it may download the
        definition files on its first-ever call by/on any given
        user/machine.
        """
        for name in self.names:
            if name not in except_for:
                setattr(target or self, name.replace('-', '_'),
                        get(name).update() if update else get(name))
        return self

    def link_as_attributes_in(self, target, except_for=('test',),
                              update=False):
        """Assign all archives as attribute of `target` except for ``'test'``.
        
        `target` is by default `self`.
        
        Details: This method can only be called when the class names and the module
        attribute `official_archives: OfficialArchive` (used in `get`) are
        available. It only creates links to the existing archives.
        """
        for name in self.names:
            if name not in except_for:
                name = name.replace('-', '_')
                setattr(target, name, getattr(self, name))

    @property
    def names(self):
        """a list of valid key names"""
        return [i[0].split('/')[-1] for i in self._list]
    
    @property
    def urls(self):
        """a name->URL dictionary"""
        return dict((key, self.url(key)) for key in self.names)
    
    def _get(self, name, j):
        """get j-th entry of `name`, where j==0 is the URL and j==1 is the class"""
        for tup in self._list:
            if tup[0].endswith(name):
                return tup[j] if j < len(tup) else None

    def url(self, name):
        """return url of "official" archive named `name`.
        
        The same value as ``self.urls.get(name)``.
        """
        url = self._get(name, 0)
        return url.rsplit('/', 1)[0] if name == 'all' else url

    def class_(self, name):
        """class of archive named `name` when returned by `get`"""
        return self._get(name, 1) or COCODataArchive

    def update_all(self):
        """update archive definition files from their remote location.

        The update is necessary to account for newly added data. This
        method requires www connectivity with only a few KB of transmitted
        data.
        """
        # self.set_as_attributes_in(update=True)
        for name in self.names:
            name = name.replace('-', '_')
            try:
                getattr(self, name).update()
            except AttributeError:
                if name != 'test':
                    raise

    def _make_folder_skeleton(self):
        """workaround to avoid bailing when www is not reachable during

        the very first import.
        """
        for url in [l[0] for l in self._list]:
            path = _url_to_folder_name(url)
            if path.endswith('/test'):
                continue
            if path.endswith('/all'):
                path = path[:-4]
                url = url[:-4]
            _makedirs(path)  # may not be necessary?
            df = _definition_file_to_read(path)
            if os.path.exists(df):
                warnings.warn('_make_folder_skeleton(): '
                              'file "{}" exists'.format(df))
            else:
                with open(df, 'w') as f:
                    f.write("[('_url_', '{}')]\n".format(url))

# official_archives = OfficialArchives()
for url in coco_urls[-1::-1]:  # loop over possible URLs until successful
    official_archives = OfficialArchives(url)  # lazy init, does kinda nothing
    coco_url = url
    try:
        # TODO-decide: when should we (try to) update/check these?
        # The following `set_as_attributes_in` calls `cocopp.archiving.get(url)` and works if the
        # connection was successful at least once (before or now)
        official_archives.set_as_attributes_in()  # set "official" archives as attributes by suite name
        break
    except:  # (HTTPError, TimeoutError, URLError)
        warnings.warn("failed to connect to " + url)
else:
    warnings.warn("Failed fo find workable URL or local folder for official archives."
                  "\n After the www connection is restored, you may need to call"
                  "\n `cocopp.archiving.official_archives.update_all()` to create"
                  "\n valid definition files.")
    official_archives._make_folder_skeleton()      
    official_archives.set_as_attributes_in()

class _old_ArchivesOfficial(ListOfArchives):
    """superseded by `OfficialArchives`
    
    Official COCO data archives.

    TODO-decide: this class is not needed, as official archives don't change frequently
    and are directly available in cocopp.archives? Or merge into the above OfficialArchives class, such that we only need to update a list entry in the remote list!?
    """
    __doc__ += ListOfArchives.__doc__

    listing_file = ListOfArchives._fullfile("official_archives")
    search_folder = _abs_path(cocopp_home, "data-archive")  # obsolete

class ArchivesLocal(ListOfArchives):
    """COCO data archives somewhere local on this machine.

    The archives need to be gathered with `_walk`.
    """
    __doc__ += ListOfArchives.__doc__

    listing_file = ListOfArchives._fullfile("local_archives")
    search_folder = _abs_path("~/")  # TODO: somewhat ambitious?

class ArchivesKnown(ListOfArchives):
    """Known (and already used) remote COCO data archives.

    These include the official archives from `OfficialArchives`.
    """
    __doc__ += ListOfArchives.__doc__

    listing_file = ListOfArchives._fullfile("known_archives")
    search_folder = default_archive_location 

class RemoteListOfArchives(_td.StrList):
    """Elements of this list can be used directly with `cocopp.archiving.get`.

    The only purpose of this list is to propose (or remind) known archive
    remote locations to the user. For this purpose, the current listing file is
    downloaded.
    """
    # was: location = coco_url + '/data-archives/'  # + 'list_known_archives.txt'
    # a hack: remove deepest folder name in coco_url
    location = coco_url.rstrip('/').rsplit('/', 1)[0]  # remove lowest folder name
    if '//' not in location:
        location = coco_url.rstrip('/')
    location += '/data-archives/'  # + 'list_known_archives.txt'

    def __init__(self, name='known_archives'):
        super(RemoteListOfArchives, self).__init__(self._download(name))

    def _download(self, name):
        fname = ListOfArchives._file(name)
        destname = os.path.join(cocopp_home_archives, '_remote_' + fname)  # not meant to stay
        _urlretrieve(self.location + fname, destname)
        with open(destname, 'rt') as f:
            return ast.literal_eval(f.read())
    
    def save(self):
        raise NotImplementedError("a remote list cannot be saved")
