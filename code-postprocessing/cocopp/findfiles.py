#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Recursively find :file:`info` and zipped files within a directory and
administer archives.

This module can be called from the shell, it will recursively look for
:file:`info` and :file:`pickle` files in the current working directory::

  $ python -c "from cocopp.findfiles import main; print(main())"

displays found (extracted) files.

TODO: we do not use pickle files anymore.
"""
from __future__ import absolute_import, division, print_function
import os
import sys
import warnings
import tarfile
import zipfile

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve
from .toolsdivers import StringList  # def StringList(list_): return list_
from . import genericsettings

# Initialization


def is_recognized_repository_filetype(filename):
    return (os.path.isdir(filename.strip())
            or filename.find('.tar') > 0
            or filename.find('.tgz') > 0
            or filename.find('.zip') > 0)

def is_recognized_repository_filetype2(filename):
    """return True if `filename` is a file and ends with a recognized extension"""
    n = filename.strip()
    if os.path.isdir(n):
        return False
    return (n.endswith('.tgz') or
            n.endswith('.zip') or
            n.endswith('.tar.gz') or
            n.endswith('.tar')
            )


def main(directory='.'):
    """Lists "data" files recursively in a given directory, tar files
    are extracted.

    The "data" files have :file:`info` and :file:`pickle` extensions.

    TODO: not only recognize .tar and .tar.gz and .tgz but .zip...

    """

    file_list = list()
    root = ''
    directory = get_directory(directory, True)

    # Search through the directory directory and all its subfolders.
    for root, _dirs, files in os.walk(directory):
        if genericsettings.verbose:
            print('Searching in %s ...' % root)

        for elem in files:
            if elem.endswith('.info') or elem.endswith('.pickle') or elem.endswith('.pickle.gz'):
                file_list.append(os.path.join(root, elem))

    if genericsettings.verbose:
        print('Found %d file(s).' % (len(file_list)))
    if not file_list:
        warnings.warn('Could not find any file of interest in %s!' % root)
    return file_list


def get_directory(directory, extract_files):

    directory = directory.strip()

    # if directory.endswith('.zip'):
    #   archive = zipfile.ZipFile(directory)
    #   for elem in archive.namelist():
    #     if elem.endswith('.info'):
    #       (root,elem) = os.path.split(elem)
    #       filelist = IndexFile(root,elem,archive)
    if not os.path.isdir(directory) and is_recognized_repository_filetype(directory):
        if '.zip' in directory:
            head, tail = os.path.split(directory[:directory.find('.z')])
            dir_name = os.path.join(head, genericsettings.extraction_folder_prefix + tail)
            # extract only if extracted folder does not exist yet or if it was
            # extracted earlier than last change of archive:
            if extract_files:
                if (not os.path.exists(dir_name)) or (os.path.getmtime(dir_name) < os.path.getmtime(directory)):
                    with zipfile.ZipFile(directory, "r") as zip_ref:
                        # check first on Windows systems if paths are not too long
                        if ('win32' in sys.platform):
                            longest_file_length = max(len(i) for i in zipfile.ZipFile.namelist(zip_ref))
                            if len(dir_name) + longest_file_length > 259:
                                raise IOError(2, 'Some of the files cannot be extracted ' +
                                              'from "%s". The path is too long.' % directory)
                        zip_ref.extractall(dir_name)

                    print('    archive extracted to folder', dir_name, '...')
            directory = dir_name
        else: # i.e. either directory or .tar or zipped .tar
            head, tail = os.path.split(directory[:directory.rfind('.t')])
            dir_name = os.path.join(head, genericsettings.extraction_folder_prefix + tail)
            # extract only if extracted folder does not exist yet or if it was
            # extracted earlier than last change of archive:
            if extract_files:
                if (not os.path.exists(dir_name)) or (os.path.getmtime(dir_name) < os.path.getmtime(directory)):
                    tar_file = tarfile.TarFile.open(directory)
                    longest_file_length = max(len(i) for i in tar_file.getnames())
                    if ('win32' in sys.platform) and len(dir_name) + longest_file_length > 259:
                        raise IOError(2, 'Some of the files cannot be extracted ' +
                                      'from "%s". The path is too long.' % directory)

                    try: tar_file.extractall(dir_name, filter='data')
                    except TypeError: tar_file.extractall(dir_name)  # Windows
                    # TarFile.open handles tar.gz/tgz
                    print('    archive extracted to folder', dir_name, '...')
            directory = dir_name
            # archive = tarfile.TarFile(directory)
            # for elem in archivefile.namelist():
            #    ~ if elem.endswith('.info'):
            #        ~ (root,elem) = os.path.split(elem)
            #        ~ filelist = IndexFile(root,elem,archive)

    return directory


def get_output_directory_sub_folder(args):

    directory = ''
    if not isinstance(args, (list, set, tuple)):
        directory = args.strip().rstrip(os.path.sep)

        if not os.path.isdir(directory) and is_recognized_repository_filetype(directory):
            directory = directory[:directory.find('.t')]
        directory = directory.split(':')[-1]
        directory = directory.split(os.sep)[-1].replace(genericsettings.extraction_folder_prefix, '')
    else:
        for index, argument in enumerate(args):
            if not os.path.isdir(argument) and is_recognized_repository_filetype(argument):
                argument = argument[:argument.find('.t')]
            argument = argument.split(':')[-1].split(os.sep)[-1]
            directory += (argument if len(argument) <= 5 else argument[:5]) + '_'
            if index >= 6:
                directory += 'et_al'
                break
        directory = directory.rstrip('_')

    if len(directory) == 0:
        raise ValueError(args)

    return directory

