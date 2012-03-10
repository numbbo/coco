#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Recursively find :file:`info` and :file:`pickle` files within a directory

This module can be called from the shell, it will recursively look for
:file:`info` and :file:`pickle` files in the current working directory::

  $ python pathtococo/bbob_pproc/findfiles.py
  Searching in ...
  Found ... file(s)!

"""

import os
import warnings
#import zipfile
#import tarfile

# Initialization
def main(directory='.', verbose=True):
    """Lists data files recursively in a given directory.

    The data files have :file:`info` and :file:`pickle` extensions.

    """
    
    # TODO: if directory is a zipped file, unzip first

    filelist = list()

    #~ if directory.endswith('.zip'):
        #~ archive = zipfile.ZipFile(directory)
        #~ for elem in archive.namelist():
            #~ if elem.endswith('.info'):
                #~ (root,elem) = os.path.split(elem)
                #~ filelist = IndexFile(root,elem,archive)
    #~ if directory.find('.tar') != -1:
        #~ archive = tarfile.TarFile(directory)
        #~ for elem in archivefile.namelist():
            #~ if elem.endswith('.info'):
                #~ (root,elem) = os.path.split(elem)
                #~ filelist = IndexFile(root,elem,archive)
    #~ else:

    # Search through the directory directory and all its subfolders.
    for root, dirs, files in os.walk(directory):
        if verbose:
            print 'Searching in %s ...' % root

        for elem in files:
            if elem.endswith('.info') or elem.endswith('.pickle'):
                filelist.append(os.path.join(root, elem))

    if verbose:
        print 'Found %d file(s)!' % (len(filelist))
    if not filelist:
        warnings.warn('Could not find any file of interest in %s!' % root)
    return filelist

if __name__ == '__main__':
    main()

