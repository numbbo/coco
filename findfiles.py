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

def is_recognized_repository_filetype(filename): 
    return os.path.isdir(filename.strip()) or filename.find('.tar') > 0 or filename.find('.tgz') > 0

def main(directory='.', verbose=True):
    """Lists "data" files recursively in a given directory, tar files
    are extracted. 

    The "data" files have :file:`info` and :file:`pickle` extensions.

    TODO: not only recognize .tar and .tar.gz and .tgz but .zip...
    
    """
    
    filelist = list()
    directory = directory.strip()
    
    #~ if directory.endswith('.zip'):
        #~ archive = zipfile.ZipFile(directory)
        #~ for elem in archive.namelist():
            #~ if elem.endswith('.info'):
                #~ (root,elem) = os.path.split(elem)
                #~ filelist = IndexFile(root,elem,archive)
    if not os.path.isdir(directory) and is_recognized_repository_filetype(directory):
        import tarfile
        dirname = '_extracted_' + directory[:directory.find('.t')]
        tarfile.TarFile.open(directory).extractall(dirname)
        # TarFile.open handles tar.gz/tgz
        directory = dirname
        print '    archive extracted to folder', directory, '...'
        # archive = tarfile.TarFile(directory)
        # for elem in archivefile.namelist():
        #    ~ if elem.endswith('.info'):
        #        ~ (root,elem) = os.path.split(elem)
        #        ~ filelist = IndexFile(root,elem,archive)
    #~ else:

    # Search through the directory directory and all its subfolders.
    for root, _dirs, files in os.walk(directory):
        if verbose:
            print 'Searching in %s ...' % root

        for elem in files:
            if elem.endswith('.info') or elem.endswith('.pickle') or elem.endswith('.pickle.gz'):
                filelist.append(os.path.join(root, elem))

    if verbose:
        print 'Found %d file(s).' % (len(filelist))
    if not filelist:
        warnings.warn('Could not find any file of interest in %s!' % root)
    return filelist

if __name__ == '__main__':
    main()

