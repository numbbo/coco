#! /usr/bin/env python

# Script to find all *.info files within a directory
# and its subfolders.

import os
import warnings
#import zipfile
#import tarfile

# Initialization
def main(directory=os.getcwd(), verbose=True):
    """Lists *.info files by browsing recursively a given directory."""

    indexFiles = list()

    #~ if directory.endswith('.zip'):
        #~ archive = zipfile.ZipFile(directory)
        #~ for elem in archive.namelist():
            #~ if elem.endswith('.info'):
                #~ (root,elem) = os.path.split(elem)
                #~ indexFile = IndexFile(root,elem,archive)
    #~ if directory.find('.tar') != -1:
        #~ archive = tarfile.TarFile(directory)
        #~ for elem in archivefile.namelist():
            #~ if elem.endswith('.info'):
                #~ (root,elem) = os.path.split(elem)
                #~ indexFile = IndexFile(root,elem,archive)
    #~ else:
    # Search through the directory directory and all its subfolders.
    # Store all *.info files within list indexFiles. All entries will
    # be instances of the class IndexFile.
    for root, dirs, files in os.walk(directory):
        if verbose:
            print 'Searching in %s ...' % root

        previousLength = len(indexFiles)    # for nice output
        for elem in files:
            if elem.endswith('.info'):
                #indexFile = IndexFile(root,elem)
                indexFiles.append(os.path.join(root, elem))

        # Print success message.
        #if verbose:
        #    if len(indexFiles) - previousLength > 0:
        #        print 'Found %d file(s)!' % (len(indexFiles) - previousLength)
            #else:
                #warnings.warn('Could not find any index file in %s!' % root)
    if verbose:
        print 'Found %d file(s)!' % (len(indexFiles))
    if not indexFiles:
        warnings.warn('Could not find any index file in %s!' % root)
    return indexFiles

if __name__ == '__main__':
    main()
