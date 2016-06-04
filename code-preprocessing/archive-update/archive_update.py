# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

if __name__ == '__main__':
    """A script for updating the archives of solutions to bi-objective problems.

       Input archives are read and merged so that only non-dominated solutions are stored in the output archives. A
       file with the best known hypervolume values is generated from these hypervolumes and the ones stored in C source
       files.
    """
    from cocoprep import merge_archives, update_best_hypervolume
    import timing

    if len(sys.argv) > 1:
        arguments = sys.argv[1]
    else:
        arguments = '1-10'

    # Merge the archives
    input_folders = '/Volumes/STORAGE/Data/archives/archives-input'
    output_folder = '/Volumes/STORAGE/Data/archives/archives-output'
    new_hypervolumes = merge_archives(input_folders, output_folder, arguments)

    timing.log('Finished merging', timing.now())

    # Use files with best hypervolume values from the src folder and update them with the new best values
    base_path = os.path.dirname(__file__)
    file_names = ['suite_biobj_best_values_hyp.c']
    file_names = [os.path.abspath(os.path.join(base_path, '..', '..', 'code-experiments/src', file_name))
                  for file_name in file_names]
    update_best_hypervolume(file_names, new_hypervolumes, '/Volumes/STORAGE/Data/archives/new_best_values_hyp.c')

