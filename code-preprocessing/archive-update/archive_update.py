# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
from cocoprep.archive_load_data import parse_range
from cocoprep.archive_functions import merge_archives, update_best_hypervolume

if __name__ == '__main__':
    """Updates the archives of solutions to bi-objective problems.

       Input archives are read and merged so that the two extreme solutions and all non-dominated solutions are stored
       in the output archives. A file with the best known hypervolume values is generated from these hypervolumes and
       the ones stored in C source files (use --merge-only if you wish to do the merging without the update of
       hypervolume values).
    """
    import timing

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--functions', type=parse_range, default=range(1, 56),
                        help='function numbers to be included in the processing of archives')
    parser.add_argument('-i', '--instances', type=parse_range, default=range(1, 11),
                        help='instance numbers to be included in the processing of archives')
    parser.add_argument('-d', '--dimensions', type=parse_range, default=[2, 3, 5, 10, 20, 40],
                        help='dimensions to be included in the processing of archives')
    parser.add_argument('--merge-only', action='store_true',
                        help='perform only merging of archives, do not update hypervolume values')
    parser.add_argument('output', help='path to the output folder')
    parser.add_argument('input', default=[], nargs='+', help='path(s) to the input folder(s)')
    args = parser.parse_args()

    print('Program called with arguments: \ninput folders = {}\noutput folder = {}'.format(args.input, args.output))
    print('functions = {} \ninstances = {}\ndimensions = {}\n'.format(args.functions, args.instances, args.dimensions))

    # Merge the archives
    new_hypervolumes = merge_archives(args.input, args.output, args.functions, args.instances, args.dimensions)

    timing.log('Finished merging', timing.now())

    # Use files with best hypervolume values from the src folder and update them with the new best values
    if not args.merge_only:
        base_path = os.path.dirname(__file__)
        file_names = ['suite_biobj_best_values_hyp.c']
        file_names = [os.path.abspath(os.path.join(base_path, '..', '..', 'code-experiments/src', file_name))
                      for file_name in file_names]
        update_best_hypervolume(file_names, new_hypervolumes, os.path.join(args.output, '..', "new_best_values_hyp.c"))
