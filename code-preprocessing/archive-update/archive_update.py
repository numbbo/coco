# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse

from cocoprep.archive_load_data import parse_range, read_best_values, write_best_values
from cocoprep.archive_functions import ArchiveInfo
from cocoprep.coco_archive import Archive, log_level


def update_best_hypervolume(old_best_files, new_best_data, new_best_file):
    """Updates the best hypervolume values. The old hypervolume values are read from old_best_files (a list of files),
       while the new ones are passed through new_best_data. The resulting best values are appended to new_best_file
       in a format that can be readily used by the COCO source code in C.
       :param old_best_files: list of files containing best hypervolumes
       :param new_best_data: dictionary with problem names and their new best hypervolumes
       :param new_best_file: name of the file to which the new values will be appended
    """
    print('Updating best hypervolume values...')

    # Read the old best values from the given files
    try:
        old_best_data = read_best_values(old_best_files)
    except IOError as err:
        print(err)
        print('Continuing nevertheless...')
        result = new_best_data
    else:
        # Create a set of problem_names contained in at least one dictionary
        problem_names = set(old_best_data.keys()).union(set(new_best_data.keys()))
        result = {}

        # Iterate over all problem names and store only the best (i.e. largest) hypervolumes
        for problem_name in problem_names:
            new_value = new_best_data.get(problem_name)
            old_value = old_best_data.get(problem_name)
            if new_value is None:
                result.update({problem_name: float(old_value)})
            elif old_value is None or (abs(float(old_value) - 1) < 1e-8):
                # New value is always better when old_value equals 1
                result.update({problem_name: float(new_value)})
            else:
                result.update({problem_name: max(float(new_value), float(old_value))})

            if new_value is not None and old_value is not None and float(new_value) > float(old_value):
                print('{} HV improved by {:.15f}'.format(problem_name, float(new_value) - float(old_value)))

    # Write the best values
    write_best_values(result, new_best_file)
    print('Done.')


def merge_archives(input_path, output_path, functions, instances, dimensions, crop_variables):
    """Merges all archives from the input_path (removes any dominated solutions) and stores the consolidated archives
       in the output_path. Returns problem names and their new best hypervolume values in the form of a dictionary.
       :param input_path: input path
       :param output_path: output path (created if not existing before)
       :param functions: functions to be included in the merging
       :param instances: instances to be included in the merging
       :param dimensions: dimensions to be included in the merging
    """
    result = {}

    print('Reading archive information...')
    archive_info = ArchiveInfo(input_path, functions, instances, dimensions)

    print('Processing archives...')
    while True:
        # Get information about the next problem instance
        problem_instance_info = archive_info.get_next_problem_instance_info()
        if problem_instance_info is None:
            break

        old_level = log_level('warning')

        # Create an archive for this problem instance
        archive = Archive(problem_instance_info.suite_name, problem_instance_info.function,
                          problem_instance_info.instance, problem_instance_info.dimension)

        # Read the solutions from the files and add them to the archive
        problem_instance_info.fill_archive(archive)

        # Write the non-dominated solutions into output folder
        problem_instance_info.write_archive_solutions(output_path, archive, crop_variables)

        result.update({str(problem_instance_info): archive.hypervolume})
        print('{}: {:.15f}'.format(problem_instance_info, archive.hypervolume))

        log_level(old_level)

    return result


if __name__ == '__main__':
    """Updates the archives of solutions to bi-objective problems.

       Input archives are read and merged so that the two extreme solutions and all non-dominated solutions are stored
       in the output archives. A file with the best known hypervolume values is generated from these hypervolumes and
       the ones stored in C source files (use --merge-only if you wish to do the merging without the update of
       hypervolume values and --crop-variables if you want to keep only the objective values).
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
    parser.add_argument('--crop-variables', action='store_true',
                        help='don\'t include information on the variables in the output archives')
    parser.add_argument('--hyp-file', default='new_best_values_hyp.c',
                        help='name of the file to store new hypervolume values')
    parser.add_argument('output', help='path to the output folder')
    parser.add_argument('input', default=[], nargs='+', help='path(s) to the input folder(s)')
    args = parser.parse_args()

    print('Program called with arguments: \ninput folders = {}\noutput folder = {}'.format(args.input, args.output))
    print('functions = {} \ninstances = {}\ndimensions = {}\n'.format(args.functions, args.instances, args.dimensions))

    # Merge the archives
    new_hypervolumes = merge_archives(args.input, args.output, args.functions, args.instances, args.dimensions,
                                      args.crop_variables)

    timing.log('Finished merging', timing.now())

    # Use files with best hypervolume values from the src folder and update them with the new best values
    if not args.merge_only:
        base_path = os.path.dirname(__file__)
        file_names = ['suite_biobj_best_values_hyp.c']
        file_names = [os.path.abspath(os.path.join(base_path, '..', '..', 'code-experiments/src', file_name))
                      for file_name in file_names]
        update_best_hypervolume(file_names, new_hypervolumes, os.path.join(args.output, '..', args.hyp_file))
