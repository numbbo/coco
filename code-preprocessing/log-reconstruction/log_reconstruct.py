# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import numpy as np

from cocoprep.archive_exceptions import PreprocessingWarning
from cocoprep.archive_load_data import parse_archive_file_name, parse_range, get_key_value
from cocoprep.archive_functions import ArchiveInfo
from cocoex import Suite, Observer


def log_reconstruct(input_path, output_path, algorithm_name, algorithm_info, functions, instances, dimensions):
    """Reconstructs the .info, .dat and .tdat files produced by the logger from the .adat files in the input_path.

       Takes into account only the given functions, instances and dimensions. If any .info, .dat and .tdat files of
       the same names already exist in the output_path, the new data is appended to them.
    """
    suite_name = 'bbob-biobj'

    print('Reading archive information...')
    archive_info = ArchiveInfo(input_path, functions, instances, dimensions, False)

    function_string = archive_info.get_function_string()
    instance_string = archive_info.get_instance_string()
    dimension_string = archive_info.get_dimension_string()
    file_name_set = archive_info.get_file_name_set()

    print('Initializing the suite and observer...')
    suite_instance = 'instances: {}'.format(instance_string)
    suite_options = 'dimensions: {} function_indices: {}'.format(dimension_string, function_string)
    suite = Suite(suite_name, suite_instance, suite_options)
    observer_options = 'result_folder: {} algorithm_name: {} algorithm_info: "{}" log_nondominated: read'. \
        format(output_path, algorithm_name, algorithm_info)
    observer = Observer(suite_name, observer_options)

    print('Reconstructing...')
    for input_file in file_name_set:

        (_suite_name, function, _instance, dimension) = parse_archive_file_name(input_file)

        with open(input_file, 'r') as f_in:
            print(input_file)

            problem = None
            objective_vector = None
            evaluation_found = False
            instance = None
            count_not_updated = 0
            evaluation = 0

            for line in f_in:

                if len(line.split()) < 3:
                    continue

                elif line[0] == '%' and 'instance' in line:
                    instance = int(get_key_value(line[1:], 'instance'))
                    if instance in instances:
                        if problem is not None:
                            if not evaluation_found:
                                raise PreprocessingWarning('Missing the line `% evaluations = ` in the previous '
                                                           'problem. This problem is file = {}, instance = {}'
                                                           .format(input_file, instance))
                            if count_not_updated > 0:
                                print('{} solutions did not update the archive'.format(count_not_updated))
                            problem.free()
                        problem = suite.get_problem_by_function_dimension_instance(function, dimension, instance,
                                                                                   observer)
                        evaluation_found = False

                elif line[0] != '%' and instance in instances:
                    try:
                        split = line.split()
                        evaluation = int(split[0])
                        objective_vector = np.array(split[1:3])
                        updated = problem.logger_biobj_feed_solution(evaluation, objective_vector)
                        if updated == 0:
                            count_not_updated += 1
                    except ValueError as error:
                        print('Problem in file {}, line {}, skipping line\n{}'.format(input_file, line, error))
                        continue

                elif line[0] == '%' and 'evaluations' in line:
                    old_evaluation = evaluation
                    evaluation = int(get_key_value(line[1:], 'evaluations'))
                    evaluation_found = True
                    if (evaluation > old_evaluation) and problem is not None and objective_vector is not None:
                        problem.logger_biobj_feed_solution(evaluation, objective_vector)

            if problem is not None:
                if not evaluation_found:
                    print('Missing the line `% evaluations = ` in this or the previous problem. This is file = {}, '
                          'instance = {}' .format(input_file, instance))
                if count_not_updated > 0:
                    print('{} solutions did not update the archive'.format(count_not_updated))
                problem.free()

            f_in.close()


if __name__ == '__main__':
    """Reconstructs the bi-objective logger output from the archive.

       Reconstructs the .info, .dat and .tdat files from the given .adat files.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--functions', type=parse_range, default=range(1, 56),
                        help='function numbers to be included in the processing of archives')
    parser.add_argument('-i', '--instances', type=parse_range, default=range(1, 11),
                        help='instance numbers to be included in the processing of archives')
    parser.add_argument('-d', '--dimensions', type=parse_range, default=[2, 3, 5, 10, 20, 40],
                        help='dimensions to be included in the processing of archives')
    parser.add_argument('-a', '--algorithm-info', default='', help='algorithm information')
    parser.add_argument('output', help='path to the output folder')
    parser.add_argument('input', help='path to the input folder')
    parser.add_argument('algorithm_name', help='algorithm name')
    args = parser.parse_args()

    print('Program called with arguments: \ninput folder = {}\noutput folder = {}'.format(args.input, args.output))
    print('functions = {} \ninstances = {}\ndimensions = {}'.format(args.functions, args.instances, args.dimensions))
    print('alg_name = {} \nalg_info = {}\n'.format(args.algorithm_name, args.algorithm_info))

    log_reconstruct(args.input, args.output, args.algorithm_name, args.algorithm_info, args.functions, args.instances,
                    args.dimensions)
