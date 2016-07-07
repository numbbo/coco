# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import argparse
import fileinput

from cocoprep.archive_exceptions import PreprocessingException, PreprocessingWarning
from cocoprep.archive_load_data import parse_archive_file_name, parse_range, get_key_value, get_file_name_list


def parse_info_file(file_name):
    """Returns a list of quadruples [function, instance, dimension, evaluations] read from the .info file with the
       given name.
    """
    info_data_list = []
    with open(file_name, 'r') as f:
        for line in f:
            instances = []
            evaluations = []
            if 'function' in line:
                function = int(get_key_value(line, 'function'))
                dimension = int(get_key_value(line, 'dim'))
                for element in line.split(','):
                    if ':' in element:
                        replaced = element.replace('|', ':')
                        instances.append(int(replaced.split(':')[0]))
                        evaluations.append(int(replaced.split(':')[1]))
                for index, instance in enumerate(instances):
                    info_data_item = [function, instance, dimension, evaluations[index]]
                    info_data_list.append(info_data_item)
        f.close()
    return info_data_list


def check_file_complete(input_paths, functions, instances, dimensions, max_diff=1000):
    """Checks the .adat files created by the bbob-biobj logger to see if they have been properly written. Outputs the
       difference between the last evaluation from the .adat file and the one noted in the .info file if they are
       greater than max_diff.

       Takes into account only the given functions, instances and dimensions.
    """

    def inspect_line(input_file, line_string, evaluations, max_diff=1e5):
        """Check that the line_string contains at least three numbers and that they are correctly written. Outputs a
           message if the difference between the evaluations and the first number in the line_string is grater than
           max_diff.
        """
        num_items = len(line_string.split())
        if num_items < 3:
            print("File {}, line {} too short".format(input_file, line_string))
        for i in range(num_items):
            try:
                float(line_string.split()[i])
            except ValueError:
                print('File {}, line {}, number {} incorrect'.format(input_file, line_string, line_string.split()[i]))
                continue

        if evaluations - int(line_string.split()[0]) > max_diff:
            print('Mismatch in evaluations in file {}\n'
                  '.info  = {}\n'
                  '.adat  = {}\n'
                  ' diff  = {}\n'.format(input_file, evaluations, line_string.split()[0],
                                         evaluations - int(line_string.split()[0])))

    # Check whether .info and .adat files exist in the input paths
    info_files = get_file_name_list(input_paths, ".info")
    if len(info_files) == 0:
        raise PreprocessingException('Folder {} does not contain .info files'.format(input_paths))

    adat_files = get_file_name_list(input_paths, ".adat")
    if len(adat_files) == 0:
        raise PreprocessingException('Folder {} does not contain .adat files'.format(input_paths))

    info_dict = {}
    print('Reading .info files...')
    for input_file in info_files:
        # Store the data from the .info files
        try:
            info_data_list = parse_info_file(input_file)
        except ValueError as error:
            raise PreprocessingException('Cannot read file {}\n{}'.format(input_file, error))

        for info_data_item in info_data_list:
            (function, instance, dimension, evaluations) = info_data_item
            if (function not in functions) or (instance not in instances) or (dimension not in dimensions):
                continue
            info_dict[(function, instance, dimension)] = evaluations

    print('Reading .adat files...')
    for input_file in adat_files:
        try:
            (suite_name, function, instance, dimension) = parse_archive_file_name(input_file)
            if (function not in functions) or (instance and instance not in instances) or \
                    (dimension not in dimensions):
                continue
        except PreprocessingWarning as warning:
            print('Skipping file {}\n{}'.format(input_file, warning))
            continue

        with open(input_file, 'r') as f:

            instance_found = False
            last_line = None

            for line in f:
                if not line.strip() or (line[0] == '%' and 'instance' not in line):
                    # Ignore empty lines and lines with comments
                    continue

                elif line[0] == '%' and 'instance' in line:
                    if last_line:
                        inspect_line(input_file, last_line, info_dict[(function, instance, dimension)])
                    instance = int(get_key_value(line[1:], 'instance'))
                    instance_found = (instance in instances)

                elif instance_found and line[0] != '%':
                    last_line = line

            if instance_found:
                inspect_line(input_file, last_line, info_dict[(function, instance, dimension)])
            f.close()


def evaluations_append(input_paths, functions, instances, dimensions, fast=False):
    """Appends the comment `% evaluations = NUMBER` to the end of every instance in the .adat files created by the
       bbob-biobj logger.

       If fast is True, it assumes the file contains only one instance (the instance is read from the file contents,
       not the file name) and appends the comment only once - at the end of the file. No check whether this should be
       done is performed - the user should know when it is safe to choose this option.

       The NUMBER is retrieved from the corresponding .info file.
       Takes into account only the given functions, instances and dimensions.
    """

    # Check whether .info and .adat files exist in the input paths
    info_files = get_file_name_list(input_paths, ".info")
    if len(info_files) == 0:
        raise PreprocessingException('Folder {} does not contain .info files'.format(input_paths))

    adat_files = get_file_name_list(input_paths, ".adat")
    if len(adat_files) == 0:
        raise PreprocessingException('Folder {} does not contain .adat files'.format(input_paths))

    info_dict = {}
    for input_file in info_files:
        try:
            info_data_list = parse_info_file(input_file)
        except ValueError as error:
            raise PreprocessingException('Cannot read file {}\n{}'.format(input_file, error))

        for info_data_item in info_data_list:
            (function, instance, dimension, evaluations) = info_data_item
            if (function not in functions) or (instance not in instances) or (dimension not in dimensions):
                continue
            info_dict[(function, instance, dimension)] = evaluations

    for input_file in adat_files:
        try:
            (suite_name, function, instance, dimension) = parse_archive_file_name(input_file)
            if (function not in functions) or (instance and instance not in instances) or \
                    (dimension not in dimensions):
                continue
        except PreprocessingWarning as warning:
            print('Skipping file {}\n{}'.format(input_file, warning))
            continue

        try:
            if instance or fast:
                # Assumes only one instance is contained in the file
                with open(input_file, 'r') as f:
                    for line in f:
                        if (line[0] == '%') and ('instance' in line):
                            instance = int(get_key_value(line[1:], 'instance'))
                            break
                    f.close()
                with open(input_file, 'a') as f:
                    f.write('% evaluations = {}'.format(info_dict[(function, instance, dimension)]))
                    f.close()

            else:
                first_instance = True
                # Take care of the non-last instances in the file
                for line in fileinput.input(input_file, inplace=True):
                    if (line[0] == '%') and ('instance' in line):
                        instance = int(get_key_value(line[1:], 'instance'))
                        if first_instance:
                            first_instance = False
                        else:
                            sys.stdout.write('% evaluations = {}\n'.format(info_dict[(function, instance, dimension)]))
                    sys.stdout.write(line)
                fileinput.close()

                # Take care of the last instance in the file
                with open(input_file, 'a') as f:
                    f.write('% evaluations = {}'.format(info_dict[(function, instance, dimension)]))
                    f.close()

        except KeyError as error:
            print('Encountered problem in file {}\n{}'.format(input_file, error))
            fileinput.close()
            continue

if __name__ == '__main__':
    """Appends the comment `% evaluations = NUMBER` to the end of every instance in the algorithm archives.

       The input folders should include .info files for all corresponding .adat files.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--functions', type=parse_range, default=range(1, 56),
                        help='function numbers to be included in the processing of archives')
    parser.add_argument('-i', '--instances', type=parse_range, default=range(1, 11),
                        help='instance numbers to be included in the processing of archives')
    parser.add_argument('-d', '--dimensions', type=parse_range, default=[2, 3, 5, 10, 20, 40],
                        help='dimensions to be included in the processing of archives')
    parser.add_argument('--fast', action='store_true',
                        help='fast option that assumes all archive files contain only one instance')
    parser.add_argument('input', default=[], nargs='+', help='path(s) to the input folder(s)')
    args = parser.parse_args()

    print('Program called with arguments: \ninput folders = {}\nfast = {}'.format(args.input, args.fast))
    print('functions = {} \ninstances = {}\ndimensions = {}\n'.format(args.functions, args.instances, args.dimensions))

    evaluations_append(args.input, args.functions, args.instances, args.dimensions, args.fast)
    #check_file_complete(args.input, args.functions, args.instances, args.dimensions)
