# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse

from cocoprep.archive_load_data import parse_range, create_path, remove_empty_file
from cocoprep.archive_load_data import get_file_name_list, parse_archive_file_name, parse_problem_instance_file_name
from cocoprep.archive_exceptions import PreprocessingException, PreprocessingWarning


def summary_analysis(input_path, output_file, lower_bound, upper_bound, functions, instances, dimensions):
    """
    Creates a summary of the analysis files from the input_path folder, which contain data in the following format:
    [evaluation_number] [objective space values] [decision space values]
    For each file records the highest values higher than the upper_bound and lowest values lower than the
    lower_bound. The output consists of two lines for each problem_id with the following format:
    [file_name] [lowest_value1] ... [lowest_valueD]
    [file_name] [highest_value1] ... [highest_valueD]
    If none of the decision space values went beyond one of the bounds, no output is done.
    """

    # Check whether input path exits
    input_files = get_file_name_list(input_path, ".txt")
    if len(input_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(input_path))

    # Read the input files one by one and save the result in the output_file
    with open(output_file, 'a') as f_out:
        for input_file in input_files:

            try:
                (suite_name, function, instance, dimension) = parse_problem_instance_file_name(input_file)
                if (function not in functions) or (instance not in instances) or (dimension not in dimensions):
                    continue
            except PreprocessingWarning as warning:
                print('Skipping file {}\n{}'.format(input_file, warning))
                continue

            print(input_file)
            column_start = 3
            column_end = 3 + dimension

            lowest = [float(lower_bound)] * dimension
            highest = [float(upper_bound)] * dimension

            with open(input_file, 'r') as f_in:
                for line in f_in:
                    for idx, number in enumerate(line.split()[column_start:column_end]):
                        num = float(number)
                        if num > highest[idx]:
                            highest[idx] = num
                        if num < lowest[idx]:
                            lowest[idx] = num
                f_in.close()

            f_out.write('{}_f{:02d}_i{:02d}_d{:02d}'.format(suite_name, function, instance, dimension))
            for number in lowest:
                f_out.write('\t{:.8E}'.format(number))
            f_out.write('\n')

            f_out.write('{}_f{:02d}_i{:02d}_d{:02d}'.format(suite_name, function, instance, dimension))
            for number in highest:
                f_out.write('\t{:.8E}'.format(number))
            f_out.write('\n')

        f_out.close()


def archive_analysis(input_paths, output_path, lower_bound, upper_bound, functions, instances, dimensions):
    """Records all instances from the archives found in input_paths where any decision space value is lower than the
       lower_bound or higher than the upper_bound. Archives of dimensions > 5, which don't include decision space values
       are skipped. The output consists of lines with the following format:
       [evaluation_number] [objective space values] [decision space values]
       Assumes one file contains one archive.
    """

    # Check whether input path exists
    input_files = get_file_name_list(input_paths, ".adat")
    if len(input_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(input_paths))

    # Read the input files one by one and save the result in the output_path
    create_path(output_path)
    for input_file in input_files:

        try:
            (suite_name, function, instance, dimension) = parse_archive_file_name(input_file)
            if not instance:
                raise PreprocessingWarning('Analysis does not work on files with multiple archives, use archive_split')
            if (function not in functions) or (instance not in instances) or (dimension not in dimensions) or \
                    (dimension > 5):
                continue
        except PreprocessingWarning as warning:
            print('Skipping file {}\n{}'.format(input_file, warning))
            continue

        print(input_file)

        column_start = 3
        column_end = 3 + dimension
        output_file = os.path.join(output_path, '{}_f{:02d}_i{:02d}_d{:02d}_analysis.txt'.format(suite_name,
                                                                                                 function,
                                                                                                 instance,
                                                                                                 dimension))
        f_out = open(output_file, 'a')

        with open(input_file, 'r') as f_in:
            for line in f_in:
                if len(line) == 0 or line[0] == '%' or len(line.split()) < 4:
                    continue
                else:
                    for number in line.split()[column_start:column_end]:
                        if (float(number) > upper_bound) or (float(number) < lower_bound):
                            string = '\t'.join(line.split()[:column_end])
                            f_out.write('{}\n'.format(string))

        f_out.close()
        remove_empty_file(output_file)


if __name__ == '__main__':
    """Performs an analysis of the archive w.r.t. the position of solutions in the decision space.

       All solutions of a problem that have at least one coordinate outside the given interval are output in a file (if
       not bounds are given as parameters, [-5, 5] is used). Finally, a summary of the analysis is performed, which
       collects the most extreme values for each coordinate and each problem.
    """
    import timing

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--functions', type=parse_range, default=range(1, 56),
                        help='function numbers to be included in the processing of archives')
    parser.add_argument('-i', '--instances', type=parse_range, default=range(1, 11),
                        help='instance numbers to be included in the processing of archives')
    parser.add_argument('-d', '--dimensions', type=parse_range, default=[2, 3, 5],
                        help='dimensions to be included in the processing of archives')
    parser.add_argument('-l', '--lower_bound', type=float, default=-5.0,
                        help='lower bound of the decision space')
    parser.add_argument('-u', '--upper_bound', type=float, default=5.0,
                        help='upper bound of the decision space')
    parser.add_argument('output', help='path to the output folder')
    parser.add_argument('summary', help='file name for the summary')
    parser.add_argument('input',  help='path to the input folder')
    args = parser.parse_args()

    print('Program called with arguments: \ninput folder = {}\noutput folder = {}'.format(args.input, args.output))
    print('summary file = {}'.format(args.summary))
    print('functions = {} \ninstances = {}\ndimensions = {}'.format(args.functions, args.instances, args.dimensions))
    print('lower bound = {} \nupper bound = {}\n'.format(args.lower_bound, args.upper_bound))

    # Analyze the archives
    archive_analysis(args.input, args.output, args.lower_bound, args.upper_bound, args.functions, args.instances,
                     args.dimensions)

    timing.log('Finished reading data', timing.now())

    summary_analysis(args.output, args.summary, args.lower_bound, args.upper_bound, args.functions, args.instances,
                     args.dimensions)
