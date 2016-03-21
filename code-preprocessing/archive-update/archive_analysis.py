# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import os

from cocoprep.archive_load_data import get_file_name_list, parse_archive_file_name, parse_problem_instance_file_name
from cocoprep.archive_load_data import get_key_value, create_path, remove_empty_file
from cocoprep.archive_exceptions import PreprocessingException, PreprocessingWarning


def summary_analysis(input_path, output_file, lower_bound, upper_bound):
    """
    Creates a summary of the analysis files from the input_path folder, which contain data in the following format:
    [evaluation_number] [objective space values] [decision space values]
    For each file records the highest values higher than the upper_bound and lowest values lower than the
    lower_bound. The output consists of two lines for each problem_id with the following format:
    [file_name] [lowest_value1] ... [lowest_valueD]
    [file_name] [highest_value1] ... [highest_valueD]
    If any of the decision space value did not go beyond one of the bounds, None is output instead of a value.
    """

    # Check whether input path exits
    input_files = get_file_name_list(input_path)
    if len(input_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(input_path))

    # Read the input files one by one and save the result in the output_file
    with open(output_file, 'a') as f_out:
        for input_file in input_files:

            try:
                (suite_name, function, instance, dimension) = parse_problem_instance_file_name(input_file)
            except PreprocessingWarning as warning:
                print('Skipping file {}\n{}'.format(input_file, warning))

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


def archive_analysis(input_path, output_path, lower_bound, upper_bound):
    """
    Records all instances from the archives found in input_path where any decision space value is lower than the
    lower_bound or higher than the upper_bound. Archives of dimensions > 5, which don't include decision space values
    are skipped. The output is saved in one file per problem and consists of lines with the following format:
    [evaluation_number] [objective space values] [decision space values]
    """

    # Check whether input path exits
    input_files = get_file_name_list(input_path)
    if len(input_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(input_path))

    lb = float(lower_bound)
    ub = float(upper_bound)

    # Read the input files one by one and save the result in the output_path
    create_path(output_path)
    for input_file in input_files:

        try:
            (suite_name, function, dimension) = parse_archive_file_name(input_file)
        except PreprocessingWarning as warning:
            print('Skipping file {}\n{}'.format(input_file, warning))

        if dimension > 5:
            continue

        print(input_file)
        column_start = 3
        column_end = 3 + dimension

        f_out = None
        f_name = ""

        with open(input_file, 'r') as f_in:
            for line in f_in:
                if line[0] == '%' and 'instance' in line:
                    if f_out and not f_out.closed:
                        f_out.close()
                        remove_empty_file(f_name)
                    instance = int(get_key_value(line[1:], 'instance').strip(' \t\n\r'))
                    f_name = os.path.join(output_path, '{}_f{:02d}_i{:02d}_d{:02d}_analysis.txt'.format(suite_name,
                                                                                                        function,
                                                                                                        instance,
                                                                                                        dimension))
                    f_out = open(f_name, 'a')
                elif len(line) == 0 or line[0] == '%' or len(line.split()) < 4:
                    continue
                else:
                    for number in line.split()[column_start:column_end]:
                        if (float(number) > ub) or (float(number) < lb):
                            string = '\t'.join(line.split()[:column_end])
                            f_out.write("{}\n".format(string))
            f_in.close()
        if f_out and not f_out.closed:
            f_out.close()
            remove_empty_file(f_name)


if __name__ == '__main__':
    """A script for analyzing the archives of solutions.
    """
    import timing

    # Set the bounds
    lower = -5
    upper = 5
    if len(sys.argv) == 2:
        lower = sys.argv[1]
        upper = sys.argv[2]

    # Analyze the archives
    archive_analysis('/Volumes/STORAGE/Data/archives/archives-output_2016_03_18_only_i01-i05',
                     '/Volumes/STORAGE/Data/archives/archive_analysis', lower, upper)

    timing.log('Finished reading data', timing.now())

    summary_analysis('/Volumes/STORAGE/Data/archives/archive_analysis',
                     '/Volumes/STORAGE/Data/archives/summary_analysis.txt',
                     lower, upper)
