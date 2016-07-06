# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

from cocoprep.archive_load_data import get_file_name_list, parse_archive_file_name, get_key_value, parse_range
from cocoprep.archive_exceptions import PreprocessingException, PreprocessingWarning


def extract_extremes(input_paths, output_file, functions, instances, dimensions):
    """
    Extracts the extreme points from the archives contained in input_paths and outputs them to the output_file in
    the following format:
    [problem_name] [extreme_point_1] [extreme_point_2]

    Assumes the two extreme points are contained in the first two lines of every instance archive. If not, that
    instance is skipped.
    Performs no kind of sorting or filtering of the problems, therefore if multiple copies of one problem are present
    in the input, multiple lines for one problem will be also present in the output.
    """

    # Check whether input paths exist
    input_files = get_file_name_list(input_paths, ".adat")
    if len(input_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(input_paths))

    # Read the input files one by one and save the result in the output_file
    with open(output_file, 'a') as f_out:
        for input_file in input_files:
            try:
                (suite_name, function, instance, dimension) = parse_archive_file_name(input_file)
                if (function not in functions) or (instance not in instances) or (dimension not in dimensions):
                    continue
            except PreprocessingWarning as warning:
                print('Skipping file {}\n{}'.format(input_file, warning))
                continue

            print(input_file)

            with open(input_file, 'r') as f_in:
                extreme1 = None
                count = 0
                for line in f_in:
                    if line[0] == '%' and 'instance' in line:
                        instance = int(get_key_value(line[1:], 'instance').strip(' \t\n\r'))
                        count = 0
                    elif count > 1 or (len(line) == 0) or line[0] == '%':
                        continue
                    elif count == 0:
                        extreme1 = line.split()[1:3]
                        count = 1
                    elif count == 1:
                        extreme2 = line.split()[1:3]
                        count = 2
                        try:
                            string = '{}_f{:02d}_i{:02d}_d{:02d}\t'.format(suite_name, function, instance, dimension)
                            string = string + '\t'.join(extreme1) + '\t' + '\t'.join(extreme2) + '\n'
                            f_out.write(string)
                        except ValueError:
                            print('Skipping instance {} in file {}'.format(instance, input_file))

                f_in.close()
                f_out.flush()
        f_out.close()


if __name__ == '__main__':
    """Extracts information on the two extreme points from the archives of solutions. Results are stored into an output
       file.
    """
    import timing

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--functions', type=parse_range, default=range(1, 56),
                        help='function numbers to be included in the processing of archives')
    parser.add_argument('-i', '--instances', type=parse_range, default=range(1, 11),
                        help='instance numbers to be included in the processing of archives')
    parser.add_argument('-d', '--dimensions', type=parse_range, default=[2, 3, 5, 10, 20, 40],
                        help='dimensions to be included in the processing of archives')
    parser.add_argument('output', help='path to the output file')
    parser.add_argument('input', default=[], nargs='+', help='path(s) to the input folder(s)')
    args = parser.parse_args()

    print('Program called with arguments: \ninput folders = {}\noutput file = {}'.format(args.input, args.output))
    print('functions = {} \ninstances = {}\ndimensions = {}\n'.format(args.functions, args.instances, args.dimensions))

    extract_extremes(args.input, args.output, args.functions, args.instances, args.dimensions)
