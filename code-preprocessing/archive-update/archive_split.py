# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse

from cocoprep.archive_load_data import parse_archive_file_name, parse_range
from cocoprep.archive_load_data import create_path, get_key_value, get_file_name_list
from cocoprep.archive_exceptions import PreprocessingException, PreprocessingWarning


def archive_split(input_paths, output_path, functions, instances, dimensions):
    """Iterates through all files in input_paths and splits those that contain multiple instances to one file per
       instance. The check for multiple instances is done only through file names.
    """

    # Check whether input paths exist
    input_files = get_file_name_list(input_paths, ".adat")
    if len(input_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(input_paths))

    # Read the input files one by one and save the result in the output_path
    create_path(output_path)
    for input_file in input_files:

        try:
            (suite_name, function, instance, dimension) = parse_archive_file_name(input_file)
            if (function not in functions) or instance or (dimension not in dimensions):
                continue

        except PreprocessingWarning as warning:
            print('Skipping file {}\n{}'.format(input_file, warning))
            continue

        print(input_file)
        f_out = None
        instance = None

        with open(input_file, 'r') as f_in:

            buffered_lines = ''

            for line in f_in:
                if not line.strip():
                    # Ignore empty lines
                    continue

                elif line[0] == '%':
                    if 'instance' in line:
                        if f_out and not f_out.closed:
                            if len(buffered_lines) > 0:
                                f_out.write(buffered_lines)
                                buffered_lines = ''
                            f_out.close()
                        instance = int(get_key_value(line[1:], 'instance'))
                        if instance in instances:
                            output_file = os.path.join(output_path,
                                                       '{}_f{:02d}_i{:02d}_d{:02d}_nondominated.adat'.format(suite_name,
                                                                                                             function,
                                                                                                             instance,
                                                                                                             dimension))
                            f_out = open(output_file, 'w')
                        else:
                            instance = None

                    if instance:
                        buffered_lines += line

                elif (line[0] != '%') and instance:
                    if len(buffered_lines) > 0:
                        f_out.write(buffered_lines)
                        buffered_lines = ''
                    f_out.write(line)

            f_in.close()

        if f_out and not f_out.closed:
            if len(buffered_lines) > 0:
                f_out.write(buffered_lines)
            f_out.close()


if __name__ == '__main__':
    """Splits the archive that contains multiple instances to one instance per file.
    """
    import timing

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--functions', type=parse_range, default=range(1, 56),
                        help='function numbers to be included in the processing of archives')
    parser.add_argument('-i', '--instances', type=parse_range, default=range(1, 11),
                        help='instance numbers to be included in the processing of archives')
    parser.add_argument('-d', '--dimensions', type=parse_range, default=[2, 3, 5, 10, 20, 40],
                        help='dimensions to be included in the processing of archives')
    parser.add_argument('output', help='path to the output folder')
    parser.add_argument('input', default=[], nargs='+', help='path(s) to the input folder(s)')
    args = parser.parse_args()

    print('Program called with arguments: \ninput folders = {}\noutput folder = {}'.format(args.input, args.output))
    print('functions = {} \ninstances = {}\ndimensions = {}'.format(args.functions, args.instances, args.dimensions))

    # Analyze the archives
    archive_split(args.input, args.output, args.functions, args.instances, args.dimensions)

