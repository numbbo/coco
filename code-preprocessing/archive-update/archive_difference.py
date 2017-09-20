# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import difflib
import argparse

from cocoprep.archive_load_data import get_file_name_list, parse_archive_file_name, parse_range
from cocoprep.archive_exceptions import PreprocessingException, PreprocessingWarning


def archive_difference(first_path, second_path, differences, functions, instances, dimensions):
    """Outputs the differences between the matching archive files found in the first and second path.
    """
    # Check whether first paths exist
    first_files = get_file_name_list(first_path, ".adat")
    if len(first_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(first_path))

    for i, first_file in enumerate(first_files):
        try:
            (suite_name, function, instance, dimension) = parse_archive_file_name(first_file)
            if (function not in functions) or (dimension not in dimensions):
                continue
            if not instance:
                raise PreprocessingWarning('Checking for differences does not work on files with multiple archives, '
                                           'use archive_split')
            if instance not in instances:
                continue
        except PreprocessingWarning as warning:
            print('Skipping file {}\n{}'.format(first_file, warning))
            first_files[i] = ''
            continue
        print(first_file)

    # Check whether second paths exist
    second_files = get_file_name_list(second_path, ".adat")
    if len(second_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(second_path))

    for i, second_file in enumerate(second_files):
        try:
            (suite_name, function, instance, dimension) = parse_archive_file_name(second_file)
            if (function not in functions) or (dimension not in dimensions):
                continue
            if not instance:
                raise PreprocessingWarning('Checking for differences does not work on files with multiple archives, '
                                           'use archive_split')
            if instance not in instances:
                continue
        except PreprocessingWarning as warning:
            print('Skipping file {}\n{}'.format(second_file, warning))
            second_files[i] = ''
            continue
        print(second_file)

    with open(differences, 'a') as f_out:
        for first_file in first_files:
            if first_file != '':
                file_name = os.path.basename(first_file)
                if file_name in [os.path.basename(second_file) for second_file in second_files]:
                    second_file = os.path.join(second_path, file_name)
                    with open(first_file, 'r') as f1:
                        with open(second_file, 'r') as f2:
                            # Find and output the differences
                            diff = difflib.unified_diff(f1.readlines(), f2.readlines(), fromfile='f1', tofile='f2')
                            f_out.write('{}\n'.format(file_name))
                            print(file_name)
                            for line in diff:
                                f_out.write(line)
                        f2.close()
                    f1.close()
        f_out.close()


if __name__ == '__main__':
    """Checks for differences in two archive files of the same name.
    """
    import timing

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--functions', type=parse_range, default=range(1, 56),
                        help='function numbers to be included in the processing of archives')
    parser.add_argument('-i', '--instances', type=parse_range, default=range(1, 11),
                        help='instance numbers to be included in the processing of archives')
    parser.add_argument('-d', '--dimensions', type=parse_range, default=[2, 3, 5, 10, 20, 40],
                        help='dimensions to be included in the processing of archives')
    parser.add_argument('first', help='path to the folder with the first archives')
    parser.add_argument('second', help='path to the folder with the second archives')
    parser.add_argument('differences', help='name of the file with the differences')
    args = parser.parse_args()

    print('Program called with arguments: \nfirst = {}\nsecond = {}\ndifferences = {}'.format(args.first, args.second,
                                                                                              args.differences))
    print('functions = {} \ninstances = {}\ndimensions = {}'.format(args.functions, args.instances, args.dimensions))

    # Analyze the archives
    archive_difference(args.first, args.second, args.differences, args.functions, args.instances, args.dimensions)

