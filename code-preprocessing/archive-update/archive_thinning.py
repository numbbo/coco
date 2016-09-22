# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse

from cocoprep.archive_load_data import get_file_name_list, parse_archive_file_name
from cocoprep.archive_load_data import create_path, parse_range
from cocoprep.archive_exceptions import PreprocessingException, PreprocessingWarning
from cocoprep.coco_archive import Archive, log_level


def archive_thinning(input_path, output_path, thinning_precision, currently_nondominated, functions, instances,
                     dimensions):
    """Performs thinning of all the archives in the input path and stores the thinned archives in the output path.
       Assumes one file contains one archive.

       For each archive, all input solutions are rounded according to the thinning precision (in the normalized
       objective space) and added to the thinned archive. If currently_nondominated is True, all solutions that
       are currently nondominated within the thinned archive are output. The two extreme solutions are not output.
       If currently_nondominated is False, only the solutions that are contained in the final archive are output.
       In this case, the two extreme solutions are also output.
    """
    # Check whether input path exists
    input_files = get_file_name_list(input_path, ".adat")
    if len(input_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(input_path))

    old_level = log_level('warning')

    for input_file in input_files:
        try:
            (suite_name, function, instance, dimension) = parse_archive_file_name(input_file)
            if (function not in functions) or (dimension not in dimensions):
                continue
            if not instance:
                raise PreprocessingWarning('Thinning does not work on files with multiple archives, use archive_split')
            if instance not in instances:
                continue
        except PreprocessingWarning as warning:
            print('Skipping file {}\n{}'.format(input_file, warning))
            continue

        print(input_file)

        output_file = input_file.replace(input_path, output_path)
        create_path(os.path.dirname(output_file))
        f_out = open(output_file, 'w')
        thinned_archive = Archive(suite_name, function, instance, dimension)
        thinned_solutions = 0
        all_solutions = 0

        extreme1_text = thinned_archive.get_next_solution_text()
        extreme2_text = thinned_archive.get_next_solution_text()
        extreme1 = [float(x) for x in extreme1_text.split()[1:3]]
        extreme2 = [float(x) for x in extreme2_text.split()[1:3]]
        ideal = [min(x, y) for x, y in zip(extreme1, extreme2)]
        nadir = [max(x, y) for x, y in zip(extreme1, extreme2)]
        normalization = [x - y for x, y in zip(nadir, ideal)]

        with open(input_file, 'r') as f_in:
            for line in f_in:

                if line[0] == '%':
                    f_out.write(line)

                elif len(line) == 0 or len(line.split()) < 3:
                    continue

                elif line.split()[0] == '0':
                    # The line contains an extreme solution, do nothing
                    all_solutions += 1
                    continue

                else:
                    # The line contains a 'regular' solution
                    try:
                        # Fill the archives with the rounded solutions values wrt the different precisions
                        f_original = [float(x) for x in line.split()[1:3]]
                        f_normalized = [(f_original[i] - ideal[i]) / normalization[i] for i in range(2)]
                        f_normalized = [round(f_normalized[i] / thinning_precision) for i in range(2)]
                        f_normalized = [ideal[i] + f_normalized[i] * thinning_precision for i in range(2)]
                        updated = thinned_archive.add_solution(f_normalized[0], f_normalized[1], line)
                    except IndexError:
                        print('Problem in file {}, line {}, skipping line'.format(input_file, line))
                        continue
                    finally:
                        all_solutions += 1

                    if currently_nondominated and (updated == 1):
                        thinned_solutions += 1
                        f_out.write(line)

        if not currently_nondominated and (thinned_archive.number_of_solutions == 2):
            # Output the two extreme solutions if they are the only two in the archive
            f_out.write(extreme1_text)
            f_out.write(extreme2_text)
            thinned_solutions = 2

        while not currently_nondominated:
            text = thinned_archive.get_next_solution_text()
            if text is None:
                break
            thinned_solutions += 1
            f_out.write(text)

        print('original: {} thinned: {} ({:.2f}%)'.format(all_solutions, thinned_solutions,
                                                          100 * thinned_solutions / all_solutions))
        f_out.close()

    log_level(old_level)


if __name__ == '__main__':
    """Performs thinning of archives w.r.t. the given precision (intended to use with already updated archives, not the
       'basic' archives returned by an algorithm).

       Important: Because the new archives always contain the two extreme solutions, any solutions outside the region
       of interest in the objective space will be dominated and therefore ignored.
    """
    import timing

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--functions', type=parse_range, default=range(1, 56),
                        help='function numbers to be included in the processing of archives')
    parser.add_argument('-i', '--instances', type=parse_range, default=range(1, 11),
                        help='instance numbers to be included in the processing of archives')
    parser.add_argument('-d', '--dimensions', type=parse_range, default=[2, 3, 5, 10, 20, 40],
                        help='dimensions to be included in the processing of archives')
    parser.add_argument('-p', '--precision', type=float, default=1e-6,
                        help='thinning precision')
    parser.add_argument('--currently-nondominated', action='store_true',
                        help='output currently nondominated solutions')
    parser.add_argument('output', help='path to the output folder')
    parser.add_argument('input', help='path to the input folder')
    args = parser.parse_args()

    print('Program called with arguments: \ninput folder = {}\noutput folder = {}'.format(args.input, args.output))
    print('functions = {} \ninstances = {}\ndimensions = {}'.format(args.functions, args.instances, args.dimensions))
    print('precision = {} \ncurrently-nondominated = {}\n'.format(args.precision, args.currently_nondominated))

    # Analyze the archives
    archive_thinning(args.input, args.output, args.precision, args.currently_nondominated, args.functions,
                     args.instances, args.dimensions)

