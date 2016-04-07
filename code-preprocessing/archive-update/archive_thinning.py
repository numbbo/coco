# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os

from cocoprep.archive_load_data import get_file_name_list, parse_archive_file_name
from cocoprep.archive_load_data import get_key_value, create_path
from cocoprep.archive_exceptions import PreprocessingException, PreprocessingWarning
from cocoprep.archive_functions import Archive, log_level


def finalize_output(precision_list, all_solutions, currently_nondominated=True):
    """Finalizes all output files contained in precision_list.

       Outputs some final statistics and closes the files. If currently_nondominated is false, outputs all the solutions
       in the thinned archives (including the extreme ones).
    """
    for precision_dict in precision_list:
        f_out = precision_dict.get('f_out')
        if f_out and not f_out.closed:
            while not currently_nondominated:
                text = precision_dict.get('archive').get_next_solution_text()
                if text is None:
                    break
                precision_dict['thinned_solutions'] += 1
                f_out.write(text)
            print('{} all: {} thinned: {} ({:.2f}%)'.format(precision_dict.get('name')[1:],
                                                            all_solutions,
                                                            precision_dict.get('thinned_solutions'),
                                                            100 * precision_dict.get(
                                                               'thinned_solutions') / all_solutions))
            f_out.close()


def archive_thinning(input_path, output_path, thinning_precisions, currently_nondominated=True):
    """Performs thinning of all the archives in the input path and stores the thinned archives in the output path.

       All input solutions are rounded according to the thinning precisions (in the normalized objective space) and
       added to the archives (one archive per thinning precision). If currently_nondominated is True, all solutions that
       are currently nondominated within the thinned archive are output. If currently_nondominated is False, only the
       solutions that are contained in the final archive are output. In this case, the two extreme solutions are also
       output.
    """
    # Check whether input path exits
    input_files = get_file_name_list(input_path, ".adat")
    if len(input_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(input_path))

    old_level = log_level('warning')

    precision_list = []  # List of dictionaries - one for each given thinning precision
    for prec in thinning_precisions:
        precision_dict = {}
        precision_dict.update({'precision': prec})
        precision_dict.update({'name': '-{:.0e}'.format(prec)})
        precision_list.append(precision_dict)

    for input_file in input_files:
        for precision_dict in precision_list:
            output_file_name = input_file.replace(input_path, output_path + precision_dict.get('name'))
            precision_dict.update({'output_file': output_file_name})
            create_path(os.path.dirname(output_file_name))
            precision_dict.update({'thinned_solutions': 0})
            precision_dict.update({'f_out': None})
            precision_dict.update({'archive': None})
            precision_dict.update({'thinned_solutions': 0})
            precision_dict.update({'updated': 0})

        try:
            (suite_name, function, dimension) = parse_archive_file_name(input_file)
        except PreprocessingWarning as warning:
            print('Skipping file {}\n{}'.format(input_file, warning))
            continue

        print(input_file)

        normalization = None
        ideal = None
        all_solutions = 0

        with open(input_file, 'r') as f_in:
            for line in f_in:
                if line[0] == '%' and 'instance' in line:
                    finalize_output(precision_list, all_solutions, currently_nondominated)
                    instance = int(get_key_value(line[1:], 'instance').strip(' \t\n\r'))
                    # Limit to instance = 1, TODO: Filter functions and instances through parameters!
                    if instance > 1:
                        break

                    # Create an archive for every precision
                    for precision_dict in precision_list:
                        precision_dict.update({'thinned_solutions': 0})
                        precision_dict.update({'f_out': open(precision_dict.get('output_file'), 'a')})
                        precision_dict.update({'archive': Archive(suite_name, function, dimension, instance)})
                        precision_dict.update({'thinned_solutions': 0})
                        precision_dict.update({'updated': 0})
                        precision_dict.get('f_out').write(line)

                    archive = precision_list[0].get('archive')
                    extreme1 = [float(x) for x in archive.get_next_solution_text().split()[1:3]]
                    extreme2 = [float(x) for x in archive.get_next_solution_text().split()[1:3]]
                    ideal = [min(x, y) for x, y in zip(extreme1, extreme2)]
                    nadir = [max(x, y) for x, y in zip(extreme1, extreme2)]
                    normalization = [x - y for x, y in zip(nadir, ideal)]

                elif line[0] == '%':
                    for precision_dict in precision_list:
                        f_out = precision_dict.get('f_out')
                        if f_out and not f_out.closed:
                            f_out.write(line)

                elif len(line) == 0 or len(line.split()) < 3:
                    continue

                else:  # The line contains a solution
                    try:
                        # Fill the archives with the rounded solutions values wrt the different precisions
                        f_original = [float(x) for x in line.split()[1:3]]
                        for precision_dict in precision_list:
                            precision = precision_dict.get('precision')
                            f_normalized = [(f_original[i] - ideal[i]) / normalization[i] for i in range(2)]
                            f_normalized = [round(f_normalized[i] / precision) for i in range(2)]
                            f_normalized = [f_normalized[i] * precision for i in range(2)]
                            updated = precision_dict.get('archive').add_solution(f_normalized[0], f_normalized[1], line)
                            precision_dict.update({'updated': updated})
                    except IndexError:
                        print('Problem in file {}, line {}, skipping line'.format(input_file, line))
                        continue
                    finally:
                        all_solutions += 1

                    if currently_nondominated:
                        for precision_dict in precision_list:
                            if precision_dict.get('updated') == 1 or line.split()[0] == '0':
                                precision_dict['thinned_solutions'] += 1
                                precision_dict.get('f_out').write(line)

        finalize_output(precision_list, all_solutions, currently_nondominated)

    log_level(old_level)


if __name__ == '__main__':
    """A script for thinning the archives of solutions w.r.t. a given precision.
    """
    import timing  # Used even if it looks like it's not (i.e. do not delete this line)

    # TODO: These data should be overwritten from the command line parameters!

    # Set the default input and output paths
    input_path = '/Volumes/STORAGE/Data/archives/archives-output_2016_03_30_only_i01-i05'
    output_path = '/Volumes/STORAGE/Data/archives/thinning/output-archives_2016_03_30_only_i01-i05'
    thinning_precisions = [5e-5, 1e-5, 5e-6]

    # Analyze the archives
    archive_thinning(input_path, output_path, thinning_precisions, False)

