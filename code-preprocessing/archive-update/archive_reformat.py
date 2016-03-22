# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import os

from cocoprep.archive_load_data import create_path, get_file_name_list, parse_old_arhive_file_name
from cocoprep.archive_exceptions import PreprocessingException


def reformat_archives(input_path, output_path):
    """
    The names of the files in the input_path have the following notation:
    f[f1]-[f2]_i[i1]-[i2]_[d]D.txt
    where f1 and f2 are function numbers used for the first and second objective, i1 and i2 are instance numbers of the
    two functions and d is the dimension (one among 2, 3, 5, 10 and 20). Each such file starts with a few lines of
    comments that start with '#', after which each line corresponds to one solutions. In files with d <= 5 the solution
    is represented by its decision and objective vector values, while files with d > 5 contain only objective vector
    values of each solution.

    The output files to be written to output_path have the following notation:
    [suite_name]_f[F]_d[D]_nondominated.adat
    where F is the function number in the suite and D is the dimension. One file contains multiple instances. Each
    starts with a line '% instance = I', where I is the instance number and is followed by a commented line (starting
    with '%'). In the subsequent lines, the solutions are written in the following format:
    num obj1 obj2 dec1 ... decn
    where num is the evaluation number of the solution (0 for extreme solutions and 1 for solutions read from the old
    file format), obj1 and obj2 are its objective values, and dec1, ... are its decision values (if they are given).

    Note this implementation is concerned only with the 'bbob-biobj' suite and applies reformatting only on the archive
    files that correspond to the problems contained in this suite.

    :param input_path: path to the folder with input archives
    :param output_path: path to the folder where output archives are stored to, if any files already exist there, they
    get appended to
    """
    suite_name = 'bbob-biobj'
    print('Reformatting archive files for the {} suite...'.format(suite_name))

    # Check whether input folder exits
    input_files = get_file_name_list(input_path)
    if len(input_files) == 0:
        raise PreprocessingException('Folder {} does not exist or is empty'.format(input_path))

    # Create output folder if it does not exist yet
    create_path(output_path)

    # Read the input files one by one
    for input_file in input_files:

        (function, dimension, instance) = parse_old_arhive_file_name(input_file)
        if (function is None) or (dimension is None) or (instance is None):
            # If any problems are encountered, the file is skipped
            continue

        # Open the output file
        output_file = os.path.join(output_path, '{}_f{:02d}_d{:02d}_nondominated.adat'.format(suite_name,
                                                                                              function,
                                                                                              dimension))

        with open(input_file, 'r') as f_in:
            with open(output_file, 'a') as f_out:
                # Perform reformatting
                print(input_file)
                f_out.write('% instance = {}\n%\n'.format(instance))

                for line in f_in:
                    if line[0] == '#':
                        continue

                    if dimension <= 5:
                        f_out.write('1 \t{} \t{}\n'.format(' \t'.join(line.split()[dimension:dimension+2]),
                                                           ' \t'.join(line.split()[0:dimension])))
                    else:
                        f_out.write('1 \t{}\n'.format(' \t'.join(line.split()[0:2])))

            f_out.close()
        f_in.close()
    print('Done!')


if __name__ == '__main__':
    """A script for reformatting the archives of solutions.

       Archives from the input folder are read, reformatted and stored in the output folder.
    """

    # Reformat the archives
    reformat_archives('/Volumes/STORAGE/Data/mobbob-data/data/REFSET_5_instances',
                      '/Volumes/STORAGE/Data/archives')
