# -*- coding: utf-8 -*-
"""Raw pre-processing routines for loading data from archive files and files with best hypervolume values.
"""
from __future__ import division, print_function, unicode_literals

import os
import os.path
import re
import six
from time import gmtime, strftime
from itertools import groupby
from operator import itemgetter

from archive_exceptions import PreprocessingWarning, PreprocessingException


def get_file_name_list(paths, ending=None):
    """Returns the list of files contained in any sub-folder in the given paths (can be a single path or a list of
       paths).
       :param paths: paths to the directory (a string or a list of strings)
       :param ending: if given, restrict to files with the given ending
    """
    file_name_list = []
    if isinstance(paths, six.string_types):
        paths = [paths]
    for path in paths:
        for dir_path, dir_names, file_names in os.walk(path):
            dir_names.sort()
            file_names.sort()
            for file_name in file_names:
                if (ending and file_name.endswith(ending)) or not ending:
                    file_name_list.append(os.path.join(dir_path, file_name))

    return file_name_list


def create_path(path):
    """Creates path if it does not already exist.
    :param path: path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def remove_empty_file(file_name):
    """Removes the file with the given name if it is empty.
    :param file_name: name of the file
    """
    if os.path.isfile(file_name) and os.path.getsize(file_name) == 0:
        os.remove(file_name)


def get_key_value(string, key):
    """Extracts the value corresponding to the first found key in the given string of comma-separated pairs of
       key = value.
       :param key: the key searched for in the string
       :param string: the string containing the key
    """
    p = re.compile(' *([^,=]+?) *= *(".+?"|\'.+?\'|[^,]+) *(?=,|$)')
    for elem0, elem1 in p.findall(string):
        if elem0.strip() == key:
            if elem1.startswith('\'') and elem1.endswith('\''):  # HACK
                elem1 = ('\'' + re.sub(r'(?<!\\)(\')', r'\\\1', elem1[1:-1]) + '\'')
            return elem1
    # If the key has not been found, return None:
    return None


def parse_problem_instance_file_name(file_name):
    """Retrieves information from the given problem instance file name and returns it in the following form:
       suite_name, function, instance, dimension
       :param file_name: problem instance file name in form [suite-name]_f[function]_i[instance]_d[dimension]_*.*
    """
    split = os.path.basename(file_name).split('_')
    if (len(split) < 4) or (split[1][0] != 'f') or (split[2][0] != 'i') or (split[3][0] != 'd'):
        raise PreprocessingWarning('File name \'{}\' not in expected format '
                                   '\'[suite-name]_f[function]_i[instance]_d[dimension]_*.*\''.format(file_name))

    suite_name = split[0]
    function = int(split[1][1:])
    instance = int(split[2][1:])
    dimension = int(split[3][1:])
    return suite_name, function, instance, dimension


def parse_archive_file_name(file_name):
    """Retrieves information from the given archive file name and returns it in the following form:
       suite_name, function, instance, dimension
       :param file_name: archive file name in either form
       [suite-name]_f[function]_d[dimension]_*.* or
       [suite-name]_f[function]_i[instance]_d[dimension]_*.*
       If the former, instance is set to None.
    """
    message = 'File name \'{}\' not in expected format \'[suite-name]_f[function]_d[dimension]_*.*\' or' \
              '\'[suite-name]_f[function]_i[instance]_d[dimension]_*.*\''.format(file_name)

    split = os.path.basename(file_name).split('_')

    if (len(split) < 3) or (split[1][0] != 'f'):
        raise PreprocessingWarning(message)

    suite_name = split[0]
    function = int(split[1][1:])
    instance = None

    if (len(split) >= 3) and (split[2][0] == 'd'):
        dimension = int(split[2][1:])
    elif (len(split) >= 4) and (split[2][0] == 'i') and (split[3][0] == 'd'):
        instance = int(split[2][1:])
        dimension = int(split[3][1:])
    else:
        raise PreprocessingWarning(message)

    return suite_name, function, instance, dimension


def parse_old_arhive_file_name(file_name):
    """Retrieves information from the given old archive file name and returns it in the following form:
       function, dimension, instance
       :param file_name: old archive file name in form f[f1]-[f2]_i[i1]-[i2]_[d]D.txt
    """
    split = os.path.basename(file_name).split('_')
    if (len(split) != 3) or (split[0][0] != 'f') or (split[1][0] != 'i'):
        raise PreprocessingWarning('File name \'{}\' not in expected format '
                                   '\'f[f1]-[f2]_i[i1]-[i2]_[d]D.txt\''.format(file_name))

    function = split[0][1:]  # in form x-y
    function1 = function[:function.find("-")]
    function2 = function[function.find("-")+1:]
    chosen_functions = [1, 2, 6, 8, 13, 14, 15, 17, 20, 21]
    try:
        function1id = chosen_functions.index(int(function1))
        function2id = chosen_functions.index(int(function2))
        function = 10 * function1id + function2id - (function1id * (function1id + 1) / 2) + 1
        function = int(function)
    except ValueError:
        function = None

    instance = split[1][1:]  # in form x-y
    if instance == '2-4':
        instance = 1
    elif instance == '3-5':
        instance = 2
    elif instance == '7-8':
        instance = 3
    elif instance == '9-10':
        instance = 4
    elif instance == '11-12':
        instance = 5
    else:
        instance = None

    try:
        dimension = int(split[2].split('D.')[0])
    except ValueError:
        dimension = None

    return function, instance, dimension


def get_instances(file_name):
    """Returns the list of instances contained in the given archive file's comments (lines beginning with %).
       :param file_name: archive file name
    """
    result = []
    with open(file_name, 'r') as f:
        for line in f:
            if line[0] == '%' and 'instance' in line:
                value = get_key_value(line[1:], 'instance')
                if value is not None:
                    result.append(int(value))
        f.close()

    if len(result) == 0:
        raise PreprocessingWarning('File \'{}\' does not contain an \'instance\' string'.format(file_name))

    return result


def get_archive_file_info(file_name, functions, instances, dimensions):
    """Returns information on the problem instances contained in the given archive file that also correspond to the
       given functions, instances and dimensions in the form of the following list of lists:
       file_name, single_instance, suite_name, function, instance1, dimension
       file_name, single_instance, suite_name, function, instance2, dimension
       ...
       The suite_name, function and dimension are always retrieved from the file name, while instances are either (1)
       retrieved from the file name, if the file name is in form [suite-name]_f[function]_i[instance]_d[dimension]_*.*,
       or (2) read from the file. Value of single_instance is set to True if (1) and False if (2).
       :param file_name: archive file name
       :param functions: functions to be considered
       :param instances: instances to be considered
       :param dimensions: dimensions to be considered
    """
    try:
        (suite_name, function, instance, dimension) = parse_archive_file_name(file_name)
        if (function not in functions) or (dimension not in dimensions):
            return None
        if not instance:
            instance_list = get_instances(file_name)
            single_instance = False
        else:
            instance_list = [instance]
            single_instance = True
    except PreprocessingWarning as warning:
        raise PreprocessingWarning('Skipping file {}\n{}'.format(file_name, warning))

    result = []
    for instance in instance_list:
        if instance in instances:
            result.append((file_name, single_instance, suite_name, function, instance, dimension))
    return result


def read_best_values(file_list):
    """Reads the best hypervolume values from files in file_list, where each is formatted as a C source file
       (starts to read in the next line from the first encountered 'static').
       Returns a dictionary containing problem names and their best known hypervolume values.
       :param file_list: list of file names
    """
    result = {}
    for file_name in file_list:
        read = False
        with open(file_name, 'r') as f:
            for line in f:
                if read:
                    if line[0:2] == '};':
                        break
                    split = re.split(',|\"|\t| |\n', line)
                    entries = [item for item in split if item]
                    result.update({entries[0]: entries[1]})
                elif line[0:6] == 'static':
                    read = True
            f.close()

    return result


def write_best_values(dic, file_name):
    """Appends problem names and hypervolume values into file_name formatted as a C source file.
    :param file_name: file name
    :param dic: dictionary containing problem names and their best known hypervolume values
    """
    with open(file_name, 'a') as f:
        f.write(strftime('/* Best values on %d.%m.%Y %H:%M:%S */\n', gmtime()))
        for key, value in sorted(dic.items()):
            f.write('  \"{} {:.15f}\",\n'.format(key, value))
        f.close()


def parse_range(input_string=""):
    """Parses the input string containing integers and integer ranges, such as:
       1, 2-4, 5, 10
       Returns a list of integers:
       [1, 2, 3, 4, 5, 10]
       :param input_string: input string with integers and integer ranges (if empty, the result is an empty list)
    """
    if not input_string:
        return None

    selection = set()
    # Tokens are comma separated values
    tokens = [x.strip() for x in input_string.split(',')]
    for i in tokens:
        try:
            # Typically tokens are plain old integers
            selection.add(int(i))
        except ValueError:
            # If not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # Try to build a valid range
                    first = token[0]
                    last = token[len(token)-1]
                    for x in range(first, last+1):
                        selection.add(x)
            except:
                raise PreprocessingException('Range {} not in correct format'.format(input_string))
    return list(selection)


def get_range(input_set):
    """Parses the input set containing integers, such as:
       (1, 2, 10, 3, 4, 5)
       Returns the shortest string of sorted integers and integer ranges:
       1, 2-5, 10
       :param input_set: input set with integers (if empty, the result is an empty string)
    """
    result = []
    for k, g in groupby(enumerate(sorted(input_set)), lambda x: x[0] - x[1]):
        i_list = list(map(itemgetter(1), g))
        if len(i_list) > 1:
            result.append('{}-{}'.format(i_list[0], i_list[-1]))
        else:
            result.append('{}'.format(i_list[0]))
    return ','.join(result)
