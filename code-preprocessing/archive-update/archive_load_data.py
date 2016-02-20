# -*- coding: utf-8 -*-
"""Raw pre-processing routines for loading data from archive files and files
   with best hypervolume values.
"""
import os
import os.path
import re
from time import gmtime, strftime
 
from archive_exceptions import PreprocessingWarning
    
def get_file_name_list(path):
    """Returns the list of files contained in any subfolder on the given path.
    """
    file_name_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_name_list.append(os.path.join(dirpath, filename))
    return file_name_list

def parse_file_name(file_name):
    """Retrieves information from the given file name in the following form:
       suite_name, function, dimension
       Assumes the following naming convention: 
       [suite-name]_f[function]_d[dimension]_*.*
    """
    split = os.path.basename(file_name).split("_")
    if (len(split) < 3) or (split[1][0] != 'f') or (split[2][0] != 'd'):
        raise PreprocessingWarning("File name '{}' not in expected format " 
        "'[suite-name]_f[function]_d[dimension]_*.*'".format(file_name))
        
    suite_name = split[0]
    function = int(split[1][1:])
    dimension = int(split[2][1:])
    return suite_name, function, dimension
    
def get_key_value(string, key):
    """Extracts the value corresponding to the first found key in the given
       string of comma-separated pairs of key = value.
    """
    p = re.compile('\ *([^,=]+?)\ *=\ *(".+?"|\'.+?\'|[^,]+)\ *(?=,|$)')
    for elem0, elem1 in p.findall(string):
        if (elem0.strip() == key):
            if elem1.startswith('\'') and elem1.endswith('\''): # HACK
                elem1 = ('\'' + re.sub(r'(?<!\\)(\')', r'\\\1', elem1[1:-1]) + '\'')
            return elem1
    # If the key has not been found, return None:
    return None
    
def get_instances(file_name):
    """Returns the list of instances contained in the given archive file's 
       comments (lines beginning with %).    
    """
    result = []
    with open(file_name, 'r') as f:
        for line in f:
            if (line[0] == '%'):
                if ("instance" in line):
                    value = get_key_value(line[1:], "instance")
                    if (value is not None):
                        result.append(int(value))
        f.close()
        
    if (len(result) == 0):
        raise PreprocessingWarning("File '{}' does not contain an "
        "'instance' string".format(file_name))
        
    return result

def get_archive_file_info(file_name):
    """Returns information on the problem instances contained in the given 
       archive file in the form of the following list of lists:
       file_name, suite_name, function, dimension, instance1
       file_name, suite_name, function, dimension, instance2
       ...
       Suite_name, function and dimension are retrieved from the file name, 
       while instance numbers are read from the file.
    """
    try:
        (suite_name, function, dimension) = parse_file_name(file_name)
        instances = get_instances(file_name)
    except PreprocessingWarning as warning:
        raise PreprocessingWarning("Skipping file {} \n {}".format(file_name, warning))
        
    result = []
    for instance in instances:
        result.append((file_name, suite_name, function, dimension, instance))
    return result        
    
def create_path(path):
    """Creates path if it does not already exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def read_best_values(file_list):
    """Reads the best hypervolume values from files in file_list, where each
       is formatted as a C source file. Returns a dictionary containing 
       problem names and their best known hypervolume values.
    """
    result = {}
    for file_name in file_list:
        read = False
        with open(file_name, 'r') as f:
            for line in f:
                if read:
                    if (line[0:2] == '};'):
                        break
                    split = re.split(',|\"|\t| |\n', line)
                    entries = [item for item in split if item]
                    result.update({entries[0] : entries[1]})
                elif (line[0:6] == 'static'):
                    read = True
            f.close()

    return result
        
def write_best_values(dic, file_name):
    """Appends problem names and hypervolume values into file_name formatted as
       a C source file.
    """
    with open(file_name, 'a') as f:
        f.write(strftime("\n/* Best values on %d.%m.%Y %H:%M:%S */\n", gmtime()))
        for key, value in sorted(dic.items()):
            f.write("  \"{} {:.15f}\",\n".format(key, value))
        f.close()
        
if __name__ == '__main__':
    pass
