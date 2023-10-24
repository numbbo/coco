#!/usr/bin/env python
"""Tests that the given logger (bbob or bbob-biobj) produces the same output as the one from the
data folder. Random search with a fixed seed is used to evaluate solutions.
"""
from __future__ import division, print_function
import os
import math
import re
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve


def regression_test_match_words(old_word, new_word, accuracy=1e-6):
    """Checks whether the two words match (takes into account a dictionary of exceptions and the
    'almost matching' file names)

    If they don't, checks whether they match as floats with the given accuracy
    """
    exceptions = {'bbob-old': 'bbob'}

    old_word = old_word.strip('\'')
    new_word = new_word.strip('\'')
    old_word = old_word.strip('\"')
    new_word = new_word.strip('\"')
    if old_word != new_word:
        if 'data' in old_word and regression_test_almost_match_file_names(old_word, new_word):
            return
        if old_word not in exceptions or \
                (old_word in exceptions and exceptions[old_word] != new_word):
            try:
                old_float = float(old_word)
                new_float = float(new_word)
                if not math.isclose(old_float, new_float, rel_tol=accuracy):
                    raise ValueError('floats {} and {} differ by more than {}'
                                     ''.format(old_float, new_float, accuracy))
            except ValueError as e:
                raise ValueError('{} and {} do not match\n{}'.format(old_word, new_word, e))


def regression_test_match_file_contents(old_file, new_file):
    """Checks whether the contents of the two files match.

    Handles separately the case of differing versions.
    """
    with open(old_file) as f_old, open(new_file) as f_new:
        for old_line, new_line in zip(f_old, f_new):
            if old_line != new_line:
                # Lines different, check why word by word
                for d in '\t\n,|;':
                    old_line = old_line.replace(d, ' ')
                    new_line = new_line.replace(d, ' ')
                old_words = old_line.split()
                new_words = new_line.split()
                iterator = zip(old_words, new_words)
                for old_word, new_word in iterator:
                    if old_word == new_word == 'coco_version':
                        next(iterator)
                        next(iterator)
                    try:
                        regression_test_match_words(old_word, new_word)
                    except ValueError as e:
                        raise ValueError('The following lines do not match\n{}\n{}\n{}'
                                         ''.format(old_line, new_line, e))


def regression_test_almost_match_file_names(old_fname, new_fname):
    """Checks whether the two file names match (if the files match up to name_i*.ext, it counts
    as if they match)
    """
    if old_fname == new_fname:
        return True

    if '_i' in old_fname:
        if re.sub('_i[0-9]*', '', old_fname) == new_fname:
            return True

    return False


def regression_test_match_logger_output(old_data_folder, new_data_folder):
    """Checks whether the contents of the two folders match.

    The check includes only the '.info', '.dat', '.tdat', and '.adat' files, i.e., ignores and
    `.rdat`, `.mdat` and other files.
    """
    endings = ('.info', '.dat', '.tdat', '.adat')
    print('\nComparing the contents of {} and {}'.format(old_data_folder, new_data_folder))

    for data_folder in [old_data_folder, new_data_folder]:
        if not (os.path.exists(data_folder) and os.path.getsize(data_folder) > 0):
            raise ValueError('{} does not exist or is empty'.format(data_folder))

    for (old_root, old_dirs, old_files), (new_root, new_dirs, new_files) in \
            zip(os.walk(old_data_folder), os.walk(new_data_folder)):
        old_files = [f for f in old_files if f.endswith(endings)]
        new_files = [f for f in new_files if f.endswith(endings)]
        # Iterate over files in both folders sorted by name
        for old_fname, new_fname in zip(sorted(old_files), sorted(new_files)):
            if old_fname.endswith(endings) and new_fname.endswith(endings):
                if not regression_test_almost_match_file_names(old_fname, new_fname):
                    raise ValueError('File names {} and {} do not match'.format(old_fname,
                                                                                new_fname))
                try:
                    regression_test_match_file_contents(os.path.join(old_root, old_fname),
                                                        os.path.join(new_root, new_fname))
                except ValueError as e:
                    raise ValueError('The following files do not match\n{}\n{}\n{}'
                                     ''.format(old_fname, new_fname, e))


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    path = os.path.join('create', 'exdata')
    comparisons = [['bbob_bbob-old_def', 'bbob_bbob_def'],
                   ['bbob-constrained_bbob-old_def', 'bbob-constrained_bbob_def']]

    exception_count = 0
    try:
        for comparison in comparisons:
            try:
                # Check that the outputs match
                regression_test_match_logger_output(os.path.join(path, comparison[0]),
                                                    os.path.join(path, comparison[1]))
            except Exception as e:
                print('{}'.format(e))
                exception_count += 1
        print('Check completed!')
        if exception_count > 0:
            raise ValueError('Found {} exceptions'.format(exception_count))
    except Exception as e:
        print('{}'.format(e))
        exit(exception_count)
