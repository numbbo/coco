#!/usr/bin/env python
"""Tests that the given logger (bbob or bbob-biobj) produces the same output as the one from the
data folder. Random search with a fixed seed is used to evaluate solutions.
"""
from __future__ import division, print_function
import os
import sys
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve
from create.create_logger_data import run_experiment
import math


def regression_test_match_words(old_word, new_word, accuracy=1e-6):
    """Checks whether the two words match

    If they don't, checks whether they match as floats with the given accuracy
    """
    if old_word != new_word:
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
                    regression_test_match_words(old_word, new_word)


def regression_test_match_logger_output(old_data_folder, new_data_folder):
    """Checks whether the contents of the two folders match.

    The check includes only the '.info', '.dat', '.tdat', and '.adat' files, i.e., ignores and
    `.rdat` and other files.
    """
    endings = ('.info', '.dat', '.tdat', '.adat')

    for data_folder in [old_data_folder, new_data_folder]:
        if not (os.path.exists(data_folder) and os.path.getsize(data_folder) > 0):
            raise ValueError('{} does not exist or is empty'.format(data_folder))

    for (old_root, old_dirs, old_files), (new_root, new_dirs, new_files) in \
            zip(os.walk(old_data_folder), os.walk(new_data_folder)):
        # Iterate over files in both folders sorted by name
        for old_fname, new_fname in zip(sorted(old_files), sorted(new_files)):
            if old_fname.endswith(endings) and new_fname.endswith(endings):
                if old_fname != new_fname:
                    raise ValueError('File names {} and {} do not match'.format(old_fname,
                                                                                new_fname))
                regression_test_match_file_contents(os.path.join(old_root, old_fname),
                                                    os.path.join(new_root, new_fname))


if __name__ == "__main__":
    logger = 'bbob'
    try:
        logger = int(sys.argv[1])
    except IndexError:
        pass
    except Exception as e:
        raise e

    try:
        # Get the old logger output
        old_data_folder = os.path.join('data', '{}_logger_data'.format(logger))
        if not os.path.exists(old_data_folder):
            remote_data_path = 'http://coco.gforge.inria.fr/regression-tests/'
            # download data from remote_data_path:
            if not os.path.exists(os.path.split(old_data_folder)[0]):
                try:
                    os.makedirs(os.path.split(old_data_folder)[0])
                except Exception as e:
                    raise e
            url = '/'.join((remote_data_path, old_data_folder))
            print("Downloading {} to {}".format(url, old_data_folder))
            urlretrieve(url, old_data_folder)

        # Produce the new logger output
        new_data_folder = os.path.join('new_data', '{}_logger_data'.format(logger))
        new_data_folder_relative = os.path.join('..', 'new_data', '{}_logger_data'.format(logger))
        run_experiment(logger, new_data_folder_relative)

        # Check that the outputs match
        regression_test_match_logger_output(old_data_folder, new_data_folder)
    except Exception as e:
        print('{}'.format(e))
        exit(1)
