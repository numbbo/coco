# A series of tests to check whether the python scripts of log-reconstruction perform correctly.
# Start the tests by writing
# py.test
# or
# python -m pytest
# in a terminal window on this folder

from os.path import dirname, abspath, join, exists
from os import walk, remove, rmdir, chdir, chmod, mkdir


def almost_equal(value1, value2, precision):
    return abs(value1 - value2) < precision


def get_lines(file_name):
    with open(file_name, 'r') as f:
        result = f.readlines()
        f.close()
    return result


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def compare_files(first_file, second_file, precision=1e-6):
    """
    Returns true if two files are equal and False otherwise. Any numbers are compared w.r.t. the given precision.
    Values of the "coco_version" are ignored.
    """

    lines1 = get_lines(first_file)
    lines2 = get_lines(second_file)

    if len(lines1) != len(lines2):
        return False

    for line1, line2 in zip(lines1, lines2):

        words1 = line1.split()
        words2 = line2.split()

        if len(words1) != len(words2):
            return False

        for word1, word2 in zip(words1, words2):

            if "coco_version" in word1 and "coco_version" in word2:
                break

            if is_float(word1) and is_float(word2):
                if not almost_equal(float(word1), float(word2), precision):
                    return False
            else:
                if word1 != word2:
                    return False
    return True


def prepare_reconstruction_data(download_data=False):
    """
    Prepares the data needed for the tests (deletes the exdata folder) and, if download_data is True, downloads the
    test data from the internet.
    """
    import urllib
    import tarfile
    cleanup_reconstruction_data()
    data_folder = abspath(join(dirname(__file__), 'test-data'))
    if download_data and (not exists(abspath(join(data_folder, 'archives-input'))) or not exists(
            abspath(join(data_folder, 'reconstruction')))):
        cleanup_reconstruction_data(True)
        chdir(abspath(dirname(__file__)))
        data_url = 'link-to-log-reconstruction-test-data.tgz'
        filename, headers = urllib.urlretrieve(data_url)
        tar_file = tarfile.open(filename)
        tar_file.extractall()

        for root, dirs, files in walk(data_folder, topdown=False):
            for name in files:
                # Change file permission so it can be deleted
                chmod(join(root, name), 0o777)


def cleanup_reconstruction_data(delete_all=False):
    """
    Deletes the exdata folder. If delete_all is True, deletes also the test-data folder.
    """

    if exists(abspath(join(dirname(__file__), 'exdata'))):
        for root, dirs, files in walk(abspath(join(dirname(__file__), 'exdata')), topdown=False):
            for name in files:
                remove(join(root, name))
            for name in dirs:
                rmdir(join(root, name))
        rmdir(abspath(join(dirname(__file__), 'exdata')))

    if delete_all and exists(abspath(join(dirname(__file__), 'test-data'))):
        for root, dirs, files in walk(abspath(join(dirname(__file__), 'test-data')), topdown=False):
            for name in files:
                remove(join(root, name))
            for name in dirs:
                rmdir(join(root, name))
        rmdir(abspath(join(dirname(__file__), 'test-data')))


def run_log_reconstruct():
    """
    Tests whether log_reconstruct() from log_reconstruct.py works correctly for the given input.
    """
    from log_reconstruct import log_reconstruct
    from cocoprep.archive_load_data import parse_range

    base_path = dirname(__file__)
    log_reconstruct(abspath(join(base_path, 'test-data', 'archives-input')),
                    'reconstruction',
                    'RECONSTRUCTOR',
                    'A test for reconstruction of logger output',
                    parse_range('1-55'),
                    parse_range('1-10'),
                    parse_range('2,3,5,10,20,40'))

    # Ignore `.rdat`, `.mdat` and other files
    endings = ('.info', '.dat', '.tdat', '.adat')

    for root, dirs, files in walk(abspath(join(base_path, 'exdata', 'reconstruction')), topdown=False):
        files = [f for f in files if f.endswith(endings)]
        for name in files:
            compare_files(abspath(join(root, name)),
                          abspath(join(root, name)).replace('exdata', 'test-data'))


def run_merge_lines():
    """
    Tests whether merge_lines_in() from merge_lines_in_info_files.py works correctly for the given input.
    """
    from merge_lines_in_info_files import merge_lines_in
    import shutil

    base_path = dirname(__file__)
    in_path = abspath(join(base_path, 'exdata', 'reconstruction'))
    out_path = abspath(join(base_path, 'exdata', 'reconstruction-merged'))
    mkdir(out_path)

    for root, dirs, files in walk(in_path, topdown=False):
        for name in files:
            if name.endswith('.info'):
                shutil.copyfile(abspath(join(in_path, name)), abspath(join(out_path, name)))
                merge_lines_in(abspath(join(root, name)), in_path, out_path)

    for root, dirs, files in walk(out_path, topdown=False):
        for name in files:
            compare_files(abspath(join(root, name)),
                          abspath(join(root, name)).replace('exdata', 'test-data'))


def test_all():
    """
    Runs a number of tests to check whether the python scripts of log-reconstruction perform correctly.
    The name of the method needs to start with "test_" so that it gets picked up by py.test.
    """

    prepare_reconstruction_data()

    run_log_reconstruct()

    run_merge_lines()

    cleanup_reconstruction_data()


if __name__ == '__main__':
    test_all()
