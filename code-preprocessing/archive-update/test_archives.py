# A series of tests to check whether the python scripts of archive-update perform correctly.
# Start the tests by writing
# py.test
# or
# python -m pytest
# in a terminal window on this folder

from os.path import dirname, abspath, join, exists
from os import walk, remove, rmdir, chdir, chmod


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


def prepare_archive_data(download_data=False):
    """
    Prepares the data needed for the tests (cleans up the test-data folder of unnecessary files) and, if download_data
    is True, downloads the test data from the internet.
    """
    import urllib
    import tarfile
    cleanup_archive_data()
    data_folder = abspath(join(dirname(__file__), 'test-data'))
    if download_data and (not exists(abspath(join(data_folder, 'archives-input'))) or not exists(
            abspath(join(data_folder, 'archives-results')))):
        cleanup_archive_data(True)
        chdir(abspath(dirname(__file__)))
        data_url = 'link-to-archive-update-test-data.tgz'
        filename, headers = urllib.urlretrieve(data_url)
        tar_file = tarfile.open(filename)
        tar_file.extractall()

        for root, dirs, files in walk(data_folder, topdown=False):
            for name in files:
                # Change file permission so it can be deleted
                chmod(join(root, name), 0777)


def cleanup_archive_data(delete_all=False):
    """
    Deletes unnecessary data from the test-data folder (keeps only archives-input and archives-results). If delete_all
    is True, deletes the entire test-data folder.
    """
    data_folder = abspath(join(dirname(__file__), 'test-data'))

    def can_delete(path):
        if delete_all:
            return True
        keep = [abspath(join(data_folder, 'archives-input')),
                abspath(join(data_folder, 'archives-results'))]
        result = True
        for keep_name in keep:
            if path.find(keep_name) >= 0:
                result = False
        return result

    for root, dirs, files in walk(data_folder, topdown=False):
        for name in files:
            if can_delete(join(root, name)):
                remove(join(root, name))
        for name in dirs:
            if can_delete(join(root, name)):
                rmdir(join(root, name))


def run_archive_update():
    """
    Tests whether merge_archives() from archive_update.py works correctly for the given input.
    """
    from archive_update import merge_archives
    from cocoprep.archive_load_data import parse_range

    base_path = dirname(__file__)
    new_hypervolumes = merge_archives(abspath(join(base_path, 'test-data', 'archives-input')),
                                      abspath(join(base_path, 'test-data', 'archives-output')),
                                      parse_range('1-55'),
                                      parse_range('1-10'),
                                      parse_range('2,3,5,10,20,40'),
                                      False)

    precision = 1e-13

    assert len(new_hypervolumes) == 22

    assert almost_equal(new_hypervolumes.get('bbob-biobj_f01_i04_d02'), 0.107610318984904, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f01_i04_d03'), 0.227870801380100, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f01_i04_d05'), 0.438362398133288, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f01_i04_d10'), 0.742933437184518, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f01_i04_d20'), 0.587349925250638, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f01_i04_d40'), 0.359511886735384, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f03_i06_d05'), 0.038070322787987, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f03_i07_d05'), 0.129884501203751, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f03_i08_d05'), 0.000760506516737, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f03_i09_d05'), 0.025178346536679, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f03_i10_d05'), 0.001064503341995, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f08_i06_d05'), 0.791099512196690, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f16_i02_d05'), 0.888980819178966, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f17_i01_d05'), 0.948755656523708, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f18_i07_d10'), 0.948488874548393, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f24_i10_d03'), 0.985816809701546, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f25_i02_d05'), 0.933561771067860, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f34_i07_d05'), 0.951275562997383, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f48_i07_d02'), 0.985762281913168, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f50_i07_d02'), 0.893071152604545, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f51_i02_d05'), 0.920488608198097, precision)
    assert almost_equal(new_hypervolumes.get('bbob-biobj_f52_i07_d02'), 0.920581303184137, precision)


def run_archive_reformat():
    """
    Tests whether reformat_archives() from archive_reformat.py works correctly for the given input.
    """
    from archive_reformat import reformat_archives
    from cocoprep.archive_load_data import parse_range

    base_path = dirname(__file__)
    reformat_archives(abspath(join(base_path, 'test-data', 'archives-input')),
                      abspath(join(base_path, 'test-data', 'archives-reformatted')),
                      parse_range('1-55'),
                      parse_range('1-10'),
                      parse_range('2,3,5,10,20,40'))

    for root, dirs, files in walk(abspath(join(base_path, 'test-data', 'archives-reformatted')), topdown=False):
        for name in files:
            assert compare_files(abspath(join(base_path, 'test-data', 'archives-results', name)),
                                 abspath(join(base_path, 'test-data', 'archives-reformatted', name)))


def run_archive_split():
    """
    Tests whether archive_split() from archive_split.py works correctly for the given input.
    """
    from archive_split import archive_split
    from cocoprep.archive_load_data import parse_range

    base_path = dirname(__file__)
    archive_split(abspath(join(base_path, 'test-data', 'archives-input')),
                  abspath(join(base_path, 'test-data', 'archives-split')),
                  parse_range('1-55'),
                  parse_range('1-10'),
                  parse_range('2,3,5,10,20,40'))

    for root, dirs, files in walk(abspath(join(base_path, 'test-data', 'archives-split')), topdown=False):
        for name in files:
            assert compare_files(abspath(join(base_path, 'test-data', 'archives-results', name)),
                                 abspath(join(base_path, 'test-data', 'archives-split', name)))


def run_archive_thinning():
    """
    Tests whether archive_thinning() from archive_thinning.py works correctly for the given input.
    """
    from archive_thinning import archive_thinning
    from cocoprep.archive_load_data import parse_range

    base_path = dirname(__file__)
    archive_thinning(abspath(join(base_path, 'test-data', 'archives-input')),
                     abspath(join(base_path, 'test-data', 'archives-thinned')),
                     1e-3,
                     False,
                     parse_range('1'),
                     parse_range('1-10'),
                     parse_range('2,3,5,10,20,40'))

    for root, dirs, files in walk(abspath(join(base_path, 'test-data', 'archives-thinned')), topdown=False):
        for name in files:
            assert compare_files(abspath(join(base_path, 'test-data', 'archives-results', name)),
                                 abspath(join(base_path, 'test-data', 'archives-thinned', name)))


def run_archive_analysis():
    """
    Tests whether archive_analysis() and summary_analysis() from archive_analysis.py work correctly for the given input.
    """
    from archive_analysis import archive_analysis, summary_analysis
    from cocoprep.archive_load_data import parse_range

    base_path = dirname(__file__)
    archive_analysis(abspath(join(base_path, 'test-data', 'archives-input')),
                     abspath(join(base_path, 'test-data', 'archives-analysis')),
                     -3.5,
                     5,
                     parse_range('48'),
                     parse_range('1-10'),
                     parse_range('2,3,5,10,20,40'))

    for root, dirs, files in walk(abspath(join(base_path, 'test-data', 'archives-analysis')), topdown=False):
        for name in files:
            assert compare_files(abspath(join(base_path, 'test-data', 'archives-results', name)),
                                 abspath(join(base_path, 'test-data', 'archives-analysis', name)))

    summary_analysis(abspath(join(base_path, 'test-data', 'archives-analysis')),
                     abspath(join(base_path, 'test-data', 'archives-analysis.txt')),
                     -3.5,
                     5,
                     parse_range('48'),
                     parse_range('1-10'),
                     parse_range('2,3,5,10,20,40'))

    assert compare_files(abspath(join(base_path, 'test-data', 'archives-analysis.txt')),
                         abspath(join(base_path, 'test-data', 'archives-results', 'archives-analysis.txt')))


def run_archive_difference():
    """
    Tests whether archive_difference() from archive_difference.py works correctly for the given input.
    """
    from archive_difference import archive_difference
    from cocoprep.archive_load_data import parse_range

    base_path = dirname(__file__)
    archive_difference(abspath(join(base_path, 'test-data', 'archives-input', 'a')),
                       abspath(join(base_path, 'test-data', 'archives-input', 'b')),
                       abspath(join(base_path, 'test-data', 'archives-diff.txt')),
                       parse_range('1-55'),
                       parse_range('1-10'),
                       parse_range('2,3,5,10,20,40'))

    assert compare_files(abspath(join(base_path, 'test-data', 'archives-diff.txt')),
                         abspath(join(base_path, 'test-data', 'archives-results', 'archives-diff.txt')))


def run_extract_extremes():
    """
    Tests whether extract_extremes() from extract_extremes.py works correctly for the given input.
    """
    from extract_extremes import extract_extremes
    from cocoprep.archive_load_data import parse_range

    base_path = dirname(__file__)
    extract_extremes(abspath(join(base_path, 'test-data', 'archives-input')),
                     abspath(join(base_path, 'test-data', 'archives-extremes.txt')),
                     parse_range('1-55'),
                     parse_range('1-10'),
                     parse_range('2,3,5,10,20,40'))

    assert compare_files(abspath(join(base_path, 'test-data', 'archives-extremes.txt')),
                         abspath(join(base_path, 'test-data', 'archives-results', 'archives-extremes.txt')))


def test_all():
    """
    Runs a number of tests to check whether the python scripts of archive-update perform correctly.
    The name of the method needs to start with "test_" so that it gets picked up by py.test.
    """
    import timing

    prepare_archive_data()
    timing.log('prepare_archive_data done', timing.now())

    run_archive_update()
    timing.log('run_archive_update done', timing.now())

    run_archive_reformat()
    timing.log('run_archive_reformat done', timing.now())

    run_archive_split()
    timing.log('run_archive_split done', timing.now())

    run_archive_thinning()
    timing.log('run_archive_thinning done', timing.now())

    run_archive_analysis()
    timing.log('run_archive_analysis done', timing.now())

    run_archive_difference()
    timing.log('run_archive_difference done', timing.now())

    run_extract_extremes()
    timing.log('run_extract_extremes done', timing.now())

    cleanup_archive_data()
    timing.log('cleanup_archive_data done', timing.now())


if __name__ == '__main__':
    test_all()
