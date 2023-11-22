#!/usr/bin/env python
"""Tests the cocopp module.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import time
import inspect
import fnmatch
import urllib
import tempfile
import shutil
import subprocess
import doctest

from cocopp import archiving

import matplotlib  # just to make sure the following is actually done first

matplotlib.use('Agg')  # To avoid window popup and use without X forwarding

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

test_bibtex = False

class InFolderGoneWithTheWind:
    """``with InFolderGoneWithTheWind(): ...`` executes the block in a

    temporary folder under the current folder. The temporary folder is
    deleted on exiting the block.

    CAVEAT: copy-pasted to toolsdivers

    >>> import os
    >>> dir_ = os.getcwd()  # for the record
    >>> len_ = len(os.listdir('.'))
    >>> with InfolderGoneWithTheWind():  # doctest: +SKIP
    ...     # do some work in a folder here, e.g. write files
    ...     len(dir_) > len(os.getcwd()) and os.getcwd() in dir_
    True
    >>> # magically we are back in the original folder
    >>> assert dir_ == os.getcwd()
    >>> assert len(os.listdir('.')) == len_

    """
    def __init__(self, prefix='_'):
        """no folder needs to be given"""
        self.prefix = prefix

    def __enter__(self):
        self.original_dir = os.getcwd()
        self.target_dir = tempfile.mkdtemp(prefix=self.prefix)
        print(f"INFO: Using temp directory '{self.target_dir}'")
        os.chdir(self.target_dir)

    def __exit__(self, *args):
        os.chdir(self.original_dir)
        shutil.rmtree(self.target_dir)


def depreciated_data_archive_get(substrs):
    """CAVEAT: this won't work anymore as the get_first method changed to
    get_one

    return first matching data paths for each element of `substrs`
    concatenated in a string.

    Specifically::

        return ' ' + ' '.join(cocopp._data_archive.get_first(substrs))

    Implemented via `subprocess` to prevent the need to import `cocopp` in
    this file here (prevent the script vs module absolute import issue).
    """
    res = subprocess.check_output(["python", "-c",
        """
from __future__ import division, print_function
from cocopp import data_archive
res = data_archive.get_first(""" + repr(substrs) + """)  # get first match for each substr
print('_split_here_ ' + ' '.join(res), end='')  # communication by print instead of return res
"""])
    # res is of type bytes
    # res contains any output printed during execution (e.g. user infos)
    # yet we should be able to split away and return the last print
    return str(res).split('_split_here_')[-1]

def data_archive_get(substrs):
    if str(substrs) == substrs:
        return substrs
    return ' ' + ' '.join(substrs)

def join_path(a, *p):
    joined_path = os.path.join(a, *p)
    return joined_path


def copy_latex_templates():
    current_folder = os.path.dirname(os.path.realpath(__file__))
    template_folder = os.path.abspath(join_path(current_folder, '..', 'latex-templates'))
    shutil.copy(join_path(template_folder, 'templateBBOBarticle.tex'), current_folder)
    shutil.copy(join_path(template_folder, 'templateBBOBcmp.tex'), current_folder)
    shutil.copy(join_path(template_folder, 'templateBBOBmany.tex'), current_folder)
    shutil.copy(join_path(template_folder, 'templateBIOBJarticle.tex'), current_folder)
    shutil.copy(join_path(template_folder, 'templateBIOBJmultiple.tex'), current_folder)
    shutil.copy(join_path(template_folder, 'templateNOISYarticle.tex'), current_folder)

    # Copy auxiliary files to the current working folder
    cwd = os.getcwd()
    shutil.copy(join_path(template_folder, 'acmart.cls'), cwd)
    shutil.copy(join_path(template_folder, 'ACM-Reference-Format.bst'), cwd)
    shutil.copy(join_path(template_folder, 'comment.sty'), cwd)
    shutil.copy(join_path(template_folder, 'acmcopyright.sty'), cwd)
    shutil.copy(join_path(template_folder, 'bbob.bib'), cwd)


def run_latex_template(filename, all_tests):
    file_path = os.path.abspath(join_path(os.path.dirname(__file__), filename))
    arguments = ['pdflatex', file_path]
    DEVNULL = open(os.devnull, 'wb')
    result = subprocess.call(arguments, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)
    assert not result, 'Test failed: error while generating pdf from %s.' % filename

    if all_tests and test_bibtex:
        file_path = os.path.splitext(file_path)[0]
        arguments = ['bibtex', file_path]
        DEVNULL = open(os.devnull, 'wb')
        output_file = open("bibtex.log", "w")
        result = subprocess.call(arguments, stdin=DEVNULL, stdout=output_file, stderr=DEVNULL)
        assert not result, 'Test failed: error while running bibtex on %s resuling in %s' % (os.path.splitext(filename)[0], str(result))


def retrieve_algorithm(data_path, folder_name, algorithm_name, file_name=None):
    """depreciated (replaced by cocopp._data_archive COCODataArchive instance
    replaced by `cocopp.archiving.get` or `cocopp.archiving.official_archives` or `cocopp.archives.all`)"""
    algorithm_file = join_path(data_path, file_name if file_name else algorithm_name)
    if not os.path.exists(algorithm_file):
        data_url = 'https://numbbo.github.io/data-archive/data-archive/%s/%s' % (folder_name, algorithm_name)
        urlretrieve(data_url, algorithm_file)


def depreciated_prepare_data(run_all_tests):
    print('preparing algorithm data')

    data_path = os.path.abspath(join_path(os.path.dirname(__file__), 'data'))

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Retrieving the algorithms
    # retrieve_algorithm(data_path, '2010', 'IPOP-ACTCMA-ES_ros_noiseless.tar.gz')
    # [outcommented and replaced by BIPOP until 2010 data is in new format]
    retrieve_algorithm(data_path, 'bbob/2009', 'BFGS_ros_noiseless.tgz')
    retrieve_algorithm(data_path, 'bbob-biobj/2016', 'RS-4.tgz')

    if run_all_tests:
        retrieve_algorithm(data_path, 'bbob/2009', 'BIPOP-CMA-ES_hansen_noiseless.tgz')
        retrieve_algorithm(data_path, 'bbob/2009', 'MCS_huyer_noiseless.tgz')
        retrieve_algorithm(data_path, 'bbob/2009', 'NEWUOA_ros_noiseless.tgz')
        retrieve_algorithm(data_path, 'bbob/2009', 'RANDOMSEARCH_auger_noiseless.tgz')
        retrieve_algorithm(data_path, 'bbob/2013', 'SMAC-BBOB_hutter_noiseless.tgz')
        retrieve_algorithm(data_path, 'bbob/2013', 'lmm-CMA-ES_auger_noiseless.tgz')
        retrieve_algorithm(data_path, 'bbob/2009', 'DE-PSO_garcia-nieto_noiseless.tgz')
        retrieve_algorithm(data_path, 'bbob/2009', 'VNS_garcia-martinez_noiseless.tgz')
        retrieve_algorithm(data_path, 'bbob-biobj/2016', 'RS-4.tgz')
        retrieve_algorithm(data_path, 'bbob-biobj/2016', 'RS-100.tgz')
        # diff. location and name due to Jenkins settings with too long paths
        retrieve_algorithm(data_path, 'test', 'N-II.tgz')
        retrieve_algorithm(data_path, 'test', 'RS-4.zip')
        retrieve_algorithm(data_path, 'bbob-noisy/2009', 'BFGS_ros_noisy.tgz')
        retrieve_algorithm(data_path, 'bbob-noisy/2009', 'MCS_huyer_noisy.tgz')

    return data_path


def process_doctest_output(stream=None):
    """ """
    import fileinput
    s1 = ""
    s2 = ""
    s3 = ""
    state = 0
    for line in fileinput.input(stream):  # takes argv as file or stdin
        if 1 < 3:

            s3 += line
            if state < -1 and line.startswith('***'):
                print(s3)
            if line.startswith('***'):
                s3 = ""

        if state == -1:  # found a failed example line
            s1 += '\n\n*** Failed Example:' + line
            s2 += '\n\n\n'  # line
            # state = 0  # wait for 'Expected:' line

        if line.startswith('Expected:'):
            state = 1
            continue
        elif line.startswith('Got:'):
            state = 2
            continue
        elif line.startswith('***'):  # marks end of failed example
            state = 0
        elif line.startswith('Failed example:'):
            state = -1
        elif line.startswith('Exception raised'):
            state = -2

        # in effect more else:
        if state == 1:
            s1 += line + ''
        if state == 2:
            s2 += line + ''


def delete_files(all_files=False):
    assert all_files
    if all_files:
        cwd = os.getcwd()
        os.remove(join_path(cwd, 'acmart.cls'))
        os.remove(join_path(cwd, 'ACM-Reference-Format.bst'))
        os.remove(join_path(cwd, 'comment.sty'))
        os.remove(join_path(cwd, 'acmcopyright.sty'))
        os.remove(join_path(cwd, 'bbob.bib'))

def tmp_test_constrained(python, command):
    url = "https://numbbo.github.io/cocopp-playground/bbob-constrained/testdata/bbob-constrained"
    archive = archiving.get(url)
    t0 = time.time()
    data_path = archive.get("fmincon_consbbob")
    print(python + command + data_path)
    with InFolderGoneWithTheWind():
        result = os.system(python + command + data_path)
    print('**  subtest 19 finished in ', time.time() - t0, ' seconds')
    assert result == 0, 'Test failed: rungeneric on one bbob-constrained algorithm.'


def main(arguments):
    """these tests are executed when ``python cocopp`` is called.

    with ``wine`` as second argument ``C:\\Python26\\python.exe``
    instead of ``python`` is called

    """

    run_all_tests = 'all' in arguments

    # Use the current Python version for all tests
    python = sys.executable
    print(f"INFO: Using {python} to run all tests.")

    command = ' -m cocopp --no-svg --settings=grayscale '  # TODO: grayscale has to go

    #copy_latex_templates()
    #print('LaTeX templates copied.')

    print('*** testing module cocopp ***')
    t0 = time.time()
    data_path = data_archive_get('BFGS_ros_noiseless')
    print(python + command + # '--conv ' +
          data_path)
    with InFolderGoneWithTheWind():
        result = os.system(python + command + # '--conv ' +
                           data_path)
    print('**  subtest 1 finished in ', time.time() - t0, ' seconds')
    assert result == 0, 'Test failed: rungeneric on one algorithm with option --conv.'
    #run_latex_template("templateBBOBarticle.tex", run_all_tests)

    t0 = time.time()
    data_path = data_archive_get('RANDOMSEARCH-4_Auger_bbob-biobj.tgz')
    print(python + command + data_path)
    with InFolderGoneWithTheWind():
        result = os.system(python + command + data_path)
    print('**  subtest 2 finished in ', time.time() - t0, ' seconds')
    assert result == 0, 'Test failed: rungeneric on one bi-objective algorithm.'
    #run_latex_template("templateBIOBJarticle.tex", run_all_tests)

    tmp_test_constrained(python, command)

    if run_all_tests:
        data_paths = data_archive_get([
                        'BIPOP-CMA-ES_hansen_noiseless',
                        'MCS_huyer_noiseless',
                        '2009/NEWUOA_ros_noiseless.tgz',
                        'RANDOMSEARCH_auger_noiseless.tgz',
                        'BFGS_ros_noiseless.tgz'])
        t0 = time.time()
        print(time.asctime())

        with InFolderGoneWithTheWind():
            result = os.system(python + command + data_paths)
        print('**  subtest 3 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on many algorithms.'
        #run_latex_template("templateBBOBmany.tex", run_all_tests)

        t0 = time.time()
        with InFolderGoneWithTheWind():
            result = os.system(python + command + data_archive_get([
                                'SMAC-BBOB_hutter_noiseless.tgz',
                                'lmm-CMA-ES_auger_noiseless.tgz']))
        print('**  subtest 4 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on two algorithms.'
        #run_latex_template("templateBBOBcmp.tex", run_all_tests)

        t0 = time.time()
        with InFolderGoneWithTheWind():
            result = os.system(python + command + ' --include-single ' + data_archive_get([
                                'DE-PSO_garcia-nieto_noiseless.tgz',
                                'VNS_garcia-martinez_noiseless.tgz']))
        print('**  subtest 5 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on two algorithms with option --include-single.'
        #run_latex_template("templateBBOBcmp.tex", run_all_tests)

        t0 = time.time()
        with InFolderGoneWithTheWind():
            result = os.system(python + command + ' --expensive ' + data_archive_get(
                                'VNS_garcia-martinez_noiseless.tgz'))
        print('**  subtest 6 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on one algorithm with option --expensive.'
        #run_latex_template("templateBBOBarticle.tex", run_all_tests)

        t0 = time.time()
        with InFolderGoneWithTheWind():
            result = os.system(python + command + data_archive_get([
                                'RANDOMSEARCH-4_Auger_bbob-biobj.tgz',
                                'RANDOMSEARCH-100_Auger_bbob-biobj.tgz']))
        print('**  subtest 7 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on two bbob-biobj algorithms.'
        #run_latex_template("templateBIOBJmultiple.tex", run_all_tests)
        
        t0 = time.time()
        # Previous note: we use the original GA-MULTIOBJ-NSGA-II.tgz data set
        # but with a shorter file name from the biobj-test folder
        # to avoid problems with too long path names on the windows
        # Jenkins slave
        with InFolderGoneWithTheWind():
            result = os.system(python + command + data_archive_get([
                                'NSGA-II-MATLAB_Auger_bbob-biobj.tgz',
                                'RANDOMSEARCH-4_Auger_bbob-biobj.tgz',
                                'RANDOMSEARCH-100_Auger_bbob-biobj.tgz']))
        print('**  subtest 8 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on three bbob-biobj algorithms.'
        #run_latex_template("templateBIOBJmultiple.tex", run_all_tests)

        # testing data from bbob-noisy suite:
        t0 = time.time()
        with InFolderGoneWithTheWind():
            result = os.system(python + command + data_archive_get([
                                'MCS_huyer_noisy.tgz',
                                'BFGS_ros_noisy.tgz']))
        print('**  subtest 9 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on two bbob-noisy algorithms.'
        #run_latex_template("templateNOISYarticle.tex", run_all_tests)
        
        # testing data from recent runs:
        recent_data_path = os.path.abspath(join_path(os.path.dirname(__file__),
                                                     '../../code-experiments/build/python/exdata'))
        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system(python + command +
                               join_path(recent_data_path, 'RS-bb'))
            print('**  subtest 10 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: rungeneric on newly generated random search data on `bbob`.'

        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system(python + command +
                               join_path(recent_data_path, 'RS-bi'))
            print('**  subtest 11 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: rungeneric on newly generated random search data on `bbob-biobj`.'

        #with InfolderGoneWithTheWind():
            # t0 = time.time()
            # result = os.system(python + command +
            #                    join_path(recent_data_path, 'RS-co'))
            # print('**  subtest 12 finished in ', time.time() - t0, ' seconds')
            # assert result == 0, 'Test failed: rungeneric on newly generated random search data on `bbob-constrained`.'
            # delete_files(all_files=True)

        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system(python + command + data_archive_get(
                'test/RS-4.zip'))
            print('**  subtest 13 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: rungeneric on RS-4.zip.'

        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system(python + command +
                               join_path(recent_data_path, 'RS-la'))
            print('**  subtest 14 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: rungeneric on newly generated random search data on `bbob-largescale`.'

        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system(python + command +
                               join_path(recent_data_path, 'RS-mi'))
            print('**  subtest 15 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: rungeneric on newly generated random search data on `bbob-mixint`.'

        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system(python + command +
                               join_path(recent_data_path, 'RS-bi-mi'))
            print('**  subtest 16 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: rungeneric on newly generated random search data on `bbob-biobj-mixint`.'

        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system(python + command +
                               'bbob/2009/BFGS! bbob-largescale/2019/LBFGS!')
            print('**  subtest 17 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: rungeneric on data from `bbob` and `bbob-largescale` suite.'

        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system(python + command + 'NSGA-II! 2019/IBEA!')
            print('**  subtest 18 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: rungeneric on data from `bbob-biobj` and `bbob-biobj-ext` suite.'

        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system("""python -c "import cocopp; cocopp.main('slsqp!'); cocopp.main('abc*')" """)
            print('**  subtest 19 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: running postprocessing twice within same python session'

        with InFolderGoneWithTheWind():
            t0 = time.time()
            result = os.system("""python -c "import cocopp; cocopp.main('constrained/2022/RandomSearch-5')" """)
            print('**  subtest 20 finished in ', time.time() - t0, ' seconds')
            assert result == 0, 'Test failed: running postprocessing on bbob-constrained data'


    print('launching doctest (it might be necessary to close a few pop up windows to finish)')
    t0 = time.time()

    failure_count = 0
    test_count = 0
    if 1 < 3:
        # doctest.testmod(report=True, verbose=True)  # this is quite cool!
        # go through the py files in the cocopp folder
        current_path = os.getcwd()
        new_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        os.chdir(new_path)
        for root, dirnames, filenames in os.walk(os.path.dirname(os.path.realpath(__file__))):
            for filename in fnmatch.filter(filenames, '*.py'):
                current_failure_count, current_test_count = doctest.testfile(
                    os.path.join(root, filename), report=True, module_relative=False)
                failure_count += current_failure_count
                test_count += current_test_count
                if current_failure_count:
                    print('doctest file "%s" failed' % os.path.join(root, filename))
        os.chdir(current_path)
    else:
        stdout = sys.stdout
        fn = '_cocopp_doctest_.txt'
        try:
            with open(fn, 'w') as f:
                sys.stdout = f
                doctest.testmod(report=True)
        finally:
            sys.stdout = stdout
        process_doctest_output(fn)
    print('** doctest finished in ', time.time() - t0, ' seconds')
    # print('    more info in file _cocopp_doctest_.txt)')
    print('*** done testing module cocopp ***')

    if failure_count > 0:
        raise ValueError('%d of %d tests failed' % (failure_count, test_count))


"""
        import cocopp
        print(dir(cocopp))
        for s in dir(cocopp):
            if(inspect.ismodule(eval("cocopp."+s)) and s[:2]!="__"):
                print("cocopp."+s)
                doctest.testmod(eval("cocopp."+s),verbose=False)
        print(cocopp.__all__)
"""

if __name__ == "__main__":
    main(sys.argv[1:])
