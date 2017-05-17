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
import shutil
import subprocess
import doctest

import matplotlib  # just to make sure the following is actually done first

matplotlib.use('Agg')  # To avoid window popup and use without X forwarding


def join_path(a, *p):
    joined_path = os.path.join(a, *p)
    return joined_path


def copy_latex_templates():
    current_folder = os.path.dirname(os.path.realpath(__file__))
    template_folder = os.path.abspath(join_path(current_folder, '..', 'latex-templates'))
    # template_folder = os.path.abspath('latex-templates')
    shutil.copy(join_path(template_folder, 'templateBBOBarticle.tex'), current_folder)
    shutil.copy(join_path(template_folder, 'templateBBOBcmp.tex'), current_folder)
    shutil.copy(join_path(template_folder, 'templateBBOBmany.tex'), current_folder)
    shutil.copy(join_path(template_folder, 'templateBIOBJarticle.tex'), current_folder)
    shutil.copy(join_path(template_folder, 'templateBIOBJmultiple.tex'), current_folder)

    # Copy auxiliary files to the current working folder
    cwd = os.getcwd()
    shutil.copy(join_path(template_folder, 'acmart.cls'), cwd)
    shutil.copy(join_path(template_folder, 'ACM-Reference-Format.bst'), cwd)
    shutil.copy(join_path(template_folder, 'comment.sty'), cwd)
    shutil.copy(join_path(template_folder, 'acmcopyright.sty'), cwd)
    shutil.copy(join_path(template_folder, 'bbob.bib'), cwd)


def run_latex_template(filename):
    file_path = os.path.abspath(join_path(os.path.dirname(__file__), filename))
    arguments = ['pdflatex', file_path]
    DEVNULL = open(os.devnull, 'wb')
    result = subprocess.call(arguments, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)
    assert not result, 'Test failed: error while generating pdf from %s.' % filename

    file_path = os.path.splitext(file_path)[0]
    arguments = ['bibtex', file_path]
    DEVNULL = open(os.devnull, 'wb')
    output_file = open("bibtex.log", "w")
    result = subprocess.call(arguments, stdin=DEVNULL, stdout=output_file, stderr=DEVNULL)
    assert not result, 'Test failed: error while running bibtex on %s.' % os.path.splitext(filename)[0]


def retrieve_algorithm(data_path, folder_name, algorithm_name, file_name=None):
    algorithm_file = join_path(data_path, file_name if file_name else algorithm_name)
    if not os.path.exists(algorithm_file):
        data_url = 'http://coco.gforge.inria.fr/data-archive/%s/%s' % (folder_name, algorithm_name)
        urllib.urlretrieve(data_url, algorithm_file)


def prepare_data(run_all_tests):
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
    shutil.rmtree('ppdata')
    if all_files:
        cwd = os.getcwd()
        os.remove(join_path(cwd, 'acmart.cls'))
        os.remove(join_path(cwd, 'ACM-Reference-Format.bst'))
        os.remove(join_path(cwd, 'comment.sty'))
        os.remove(join_path(cwd, 'acmcopyright.sty'))
        os.remove(join_path(cwd, 'bbob.bib'))


def main(arguments):
    """these tests are executed when ``python cocopp`` is called.

    with ``wine`` as second argument ``C:\\Python26\\python.exe``
    instead of ``python`` is called

    """

    run_all_tests = len(arguments) == 1 and arguments[0] == 'all'

    python = 'python -m '  # how to call python
    if len(sys.argv) > 1 and sys.argv[1] == 'wine':
        python = 'C:\\Python26\\python.exe '  # works for wine

    data_path = ' ' + prepare_data(run_all_tests)

    command = ' cocopp --no-svg --settings=grayscale '

    copy_latex_templates()
    print('LaTeX templates copied.')

    print('*** testing module cocopp ***')
    t0 = time.time()
    print(python + command + '--conv' + join_path(data_path, 'BFGS_ros_noiseless.tgz'))
    result = os.system(python + command + '--conv' + join_path(data_path, 'BFGS_ros_noiseless.tgz'))
    print('**  subtest 1 finished in ', time.time() - t0, ' seconds')
    assert result == 0, 'Test failed: rungeneric on one algorithm with option --conv.'
    run_latex_template("templateBBOBarticle.tex")
    delete_files()

    t0 = time.time()
    print(python + command + join_path(data_path, 'RS-4.tgz'))
    result = os.system(python + command + join_path(data_path, 'RS-4.tgz'))
    print('**  subtest 2 finished in ', time.time() - t0, ' seconds')
    assert result == 0, 'Test failed: rungeneric on one bi-objective algorithm.'
    run_latex_template("templateBIOBJarticle.tex")
    delete_files()

    if run_all_tests:
        t0 = time.time()
        print(time.asctime())
        result = os.system(python + command +
                           join_path(data_path, 'BIPOP-CMA-ES_hansen_noiseless.tgz') +
                           join_path(data_path, 'MCS_huyer_noiseless.tgz') +
                           join_path(data_path, 'NEWUOA_ros_noiseless.tgz') +
                           join_path(data_path, 'RANDOMSEARCH_auger_noiseless.tgz') +
                           join_path(data_path, 'BFGS_ros_noiseless.tgz'))
        print('**  subtest 3 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on many algorithms.'
        run_latex_template("templateBBOBmany.tex")
        delete_files()

        t0 = time.time()
        result = os.system(python + command + '--conv' +
                           join_path(data_path, 'SMAC-BBOB_hutter_noiseless.tgz') +
                           join_path(data_path, 'lmm-CMA-ES_auger_noiseless.tgz'))
        print('**  subtest 4 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on two algorithms with option --conv.'
        run_latex_template("templateBBOBcmp.tex")
        delete_files()

        t0 = time.time()
        result = os.system(python + command + ' --omit-single ' +
                           join_path(data_path, 'DE-PSO_garcia-nieto_noiseless.tgz') +
                           join_path(data_path, 'VNS_garcia-martinez_noiseless.tgz'))
        print('**  subtest 5 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on two algorithms with option --omit-single.'
        run_latex_template("templateBBOBcmp.tex")
        delete_files()

        t0 = time.time()
        result = os.system(python + command + ' --expensive ' +
                           join_path(data_path, 'VNS_garcia-martinez_noiseless.tgz'))
        print('**  subtest 6 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on one algorithm with option --expensive.'
        run_latex_template("templateBBOBarticle.tex")
        delete_files()

        t0 = time.time()
        result = os.system(python + command + ' --omit-single ' +
                           join_path(data_path, 'RS-4.tgz') +
                           join_path(data_path, 'RS-100.tgz'))
        print('**  subtest 7 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on two bbob-biobj algorithms.'
        run_latex_template("templateBIOBJmultiple.tex")
        delete_files()

        t0 = time.time()
        # Note: we use the original GA-MULTIOBJ-NSGA-II.tgz data set
        # but with a shorter file name from the biobj-test folder
        # to avoid problems with too long path names on the windows
        # Jenkins slave
        result = os.system(python + command + ' --omit-single ' +
                           join_path(data_path, 'N-II.tgz') +
                           join_path(data_path, 'RS-4.tgz') +
                           join_path(data_path, 'RS-100.tgz'))
        print('**  subtest 8 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on three bbob-biobj algorithms.'
        run_latex_template("templateBIOBJmultiple.tex")
        delete_files()

        # testing data from bbob-noisy suite:
        t0 = time.time()
        result = os.system(python + command + ' --omit-single ' +
                           join_path(data_path, 'MCS_huyer_noisy.tgz') +
                           join_path(data_path, 'BFGS_ros_noisy.tgz'))
        print('**  subtest 9 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on two bbob-noisy algorithms.'
        # TODO: include noisy LaTeX templates into github repository and add test:
        # run_latex_template("templateBBOBnoisy.tex")
        delete_files()

        # testing data from recent runs, prepared in do.py:
        recent_data_path = os.path.abspath(join_path(os.path.dirname(__file__),
                                                     '../../code-experiments/build/python/exdata'))
        t0 = time.time()
        result = os.system(python + command +
                           join_path(recent_data_path, 'RS-bb'))
        print('**  subtest 10 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on newly generated random search data on `bbob`.'
        delete_files()

        t0 = time.time()
        result = os.system(python + command +
                           join_path(recent_data_path, 'RS-bi'))
        print('**  subtest 11 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on newly generated random search data on `bbob-biobj`.'
        delete_files()

        t0 = time.time()
        result = os.system(python + command +
                           join_path(recent_data_path, 'RS-co'))
        print('**  subtest 12 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on newly generated random search data on `bbob-constrained`.'
        delete_files(all_files=True)

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
        sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
        import cocopp as bb
        print(dir(bb))
        for s in dir(bb):
            if(inspect.ismodule(eval("bb."+s)) and s[:2]!="__"):
                print("bb."+s)
                doctest.testmod(eval("bb."+s),verbose=False)
        print(bb.__all__)
"""

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) else []
    path = os.path.split(sys.argv[0])[0]
    sys.path.append(os.path.join(path, os.path.pardir))  # needed in do.py
    main(args)
