#!/usr/bin/env python
"""``python cocopp`` tests the package cocopp and should run through
smoothly from a system command shell. It however depends on data files that
might not be available (to be improved).

This test can and should become much more sophisticated.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, time, inspect
import fnmatch
import urllib
import shutil
import subprocess
import doctest

try:
    from . import rungeneric

    is_module = True
except:
    is_module = False
import matplotlib  # just to make sure the following is actually done first

matplotlib.use('Agg')  # To avoid window popup and use without X forwarding

# depreciated, to be removed, see end of file
if 11 < 3 and __name__ == "__main__" and not is_module:
    """import cocopp as module and run tests or rungeneric.main"""
    args = sys.argv[1:] if len(sys.argv) else []
    filepath = os.path.split(sys.argv[0])[0]
    sys.path.append(os.path.join(os.getcwd(), filepath))  # needed from the shell
    sys.path.append(os.path.join(filepath, os.path.pardir))  # needed in do.py

    import cocopp
    # run either this main here as cocopp._main or rungeneric.main
    if len(args) == 0:
        print("WARNING: this tests the post-processing, this will change "
              + "in future (use -h for help)")
        cocopp._main(args)
    elif args[0] == '-t' or args[0].startswith('--t'):
        args.pop(0)
        cocopp._main(args)
    elif args[0] == 'all':
        print("WARNING: this tests the post-processing and doesn't run anything else")
        cocopp._main(args)
    else:
        cocopp.rungeneric.main(args)


def join_path(a, *p):
    path = os.path.join(a, *p)
    return path


def copy_latex_templates():
    currentFolder = os.path.dirname(os.path.realpath(__file__))
    templateFolder = os.path.abspath(join_path(currentFolder, '..', 'latex-templates'))
    # templateFolder = os.path.abspath('latex-templates')
    shutil.copy(join_path(templateFolder, 'templateBBOBarticle.tex'), currentFolder)
    shutil.copy(join_path(templateFolder, 'templateBBOBcmp.tex'), currentFolder)
    shutil.copy(join_path(templateFolder, 'templateBBOBmany.tex'), currentFolder)
    shutil.copy(join_path(templateFolder, 'templateBIOBJarticle.tex'), currentFolder)
    shutil.copy(join_path(templateFolder, 'templateBIOBJmultiple.tex'), currentFolder)
    shutil.copy(join_path(templateFolder, 'sig-alternate.cls'), currentFolder)
    shutil.copy(join_path(templateFolder, 'comment.sty'), currentFolder)
    shutil.copy(join_path(templateFolder, 'acmcopyright.sty'), currentFolder)
    shutil.copy(join_path(templateFolder, 'bbob.bib'), currentFolder)


def run_latex_template(filename):
    filePath = os.path.abspath(join_path(os.path.dirname(__file__), filename))
    args = ['pdflatex', filePath]
    DEVNULL = open(os.devnull, 'wb')
    result = subprocess.call(args, stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL)
    assert not result, 'Test failed: error while generating pdf from %s.' % filename

    # filePath = os.path.splitext(filePath)[0]
    # args = ['bibtex', filePath]
    # DEVNULL = open(os.devnull, 'wb')
    # output_file = open("bibtex.log", "w")
    # result = subprocess.call(args, stdin=DEVNULL, stdout=output_file, stderr=DEVNULL)
    # assert not result, 'Test failed: error while running bibtex on %s.' % os.path.splitext(filename)[0]


def retrieve_algorithm(dataPath, folderName, algorithmName, fileName=None):
    algorithmFile = join_path(dataPath, fileName if fileName else algorithmName)
    if not os.path.exists(algorithmFile):
        dataurl = 'http://coco.gforge.inria.fr/data-archive/%s/%s' % (folderName, algorithmName)
        urllib.urlretrieve(dataurl, algorithmFile)


def prepare_data(run_all_tests):
    print('preparing algorithm data')

    dataPath = os.path.abspath(join_path(os.path.dirname(__file__), 'data'))

    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    # Retrieving the algorithms
    # retrieve_algorithm(dataPath, '2010', 'IPOP-ACTCMA-ES_ros_noiseless.tar.gz')
    # [outcommented and replaced by BIPOP until 2010 data is in new format]
    retrieve_algorithm(dataPath, '2009', 'BFGS_ros_noiseless.tgz')
    retrieve_algorithm(dataPath, 'bbob-biobj/2016', 'RS-4.tgz')

    if run_all_tests:
        retrieve_algorithm(dataPath, '2009', 'BIPOP-CMA-ES_hansen_noiseless.tgz')
        retrieve_algorithm(dataPath, '2009', 'MCS_huyer_noiseless.tgz')
        retrieve_algorithm(dataPath, '2009', 'NEWUOA_ros_noiseless.tgz')
        retrieve_algorithm(dataPath, '2009', 'RANDOMSEARCH_auger_noiseless.tgz')
        retrieve_algorithm(dataPath, '2013', 'SMAC-BBOB_hutter_noiseless.tgz')
        retrieve_algorithm(dataPath, '2013', 'lmm-CMA-ES_auger_noiseless.tgz')
        retrieve_algorithm(dataPath, '2009', 'DE-PSO_garcia-nieto_noiseless.tgz')
        retrieve_algorithm(dataPath, '2009', 'VNS_garcia-martinez_noiseless.tgz')
        retrieve_algorithm(dataPath, 'bbob-biobj/2016', 'RS-4.tgz')
        retrieve_algorithm(dataPath, 'bbob-biobj/2016', 'RS-100.tgz')
        retrieve_algorithm(dataPath, 'test', 'N-II.tgz') # diff. location and name due to Jenkins settings with too long paths
        retrieve_algorithm(dataPath, '2009', 'BFGS_ros_noisy.tgz')
        retrieve_algorithm(dataPath, '2009', 'MCS_huyer_noisy.tgz')        

    return dataPath


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


def delete_files():
    shutil.rmtree('ppdata')


def main(args):
    """these tests are executed when ``python cocopp`` is called.

    with ``wine`` as second argument ``C:\\Python26\\python.exe``
    instead of ``python`` is called

    """

    run_all_tests = len(args) == 1 and args[0] == 'all'

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
        result = os.system(python + command +
                           join_path(data_path, 'MCS_huyer_noisy.tgz') +
                           join_path(data_path, 'BFGS_ros_noisy.tgz'))
        print('**  subtest 9 finished in ', time.time() - t0, ' seconds')
        assert result == 0, 'Test failed: rungeneric on two bbob-noisy algorithms.'
        # TODO: include noisy LaTeX templates into github repository and add test:
        #run_latex_template("templateBBOBnoisy.tex")
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


    print('launching doctest (it might be necessary to close a few pop up windows to finish)')
    t0 = time.time()

    if 1 < 3:
        failure_count = 0
        test_count = 0
        # doctest.testmod(report=True, verbose=True)  # this is quite cool!
        # go through the py files in the cocopp folder
        currentPath = os.getcwd()
        newPath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        os.chdir(newPath)
        for root, dirnames, filenames in os.walk(os.path.dirname(os.path.realpath(__file__))):
            for filename in fnmatch.filter(filenames, '*.py'):
                current_failure_count, current_test_count = doctest.testfile(
                    os.path.join(root, filename), report=True, module_relative=False)
                failure_count += current_failure_count
                test_count += current_test_count
                if current_failure_count:
                    print('doctest file "%s" failed' % os.path.join(root, filename))
        os.chdir(currentPath)
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
    """run either tests or rungeneric.main"""
    args = sys.argv[1:] if len(sys.argv) else []
    filepath = os.path.split(sys.argv[0])[0]
    # sys.path.append(os.path.join(os.getcwd(), filepath))  # tests from shell fail, but why?
    sys.path.append(os.path.join(filepath, os.path.pardir))  # needed in do.py
    # run either this main or rungeneric.main
    if len(args) == 0:
        if is_module:
            rungeneric.main(args)  # just prints help
        else:
            print("WARNING: this tests the post-processing, this might change "
                  + "in future (use -h for help)")
            main(args)
    elif args[0] == '-t' or args[0].startswith('--t'):
        args.pop(0)
        main(args)  # is not likely to work
    elif args[0] == 'all':
        print("WARNING: this tests the post-processing and doesn't run anything else")
        main(args)
    else:
        if not is_module:
            raise ValueError('try calling "python -m ..." instead of "python ..."')
        rungeneric.main(args)
