numbbo/coco: Comparing Continuous Optimizers
============================================

This code reimplements the original Comparing Continous Optimizer platform (http://coco.gforge.inria.fr/),
now rewritten fully in ANSI C with the other languages calling the C code. Languages currently available 
are C, Java, MATLAB, and Python. Languages available in near future are C++ and Octave. Contributions to 
link further languages are more than welcome.

Requirements
------------
1. For a machine running experiments 
  - A `C` compiler, such as gcc
  - make, such as GNU make
  - Python >=2.6 with `setuptools` installed
2. For a machine running the post-processing
  - Python 2.6 or 2.7 with `numpy` (preferably >=1.7) and `matplotlib` installed. 
    We recommend to install the [Anaconda library](https://www.continuum.io/downloads). 
    Python 3 is not yet supported with the post-processing part of NumBBO/CoCO!

### Windows Specifics
Under Windows, two alternative compile toolchains can be used: 

1. [Cygwin](https://www.cygwin.com/) which comes with gcc and make, available in 32- and 64-bit versions.  
2. MinGW's gcc (http://www.mingw.org/) and GNU make (http://gnuwin32.sourceforge.net/packages/make.htm).
  MinGW only comes in 32-bit, but also runs on 64-bit machines. 

For using `git` under Windows (optional), we recommend installing [TortoiseGit](https://tortoisegit.org/).

### Language Specifics
Additional requirements for running an algorithm in a specific language.

* Java: none, but see [here](./code-experiments/build/java/README.txt) for details on the compilation
* Python: none, see [here](./code-experiments/build/python/README.md) for details on the installation
* MATLAB: at least MATLAB 2008, for details, see [here](./code-experiments/build/matlab/README.txt)

### Guaranties (None)
We tested the framework on Mac OSX, Ubuntu linux, Fedora linux, and Windows (XP,
7, 10) in various combinations of 32-bit and 64-bit compilers, python versions
etc. Naturally, we cannot guarantee that the framework runs on any combination
of operating system and software installed. In case you experience some incompatibilies,
we will be happy if you can document them in detail on our [issue tracker](https://github.com/numbbo/coco/issues). 

Getting Started
---------------
**Download** the [COCO framework code](https://github.com/numbbo/coco) from
github by clicking [here](https://github.com/numbbo/coco/archive/master.zip), 
**CAVEAT: this code is still under development**, and unzip the `zip` file. 

1. In a system shell, **`cd` into** the `numbbo` (framework root) folder, where the 
   file `do.py` lies. Type, i.e. **execute**, one of the following commands once
  ```
    python do.py run-c
    python do.py run-java
    python do.py run-matlab
    python do.py run-python
  ```  
  depending which language is used to run the experiments. `run-*` will build the 
  respective code and run the example experiment once. The build result and the example
  experiment code can be found under `code-experiments/build/*`. 
  
2. On the computer where experiment data shall be post-processed, run
  ```
    python do.py install-postprocessing
  ```
  to (user-locally) install the post-processing. 
  
3. If the example experiment runs, **connect** your favorite algorithm
  to Coco: copy the `code-experiments/build/YOUR-FAVORITE-LANGUAGE` folder to
  another location. Replace the call to the random search optimizer in the
  example experiment file by a call to your algorithm (the details vary), see
  respective the read-me's and example experiment files:

  - `C` [read me](https://github.com/numbbo/coco/blob/master/coco-experiments/build/c/README.txt) 
    and [example experiment](https://github.com/numbbo/coco/blob/development/code-experiments/build/c/example_experiment.c)
  - `Java` [read me](https://github.com/numbbo/coco/blob/master/code-experiments/build/java/README.txt)
    and [example experiment](https://github.com/numbbo/coco/blob/master/code-experiments/build/java/ExampleExperiment.java)
  - `Matlab` [read me](https://github.com/numbbo/coco/blob/master/code-experiments/build/matlab/README.txt)
    and [example experiment](https://github.com/numbbo/coco/blob/master/code-experiments/build/matlab/exampleexperiment.m) 
  - `Python` [read me](https://github.com/numbbo/coco/blob/master/code-experiments/build/python/README.md)
    and [example experiment`](https://github.com/numbbo/coco/blob/master/code-experiments/build/python/example_experiment.py)

  Another entry point for your own experiments can be
  the `code-experiments/examples` folder. In any case, update the output
  (observer options) result_folder and the algorithm_name and info in the
  experiment file.

4. Now you can **run** your favorite algorithm on the `bbob-biobj` (for
  multi-objective algorithms) or on the `bbob` suite (for single-objective
  algorithms). Output is automatically generated in the specified data 
  `result_folder`.

5. **Postprocess** that data from the results folder by typing

    ```
    python -m bbob_pproc YOURDATAFOLDER [MORE_FOLDERS]
    ```

  The name `bbob_pproc` will become `cocopp` in future. Any subfolder in the
  folder arguments will be search for logged data. That is, experiments from
  different batches can be in different folders collected under a single "root" 
  folder. We can also compare more than one algorithm by specifying
  several data result folders generated by different algorithms.
  
  A folder named `ppdata` by default will be generated (the folder name can be
  changed by the `-o FOLDERNAME` option), which contains all output from the 
  post-processing. 

  Within the postprocessing's output folder, you will find pdfs of all kinds
  of plots (e.g. data profiles). With the `--svg` option, figures for the
  `template*.html` file are generated, which can be explored in a browser. 
  
  For the single-objective `bbob` suite, a summary pdf can be produced via 
  LaTeX. The corresponding templates in ACM format can be found in the
  `code-postprocessing/latex-templates` folder. LaTeX templates for the
  multi-objective `bbob-biobj` suite will follow in a later release. A basic
  html output is also available in the result folder of the postprocessing
  (file `templateBBOBarticle.html`).

6. Once your algorithm runs well, **increase the budget** in your experiment
  script, if necessary implement randomized independent restarts, and follow 
  the above steps successively until you are happy.

If you detect bugs or other issues, please let us know by opening an issue in
our issue tracker at https://github.com/numbbo/coco/issues.

## Description by Folder

* the `do.py` file in the root folder is a tool to build the entire
  distribution. `do.py` is a neat and simplifying replacement for make. It has
  switches for just building some languages etc, type
  ```
    python do.py
  ```
  to see a list of all available commandes. 

* the code-experiments/build folder is to a large extend the output folder of
  the `./do.py build` command.
   - the `exampleexperiment.???` files in the build folder are the entry points to
     understand the usage of the code (as end-user). They are supposed to
     actually be executable (in case, after compilation, which should be taken
     care of by do.py and/or make) and run typically random search on (some of)
     the provided benchmark suites.

* documentation and examples might not be too meaningful for the time being,
  even though code-experiments/documentation/onion.py describes a (heavily) used
  design pattern (namely: inheritance) in a comparatively understandable way
  (though the implementation in C naturally looks somewhat different). In the
  future, documentation will be contained mainly in the docs/ subfolder with the
  source code extracted automatically into pdfs in this folder and to web pages
  under the numbbo.github.io/ domain.

* the code-experiments/src folder is where most of the important/interesting
  things happen. Many files provide comparatively decent documentation at the
  moment which are translated via doxygen into a more readable web page at
  numbbo.github.io/coco-doc/C/. Generally:
  - coco.h is the public interface, in particular as used in the demo.c file, however check out https://code.google.com/p/numbbo/issues/detail?id=98
  - coco_internal.h provides the type definition of coco_problem_t
  - coco_suite.c is code that deals with an entire benchmark suite (i.e. a set of functions, eg. sweeping through them etc...)
  - coco_generics.c is somewhat generic code, e.g. defining a function call via coco_evaluate_function etc
  - coco_problem.c is the implementation of the coco_problem_t type/object (allocation etc).
  - observer / logger files implement data logging (as wrappers around a coco
    problem inheriting thereby all properties of a coco problem)
  - most other files implement more or less what they say, e.g. the actual
    benchmark functions, transformations, benchmark suites, etc.
  - currently, three benchmark suites and corresponding logging facilities are implemented:
    * bbob: standard single-objective BBOB benchmark suite with 24 noiseless, scalable test functions
    * bbob-biobj: a bi-objective benchmark suite, combining 10 selected
      functions from the bbob suite, resulting in 55 noiseless functions
    * toy: a simple, probably easier-to-understand example for reading and testing

* code-experiments/tools are a few meta-tools, mainly the amalgamate.py to merge all the C code into one file

* code-experiments/test contains unit- and integration-tests, mainly for internal use

* code-postprocessing/bbob_pproc contains the postprocessing code, written in
  python, with which algorithm data sets can be read in and the performance of
  the algorithms can be displayed in terms of data profiles, ERT vs. dimension
  plots, or simple tables.

* code-postprocessing/latex-templates contains LaTeX templates for displaying
  algorithm performances in publisher-conform PDFs for the GECCO and CEC
  conferences (for the single-objective bbob suite only, templates for the
  bi-objective bbob-biobj suite will be provided in a later release).

* docs should contain an updated version of the documentation, see above.

* howtos contains a few text files with internal howtos.


Known Issues / Trouble-Shooting
-------------------------------
### Python

#### `setuptools` is not installed
If you see something like this
```
$ python do.py run-python  # or build-python
[...]
PYTHON  setup.py install --user in code-experiments/build/python
ERROR: return value=1
Traceback (most recent call last):
 File "setup.py", line 8, in <module>
   import setuptools
ImportError: No module named setuptools

Traceback (most recent call last):
 File "do.py", line 562, in <module>
   main(sys.argv[1:])
 File "do.py", line 539, in main
   elif cmd == 'build-python': build_python()
 File "do.py", line 203, in build_python
   python('code-experiments/build/python', ['setup.py', 'install', '--user'])
 File "/vol2/twagner/numbbo/code-experiments/tools/cocoutils.py", line 92, in p                                         ython
   universal_newlines=True)
 File "/usr/local/lib/python2.7/subprocess.py", line 575, in check_output
   raise CalledProcessError(retcode, cmd, output=output)
subprocess.CalledProcessError: Command '['/usr/local/bin/python', 'setup.py', 'i                                        nstall', '--user']' returned non-zero exit status 1
```
then `setuptools` needs to be installed: 
```
    pip install setuptools
```
or `easy_install setuptools` should do the job. 

#### Compilation During Install of `cocoex` Fails (under Linux) 
If you see something like this:
``` 
$ python do.py run-python  # or build-python
[...]
cython/interface.c -o build/temp.linux-i686-2.6/cython/interface.o
cython/interface.c:4:20: error: Python.h: file not found
cython/interface.c:6:6: error: #error Python headers needed to compile C extensions, please install development version of Python.
error: command 'gcc' failed with exit status 1
```
Under Linux
```
  sudo apt-get install python-dev
```
should do the trick. 


### Matlab

#### `build-matlab` crashes under Linux
The Matlab wrapper does not always work under Linux with the current code: an issue is filed for the Ubuntu operating system at https://github.com/numbbo/coco/issues/318
### Path to matlab
If you see something like this when running ``python do.py build-matlab``
```
AML	['code-experiments/src/coco_generics.c', 'code-experiments/src/coco_random.c', 'code-experiments/src/coco_suite.c', 'code-experiments/src/coco_suites.c', 'code-experiments/src/coco_observer.c', 'code-experiments/src/coco_runtime_c.c'] -> code-experiments/build/matlab/coco.c
COPY	code-experiments/src/coco.h -> code-experiments/build/matlab/coco.h
COPY	code-experiments/src/best_values_hyp.txt -> code-experiments/build/matlab/best_values_hyp.txt
WRITE	code-experiments/build/matlab/REVISION
WRITE	code-experiments/build/matlab/VERSION
RUN	matlab -nodisplay -nosplash -r setup, exit in code-experiments/build/matlab
Traceback (most recent call last):
  File "do.py", line 447, in <module>
    main(sys.argv[1:])
  File "do.py", line 429, in main
    elif cmd == 'build-matlab': build_matlab()
  File "do.py", line 278, in build_matlab
    run('code-experiments/build/matlab', ['matlab', '-nodisplay', '-nosplash', '-r', 'setup, exit'])
  File "/Users/auger/workviasvn/newcoco/numbbo/code-experiments/tools/cocoutils.py", line 68, in run
    universal_newlines=True)
  File "//anaconda/lib/python2.7/subprocess.py", line 566, in check_output
    process = Popen(stdout=PIPE, *popenargs, **kwargs)
  File "//anaconda/lib/python2.7/subprocess.py", line 710, in __init__
    errread, errwrite)
  File "//anaconda/lib/python2.7/subprocess.py", line 1335, in _execute_child
    raise child_exception
OSError: [Errno 2] No such file or directory
```
It might be because your system does not know the ``matlab`` command. To fix this you should edit the file ``/etc/paths`` and add the path to the ``matlab`` bin file. For instance the ``etc/paths`` should look like something like this
```
/usr/local/bin
/usr/bin
/bin
/usr/sbin
/sbin
/Applications/MATLAB_R2012a.app/bin/
```

#### SMA-EMOA example does not compile under Mac 
With the more complex SMS-EMOA example. The problem is related to the compilation of the external C++ hypervolume calculation in hv.cpp. 

A fix for this issue consists in adding to the files "hv.cpp" and "paretofront.c"  
`#define char16_t UINT16_T`
just before the line:
`#include "mex.h"`

### Octave

#### Command Window Closes Unexpectedly Under Windows
If it happens that the command window, from which the `python do.py run-octave` is run, closes unexpectely under Windows, you might want to change the general way, Octave is called. Find your `octave.bat` file, which is in your Octave installation folder (typically something like `C:\Octave\Octave-4.0.0\` and remove or outcomment the last line, saying
```
Rem   Close the batch file's cmd.exe window
exit
```
We think already about a way to solve this issue directly in the `do.py` but it has low priority for the moment.


Details
-------
- The C code features an object oriented implementation, where the
  `coco_problem_t` is the most central data structure / object. `coco.h`,
  `example_experiment.c` and `coco_internal.h` are probably the best pointers to
  start __to investigate the code__ (but see also below). `coco_problem_t`
  defines a benchmark function instance (in a given dimension), and is called
  via `coco_evaluate_function`.

- Building, running, and testing of the code is done by merging/amalgamation of
  all C-code into a single C file, `coco.c`, and `coco.h`. (by calling `do.py`,
  see above). Like this it becomes very simple to include/use the code in
  different projects.

- [Cython](http://docs.cython.org/src/quickstart/install.html) is used to
  compile the C interface in `build/python/interface.pyx`. The Python module
  installation file `setup.py` uses the compiled `interface.c`, if
  `interface.pyx` has not changed. 

- We continuously test the code through the open source automation server
  Jenkins on one ubuntu 12.04 machine, one OSX 10.9 machine, and one 32-bit
  Windows 7 machine with cygwin.


Links and Further Documentation
-------------------------------
* The [_BBOB workshop series_](http://numbbo.github.io/workshops), which uses the
  NumBBO/Coco framework extensively, can be tracked at 
  [here](http://numbbo.github.io/workshops "BBOB Workshops")
* Stay informed about the BBOB workshop series and releases of the NumBBO/Coco software 
  by registering at http://coco.gforge.inria.fr/register
* More detailed documentation of the existing benchmark suites can be found here:
  - for the "**BBOB**" testbed at http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf 
    with the experimental setup at http://coco.lri.fr/downloads/download15.03/bbobdocexperiment.pdf
  - for the **bbob-biobj** functions at http://numbbo.github.io/bbob-biobj-functions-doc
* Online documentation of the NumBBO/Coco API (i.e. for the ANSI C code) is available at 
  http://numbbo.github.io/coco-doc/C
* Downloading this repository 
  - via the above "Download ZIP" button or 
  - by typing `git clone https://github.com/numbbo/coco.git` or 
  - via https://github.com/numbbo/coco/archive/master.zip in your browser
