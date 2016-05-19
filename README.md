numbbo/coco: Comparing Continuous Optimizers
============================================

[This code](https://github.com/numbbo/coco) reimplements the original Comparing Continous Optimizer platform, 
now rewritten fully in `ANSI C` with other languages calling the `C` code. As the name suggests,
the code provides a platform to benchmark and compare continuous optimizers, AKA non-linear 
solvers for numerical optimization. Languages currently available are 

  - `C/C++`
  - `Java`
  - `MATLAB/Octave`
  - `Python`

Contributions to link further languages (including a better
example in `C++`) are more than welcome.

For more information, 
- read our [benchmarking guidelines introduction](http://numbbo.github.io/coco-doc/)
- read the [COCO experimental setup](http://numbbo.github.io/coco-doc/experimental-setup) description
- see the [`bbob-biobj` COCO multi-objective functions testbed](http://numbbo.github.io/coco-doc/bbob-biobj/functions) documentation and the [specificities of the performance assessment for the bi-objective testbed](http://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment).
- consult the [BBOB workshops series](http://numbbo.github.io/workshops),
- consider to [register here](http://numbbo.github.io/register) for news, 
- see the [previous COCO home page here](http://coco.gforge.inria.fr/) and 
- see the [links below](#Links) to learn more about the ideas behind CoCO.

Requirements  <a name="Requirements"></a>
------------
1. For a machine running experiments 
  - A `C` compiler, such as gcc
  - Python >=2.6 with `setuptools` installed
  - optional: `git`
2. For a machine running the post-processing
  - Python 2.6 or 2.7 with `numpy` (preferably >=1.7) and `matplotlib` installed. 
    We recommend to install the [Anaconda Python 2.7 library](https://www.continuum.io/downloads). 
    Python 3 is not yet supported with the post-processing part of NumBBO/CoCO!

### Windows Specifics
Under Windows, two alternative compile toolchains can be used: 

1. [Cygwin](https://www.cygwin.com/) which comes with gcc and make, available in 32- and 64-bit versions.  
2. MinGW's gcc (http://www.mingw.org/), which only comes in 32-bit, but also runs on 64-bit machines. 

For using `git` under Windows (optional), we recommend installing [TortoiseGit](https://tortoisegit.org/).

### Language Specifics  <a name="Language-Specifics"></a>
_Additional_ requirements for running an algorithm in a specific language.

* **C**: make, such as GNU make ([GNU make for Windows](http://gnuwin32.sourceforge.net/packages/make.htm)). 
* **Java**: `gcc` and any Java Development Kit (JDK), such that `javac` and `javah` are accessible 
  (i.e. in the system path). 
* **MATLAB**: at least MATLAB 2008, for details, see [here](./code-experiments/build/matlab/README.md)
* **Python on Windows with MinGW**: Python 2.7 and the Microsoft compiler package for Python 2.7 
  containing VC9, available [here](https://www.microsoft.com/en-us/download/details.aspx?id=44266). 
  These are necessary to build the C extensions for the Python `cocoex` module for Windows. 
  The package contains 32-bit and 64-bit compilers and the Windows SDK headers.
* **Pyhon on Linux**: `python-dev` must be installed to compile/install the `cocoex` module.
* **Octave**: Octave 4.0.0 or later. On operating systems other than Windows, earlier versions might work.
  Under Linux the package `liboctave-dev` might be necessary. 

### Guaranties (None)
We tested the framework on Mac OSX, Ubuntu linux, Fedora linux, and Windows (XP,
7, 10) in various combinations of 32-bit and 64-bit compilers, python versions
etc. Naturally, we cannot guarantee that the framework runs on any combination
of operating system and software installed. In case you experience some incompatibilies,
check out the [_Known Issues / Trouble Shooting_ Section](#Known-Issues) below. 
Otherwise we will be happy if you can document them in detail on the 
[issue tracker](https://github.com/numbbo/coco/issues). 


Getting Started <a name="Getting-Started"></a>
---------------
0. Check out the [_Requirements_](#Requirements) above.

1. **Download** the COCO framework code from github, 

  - either by clicking the [Download ZIP button](https://github.com/numbbo/coco/archive/master.zip) 
    and unzip the `zip` file, 
  - or (preferred) by typing `git clone https://github.com/numbbo/coco.git`. This way 
    allows to remain up-to-date easily (but needs `git` to be installed). After 
    cloning, `git pull` keeps the code up-to-date with the latest release. 

  **CAVEAT: this code is still under heavy development**. The record of official releases can 
  be found [here](https://github.com/numbbo/coco/releases). The latest release corresponds 
  to the [master branch](https://github.com/numbbo/coco/tree/master) as linked above. 

2. In a system shell, **`cd` into** the `coco` or `coco-<version>` folder (framework root), 
  where the file `do.py` can be found. Type, i.e. **execute**, one of the following commands once
  ```
    python do.py run-c
    python do.py run-java
    python do.py run-matlab
    python do.py run-octave
    python do.py run-python
  ```  
  depending on which language shall be used to run the experiments. `run-*` will build the 
  respective code and run the example experiment once. The build result and the example
  experiment code can be found under `code-experiments/build/<language>` (`<language>=matlab` 
  for Octave). `python do.py` lists all available commands. 
  
3. On the computer where experiment data shall be post-processed, run
  ```
    python do.py install-postprocessing
  ```
  to (user-locally) install the post-processing. From here on, `do.py` has done
  its job and is only needed again for updating the builds to a new release.
  
  
4. **Copy** the folder `code-experiments/build/YOUR-FAVORITE-LANGUAGE` and
  its content to another location. In Python it is sufficient to copy the 
  file `example_experiment.py`. Run the example experiment (it already is
  compiled, in case). As the details vary, see the respective read-me's 
  and/or example experiment files:

  - `C` [read me](./coco-experiments/build/c/README.txt) 
    and [example experiment](./code-experiments/build/c/example_experiment.c)
  - `Java` [read me](./code-experiments/build/java/README.md)
    and [example experiment](./code-experiments/build/java/ExampleExperiment.java)
  - `Matlab/Octave` [read me](./code-experiments/build/matlab/README.md)
    and [example experiment](./code-experiments/build/matlab/exampleexperiment.m) 
  - `Python` [read me](./code-experiments/build/python/README.md)
    and [example experiment`](./code-experiments/build/python/example_experiment.py)

  If the example experiment runs, **connect** your favorite algorithm
  to Coco: replace the call to the random search optimizer in the
  example experiment file by a call to your algorithm (see above).
  **Update** the output `result_folder`, the `algorithm_name` and `algorithm_info` 
  of the observer options in the example experiment file.

  Another entry point for your own experiments can be the `code-experiments/examples`
  folder. 

5. Now you can **run** your favorite algorithm on the `bbob-biobj` (for
  multi-objective algorithms) or on the `bbob` suite (for single-objective
  algorithms). Output is automatically generated in the specified data 
  `result_folder`.

  <a name="Getting-Started-pp"></a>
6.  **Postprocess** the data from the results folder by typing

    ```
    python -m bbob_pproc [-o OUTPUT_FOLDERNAME] YOURDATAFOLDER [MORE_DATAFOLDERS]
    ```

  The name `bbob_pproc` will become `cocopp` in future. Any subfolder in the
  folder arguments will be searched for logged data. That is, experiments from
  different batches can be in different folders collected under a single "root" 
  `YOURDATAFOLDER` folder. We can also compare more than one algorithm by specifying
  several data result folders generated by different algorithms.
  
  A folder, `ppdata` by default, will be generated, which contains all output from 
  the post-processing, including a `ppdata.html` file, useful as main entry point to 
  explore the result with a browser. Data might be overwritten, 
  it is therefore useful to change the output folder name with the `-o OUTPUT_FOLDERNAME` 
  option.

  For the single-objective `bbob` suite, a summary pdf can be produced via 
  LaTeX. The corresponding templates in ACM format can be found in the
  `code-postprocessing/latex-templates` folder. LaTeX templates for the
  multi-objective `bbob-biobj` suite will follow in a later release. A basic
  html output is also available in the result folder of the postprocessing
  (file `templateBBOBarticle.html`).

7. Once your algorithm runs well, **increase the budget** in your experiment
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
     care of by `do.py` and/or `make`) and run typically random search on (some of)
     the provided benchmark suites.

* documentation and examples might not be too meaningful for the time being,
  even though code-experiments/documentation/onion.py describes a (heavily) used
  design pattern (namely: inheritance) in a comparatively understandable way
  (though the implementation in C naturally looks somewhat different). In the
  future, documentation will be contained mainly in the docs/ subfolder with the
  source code extracted automatically into pdfs in this folder and to web pages
  under the numbbo.github.io/ domain, see also
  [here](https://github.com/numbbo/coco#links-and-documentation-).

* the code-experiments/src folder is where most of the important/interesting
  things happen. Many files provide comparatively decent documentation at the
  moment which are translated via doxygen into a more readable web page at
  http://numbbo.github.io/coco-doc/C/. Generally:
  - [coco.h](./code-experiments/src/coco.h) is the public interface, in particular
    as used in the example_experiment.c file
  - coco_internal.h provides the type definition of coco_problem_t
  - coco_suite.c is code that deals with an entire benchmark suite (i.e. a set of
    functions, eg. sweeping through them etc...)
  - coco_generics.c is somewhat generic code, e.g. defining a function call via
    coco_evaluate_function etc
  - coco_problem.c is the implementation of the coco_problem_t type/object
    (allocation etc).
  - observer / logger files implement data logging (as wrappers around a coco
    problem inheriting thereby all properties of a coco problem)
  - most other files implement more or less what they say, e.g. the actual
    benchmark functions, transformations, benchmark suites, etc.
  - currently, three benchmark suites and corresponding logging facilities are
    implemented:
    * `bbob`: standard single-objective BBOB benchmark suite with 24 noiseless,
      scalable test functions
    * `bbob-biobj`: a bi-objective benchmark suite, combining 10 selected
      functions from the bbob suite, resulting in 55 noiseless functions
    * `toy`: a simple, probably easier-to-understand example for reading and testing

* code-experiments/tools are a few meta-tools, mainly the amalgamate.py to merge all
  the C code into one file

* code-experiments/test contains unit- and integration-tests, mainly for internal use

* code-postprocessing/bbob_pproc contains the postprocessing code, written in
  python, with which algorithm data sets can be read in and the performance of
  the algorithms can be displayed in terms of data profiles, aRT vs. dimension
  plots, or simple tables.

* code-postprocessing/latex-templates contains LaTeX templates for displaying
  algorithm performances in publisher-conform PDFs for the GECCO
  conference.

* code-preprocessing/archive-update/ contains internal code for combining
  the archives of algorithms to create/update the hypervolume
  reference values for the `bbob-biobj` test suite

* docs should contain an updated version of the documentation, see above.

* howtos contains a few text files with internal howtos.


Known Issues / Trouble-Shooting <a name="Known-Issues"></a>
-------------------------------
### Java
#### `javah` call fails
If you see something like this when running `python do.py run-java` or `build-java`
under Linux
```
COPY    code-experiments/src/coco.h -> code-experiments/build/java/coco.h
WRITE   code-experiments/build/java/REVISION
WRITE   code-experiments/build/java/VERSION
RUN     javac CocoJNI.java in code-experiments/build/java
RUN     javah CocoJNI in code-experiments/build/java
Traceback (most recent call last):
  File "do.py", line 590, in <module>
    main(sys.argv[1:])
  File "do.py", line 563, in main
    elif cmd == 'build-java': build_java()
  File "do.py", line 437, in build_java
    env = os.environ, universal_newlines = True)
  File "/..../code-experiments/tools/cocoutils.py", line 34, in check_output
    raise error
subprocess.CalledProcessError: Command '['locate', 'jni.h']' returned non-zero exit status 1
```
it means `javah` is either not installed (see above) or cannot be found in the system
path, see [this](http://stackoverflow.com/questions/13526701/javah-missing-after-jdk-install-linux)
and possibly [this](https://github.com/numbbo/coco/issues/416) for a solution. 

### Matlab

#### Path to matlab
If you see something like this when running `python do.py build-matlab`
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
it might be because your system does not know the `matlab` command. To fix this,
you should edit the file `/etc/paths` and add the path to the `matlab` bin file 
(Linux/Mac) or add the path to the folder where the `matlab.exe` lies to your 
Windows path. For instance, the `etc/paths` should look like something like this
```
/usr/local/bin
/usr/bin
/bin
/usr/sbin
/sbin
/Applications/MATLAB_R2012a.app/bin/
```

#### SMS-EMOA example does not compile under Mac 
With the more complex SMS-EMOA example, the problem is related to the compilation
of the external C++ hypervolume calculation in `hv.cpp`. 

A fix for this issue consists in adding to the files `hv.cpp` and `paretofront.c`
```
#define char16_t UINT16_T
```
just before the line:
```
#include "mex.h"
```

#### Access to mex files denied
If it happens that you get some `Access is denied` errors during
`python do.py build-matlab` or `python do.py run-matlab` like this one
```
C:\Users\dimo\Desktop\numbbo-brockho>python do.py run-matlab
Traceback (most recent call last):
  File "do.py", line 649, in <module>
    main(sys.argv[1:])
  File "do.py", line 630, in main
    elif cmd == 'run-matlab': run_matlab()
  File "do.py", line 312, in run_matlab
    os.remove( filename )
WindowsError: [Error 5] Access is denied: 'code-experiments/build/matlab\\cocoEv
aluateFunction.mexw32'
```
a reason can be that a previously opened Matlab window still has some
file handles open. Simply close all Matlab windows (and all running Matlab
processes if there is any) before to run the `do.py` command again.


### Octave

#### `octave-dev` under Linux
When running 
```
  python do.py run-octave
```
or 
```
  python do.py build-octave
```
and seeing something like
```
   [...]
   compiling cocoCall.c...error: mkoctfile: please install the Debian package "liboctave-dev" to get the mkoctfile command
```
then, unsurprisingly, installing `liboctave-dev` like
```
  sudo apt-get install liboctave-dev
```
should do the job. 


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
or
```
$ python do.py run-python  # or build-python
[...]
cython/interface.c -o build/temp.linux-x86_64-2.7/cython/interface.o
cython/interface.c:4:20: fatal error: Python.h: No such file or directory
#include "Python.h"
^
compilation terminated.
error: command 'x86_64-linux-gnu-gcc' failed with exit status 1
```
Under Linux
```
  sudo apt-get install python-dev
```
should do the trick. 

#### Module Update/Install Does Not Propagate
We have observed a case where the update of the `cocoex` Python module seemed to have no 
effect. In this case it has been successful to remove all previously installed versions, 
see [here](https://github.com/numbbo/coco/issues/586) for a few more details. 


#### Too long paths for postprocessing
It can happen that the postprocessing fails due to too long paths to the algorithm data.
Unfortunately, the error you get in this case does not indicate directly to the problem
but only tells that a certain file could not be read. Please try to shorten the
folder names in such a case.


Details
-------
- The C code features an object oriented implementation, where the
  `coco_problem_t` is the most central data structure / object. `coco.h`,
  `example_experiment.c` and `coco_internal.h` are probably the best pointers to
  start __to investigate the code__ (but see also 
  [here](https://numbbo.github.io/coco-doc/C/annotated.html)). `coco_problem_t`
  defines a benchmark function instance (in a given dimension), and is called
  via `coco_evaluate_function`.

- Building, running, and testing of the code is done by merging/amalgamation of
  all C-code into a single C file, `coco.c`, and `coco.h`. (by calling `do.py`,
  see above). Like this it becomes very simple to include/use the code in
  different projects.

- [Cython](http://docs.cython.org/src/quickstart/install.html) is used to
  compile the C to Python interface in `build/python/interface.pyx`. The Python
  module installation file `setup.py` uses the compiled `interface.c`, if
  `interface.pyx` has not changed. For this reason, Cython is not a requirement
  for the end-user.

- We continuously test the code through the open source automation server
  Jenkins on one ubuntu 12.04 machine, one OSX 10.9 machine, and one 32-bit
  Windows 7 machine with cygwin.


Links and Documentation <a name="Links"></a>
-----------------------
* The [_BBOB workshop series_](http://numbbo.github.io/workshops), which uses the
  NumBBO/Coco framework extensively, can be tracked
  [here](http://numbbo.github.io/workshops "BBOB Workshops")
* Stay informed about the BBOB workshop series and releases of the NumBBO/Coco software 
  by registering at http://coco.gforge.inria.fr/register
* Read about the basic principles behind the Coco platform at http://numbbo.github.io/coco-doc/.
* Please refer to http://numbbo.github.io/coco-doc/experimental-setup/ for more details on the
  experimental set-up for black-box optimization benchmarking.
* More detailed documentation of the existing benchmark suites can be found here:
  - for the **`BBOB`** problem suite at http://coco.lri.fr/downloads/download15.03/bbobdocfunctions.pdf 
    with the experimental setup at http://coco.lri.fr/downloads/download15.03/bbobdocexperiment.pdf
  - for the **`bbob-biobj`** problem suite at http://numbbo.github.io/coco-doc/bbob-biobj/functions
* Online documentation of the NumBBO/Coco API (i.e. for the ANSI C code) is available at 
  http://numbbo.github.io/coco-doc/C
* Downloading this repository 
  - via the above "Download ZIP" button or 
  - by typing `git clone https://github.com/numbbo/coco.git` or 
  - via https://github.com/numbbo/coco/archive/master.zip in your browser
