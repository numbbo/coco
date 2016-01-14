# numbbo: Numerical Black-Box Optimization Benchmarking Framework 

The code reimplements the original Comparing Continous Optimizer platform (http://coco.gforge.inria.fr/),
now rewritten fully in ANSI C with the other three languages Java, MATLAB, and python calling the C code.

Generally, the code features an object oriented implementation, where the coco_problem_t is the most central
data structure / object. coco.h, demo.c and coco_internal.h are probably the best pointers to start with
(but see also below). coco_problem_t defines a benchmark function instance (in a given dimension), and is 
called via coco_evaluate_function.

# Building the Code
Building, running, and testing of the code is done by merging/amalgamation of all C-code into a single C file, coco.c
(by calling do.py, see below). Like this it becomes very simple to include/use the code in different projects.

# Description by Folder

* the do.py file in the root folder is a tool to build the entire distribution (like a make file, maybe it should rather have been named make.py). It has switches for just building some languages etc, e.g.

    python ./do.py build  # builds all
    python ./do.py build-python
    python ./do.py build-c
    python ./do.py build-java
    python ./do.py build-matlab

are valid commands (on a Linux or OSX shell, you can omit the leading 'python' if you wish). Our do.py is a neat and simplifying replacement for make. It can also run the experiments directly, e.g., by typing

    python ./do.py run # runs all
    python ./do.py run-python
    python ./do.py run-c
    python ./do.py run-java
    python ./do.py run-matlab

you can start the corresponding example experiments scripts in code-experiments/build/LANGUAGE/.

* the code-experiments/build folder is to a large extend the output folder of the "./do.py build" command.
   - the exampleexperiment.??? files in the build folder are the entry points to understand the usage of the code (as end-user). They are supposed to actually be executable (in case, after compilation, which should be taken care of by do.py and/or make) and run typically random search on (some of) the provided benchmark suites.

* documentation and examples might not be too meaningful for the time being, even though code-experiments/documentation/onion.py describes a (heavily) used design pattern (namely: inheritance) in a comparatively understandable way (though the implementation in C naturally looks somewhat different). In the future, documentation will be contained mainly in the docs/ subfolder with the source code extracted automatically into pdfs in this folder and to web pages under the numbbo.github.io/ domain.

* the code-experiments/src folder is where most of the important/interesting things happen. Many files provide comparatively decent documentation at the moment which are translated via doxygen into a more readable web page at numbbo.github.io/COCOdoc/C/. Generally:
  - coco.h is the public interface, in particular as used in the demo.c file, however check out https://code.google.com/p/numbbo/issues/detail?id=98
  - coco_internal.h provides the type definition of coco_problem_t
  - coco_suite.c is code that deals with an entire benchmark suite (i.e. a set of functions, eg. sweeping through them etc...)
  - coco_generics.c is somewhat generic code, e.g. defining a function call via coco_evaluate_function etc
  - coco_problem.c is the implementation of the coco_problem_t type/object (allocation etc).
  - observer / logger files implement data logging (as wrappers around a coco problem inheriting thereby all properties of a coco problem)
  - most other files implement more or less what they say, e.g. the actual benchmark functions, transformations, benchmark suites, etc.
  - currently, three benchmark suites and corresponding logging facilities are implemented:
    * bbob: standard single-objective BBOB benchmark suite with 24 noiseless, scalable test functions
    * bbob-biobj: a bi-objective benchmark suite, combining 10 selected functions from the bbob suite, resulting in 55 noiseless functions
    * toy: a simple, probably easier-to-understand example for reading and testing

* code-experiments/tools are a few meta-tools, mainly the amalgamate.py to merge all the C code into one file

* code-experiments/test contains unit- and integration-tests, mainly for internal use

* code-postprocessing/bbob_pproc contains the postprocessing code, written in python, with which algorithm data sets can be read in and the performance of the algorithms can be displayed in terms of data profiles, ERT vs. dimension plots, or simple tables.

* code-postprocessing/latex-templates contains LaTeX templates for displaying algorithm performances in publisher-conform PDFs for the GECCO and CEC conferences (for the single-objective bbob suite only, templates for the bi-objective bbob-biobj suite will be provided in a later release).

* docs should contain an updated version of the documentation, see above.

* howtos contains a few text files with internal howtos.


# Requirements and Installation
Minimal requirements (running an algorithm in C):
* git
* Python 2.7 (or Python 3) with numpy and matplotlib
* a C compiler, such as gcc
* GNU make
* LaTeX

For Python, we recommend to install the Anaconda library (https://www.continuum.io/downloads). Under Windows, two main toolchains can be installed: 1) Via Cygwin (https://www.cygwin.com/) which comes with gcc and make or 2) with MinGW's gcc (http://www.mingw.org/) and GNU make for Windows (http://gnuwin32.sourceforge.net/packages/make.htm). While the former is available in 32- and 64-bit versions, the latter only comes in 32-bit, but also runs on 64-bit machines. For using git under Windows, we recommend installing TortoiseGit in addition (https://tortoisegit.org/).

Additional requirements for running an algorithm in a specific language:
* Java: none, but see code-experiments/build/java/README.txt for details on the compilation
* Python: cython (http://docs.cython.org/src/quickstart/install.html), see code-experiments/build/python/README for details
* MATLAB: at least MATLAB 2008, for details, see code-experiments/build/matlab/README

Unfortunately, we cannot guarantee that the software runs on any combination of operating system and software. However, we tested it (in part) on Mac OSX, Ubuntu linux, Fedora linux, and Windows (XP, 7, 10) in various combinations of 32-bit and 64-bit compilers, python versions etc. In particular, we continuously test the code through the open source automation server Jenkins on one ubuntu 12.04 machine, one OSX 10.9 machine, and one 32-bit Windows 7 machine with cygwin.


# Typical Workflow
* download the NumBBO/Coco code by clicking on the *Download ZIP* link on the top right of the main NumBBO github page
* run once

    ```
    python do.py run-YOURFAVORITELANGUAGE
    ```

to build and test the example experiment code for the language of your choice (`YOURFAVORITELANGUAGE` can thereby be any of `python`, `c`, `java`, or `matlab`).
* If the example experiment runs, you can start connecting your favorite algorithm to Coco. The probably easiest is to replace the random search in the example experiment within the `code-experiments/build/YOURFAVORITELANGUAGE` folder by the call to your algorithm. To this end, you can move the `code-experiments/build/YOURFAVORITELANGUAGE` folder wherever you want without the need to build/compile the NumBBO/Coco part a second time. Another entry point for your own experiments can be the more complicated algorithms CMA-ES (for the single-objective suites) and SMS-EMOA (for the multiobjective case) in the `code-experiments/examples` folder. Do not forget, in any case, to update the result folder and the algorithm name and info in the experiment.
* Now you can run your favorite algorithm on the `bbob-biobj` (for a multi-objective algorithm) or on the `bbob` suite (for a single-objective algorithm). Output is automatically generated in the specified result folder.
* Postprocess your data from the results folder by typing

    ```
    python rungeneric.py YOURRESULTFOLDER
    ```

within the `code-postprocessing/bbob_pproc` folder (or running the `rungeneric.py` script from where your results lie). A folder named `ppdata` by default will be generated (the folder name can be changed by the `-o FOLDERNAME` option). Note that you can also compare more than one algorithm by specifying more algorithm result folders, separated by blanks.
* Within the postprocessing's output folder, you will find pdfs of all kinds of plots (e.g. data profiles). For the single-objective `bbob` suite, they can be used to produce a summary pdf via LaTeX. The corresponding templates in ACM format can be found in the `code-postprocessing/latex-templates` folder. LaTeX templates for the multi-objective `bbob-biobj` suite will follow in a later release. A basic html output is also available in the result folder of the postprocessing (file `templateBBOBarticle.html`).
* Once your algorithm runs well, increase the budget in your experiment script and follow the above steps successively until you are happy.
* If you detect bugs or other issues, please let us know by opening an issue in our issue tracker at https://github.com/numbbo/numbbo/issues.


# Links and Further Documentation
* Downloading this repository via the above "Download ZIP" button or by typing 'git clone https://github.com/numbbo/numbbo.git' or by typing https://github.com/numbbo/numbbo/archive/master.zip in your browser's address line
* The BBOB workshop series, which uses the NumBBO/Coco framework extensively, can be tracked at http://numbbo.github.io/workshops/
* Stay informed about the BBOB workshop series and releases of the NumBBO/Coco software by registering at http://coco.gforge.inria.fr/register
* More detailed documentation of the existing benchmark suites can be found here:
  * for **bbob** at http://coco.lri.fr/downloads/download15.03/bbobdocexperiment.pdf and http://coco.lri.fr/downloads/download15.03/bbobdocexperiment.pdf
  * for the **bbob-biobj** functions at http://numbbo.github.io/bbob-biobj-functions-doc
* Online documentation of the NumBBO/Coco API (i.e. for the ANSI C code) is available at http://numbbo.github.io/COCOdoc/C