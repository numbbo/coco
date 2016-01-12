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

# Description by folder:

o the do.py file in the root folder is a tool to build the entire distribution (like a make file, maybe it
should rather have been named make.py). It has switches for just building some languages etc, e.g.

    python ./do.py build  # builds all
    python ./do.py build-python
    python ./do.py build-c
    python ./do.py build-java
    python ./do.py build-matlab

are valid commands (on a Linux or OSX shell, you can omit the leading 'python' if you wish). Our do.py is a neat and
simplifying replacement for make. It can also run the experiments directly, e.g., by typing

    python ./do.py run # runs all
    python ./do.py run-python
    python ./do.py run-c
    python ./do.py run-java
    python ./do.py run-matlab

you can start the corresponding example experiments scripts in code-experiments/build/LANGUAGE/.

o the code-experiments/build folder is to a large extend the output folder of the "./do.py build" command.
   - the exampleexperiment.??? files in the build folder are the entry points to understand the usage of the code (as
     end-user). They are supposed to actually be executable (in case, after compilation, which should be
     taken care of by do.py and/or make) and run typically random search on (some of) the provided benchmark suites.

o documentation and examples might not be too meaningful for the time being, even though
  code-experiments/documentation/onion.py describes a (heavily) used design pattern (namely: inheritance) in a
  comparatively understandable way (though the implementation in C naturally looks somewhat different). In the
  future, documentation will be contained mainly in the docs/ subfolder with the source code extracted automatically
  into pdfs in this folder and to web pages under the numbbo.github.io/ domain.

o the code-experiments/src folder is where most of the important/interesting things happen. Many files provide
  comparatively decent documentation at the moment which are translated via doxygen into a more readable web page at
  numbbo.github.io/COCOdoc/C/. Generally:
  - coco.h is the public interface, in particular as used in the demo.c file, however check out
       https://code.google.com/p/numbbo/issues/detail?id=98
  - coco_internal.h provides the type definition of coco_problem_t
  - coco_suite.c is code that deals with an entire benchmark suite (i.e. a set of functions, eg. sweeping
    through them etc...)
  - coco_generics.c is somewhat generic code, e.g. defining a function call via coco_evaluate_function etc
  - coco_problem.c is the implementation of the coco_problem_t type/object (allocation etc).
  - observer / logger files implement data logging (as wrappers around a coco problem inheriting thereby 
    all properties of a coco problem)
  - most other files implement more or less what they say, e.g. the actual benchmark functions, 
    transformations, benchmark suites, etc.
  - currently, three benchmark suites and corresponding logging facilities are implemented:
    * bbob: standard single-objective BBOB benchmark suite with 24 noiseless, scalable test functions
    * bbob-biobj: a bi-objective benchmark suite, combining 10 selected functions from the bbob suite, giving 55
      noiseless functions
    * toy: a simple, probably easier-to-understand example for reading and testing

o code-experiments/tools are a few meta-tools, mainly the amalgamate.py to merge all the C code into one file

o code-experiments/test contains unit- and integration-tests, mainly for internal use

o code-postprocessing/bbob_pproc contains the postprocessing code, written in python, with which algorithm data sets can
  be read in and the performance of the algorithms can be displayed in terms of data profiles, ERT vs. dimension plots,
  or simple tables.

o code-postprocessing/latex-templates contains LaTeX templates for displaying algorithm performances in
  publisher-conform PDFs for the GECCO and CEC conferences (for the single-objective bbob suite only, templates for
  the bi-objective bbob-biobj suite will be provided in a later release).

o docs should contain an updated version of the documentation, see above.

o howtos contains a few text files with internal howtos.

