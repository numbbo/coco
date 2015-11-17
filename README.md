# numbbo: Numerical Black-Box Optimization Benchmarking Framework

The code reimplements the original Comparing Continous Optimizer platform (http://coco.gforge.inria.fr/),
now rewritten fully in ANSI C with the other four languages Java, MATLAB, python, and R calling the C code.

Generally, the code features an object oriented implementation, where the coco_problem_t is the most central
data structure / object. coco.h, demo.c and coco_internal.h are probably the best pointers to start with
(but see also below). coco_problem_t defines a benchmark function instance (in a given dimension), and is 
called via coco_evaluate_function.

# Building the Code
Building of the code is done by merging/amalgamation of all C-code into a single C file, coco.c (by calling
do.py, see below). Like this it becomes very simple to include/use the code in different projects.

# Description by folder:

o the do.py file in the root folder is a tool to build the entire distribution (like a make file, maybe it
should rather have been named make.py). It has switches for just building some languages etc, e.g.

    ./do.py build  # builds all
    ./do.py build-python
    ./do.py build-c
    ./do.py build-java

are valid commands (on a Linux or OSX shell). do.py is a neat and simplifying replacement for make.

o the code-experiments/build folder is to a large extend the output folder of the "./do.py build" command.
   - the demo.??? files in the build folder are the entry points to understand the usage of the code (as
     end-user). They are supposed to actually be executable (in case, after compilation, which should be
     taken care of by do.py and/or make). 

o documentation and examples might not be too meaningful for the time being, even though
  code-experiments/documentation/onion.py describes a (heavily) used design pattern (namely: inheritance) in a
  comparatively understandable way (though the implementation in C naturally looks somewhat different).  

o the code-experiments/src folder is where most of the important/interesting things happen. Many files provide
  comparatively decent documentation at the moment (the idea was to use these docs to generate a doc
  document). Generally:
  - coco.h is the public interface, in particular as used in the demo.c file, however check out
       https://code.google.com/p/numbbo/issues/detail?id=98
  - coco_internal.h provides the type definition of coco_problem_t
  - coco_benchmark.c is code that deals with an entire benchmark (i.e. a set of functions, eg. sweeping 
    through them etc...)
  - coco_generics.c is somewhat generic code, e.g. defining a function call via coco_evaluate_function etc
  - coco_problem.c is the implementation of the coco_problem_t type/object (allocation etc).
  - observer / logger files implement data logging (as wrappers around a coco problem inheriting thereby 
    all properties of a coco problem)
  - most other files implement more or less what they say, e.g. the actual benchmark functions, 
    transformations, benchmark suites, etc

o code-experiments/tools are a few meta-tools, mainly the amalgamate.py to merge all the C code into one file

o code-postprocessing contains the postprocessing code, written in python, with which algorithm data sets can
  be read in and the performance of the algorithms can be displayed in terms of data profiles, ERT vs. dimension plots,
  or simple tables.

o docs should contain an updated version of the documentation

o latextemplates should contain the latex templates for displaying algorithm performances in publisher conform PDFs.

o data contains a few algorithm data sets used for testing.

