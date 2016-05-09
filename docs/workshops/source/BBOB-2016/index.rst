.. _bbob2016page:

GECCO Workshop on Real-Parameter Black-Box Optimization Benchmarking (BBOB 2016) - focus on multi-objective problems
==========================================================================================================================


Welcome to the web page of the upcoming Workshop on Real-Parameter Black-Box Optimization Benchmarking (BBOB 2016)
with focus on multi-objective problems with two objective functions which is going to take place during GECCO 2016.

    **WORKSHOP ON REAL-PARAMETER BLACK-BOX OPTIMIZATION BENCHMARKING (BBOB 2016) - with a new focus on multi-objective problems**

    | to be held as part of the
    |
    | **2016 Genetic and Evolutionary Computation Conference (GECCO-2016)**
    | July 20-24, Denver, CO, USA
    | http://gecco-2016.sigevo.org/


| Submission Deadline: extended to Sunday, April 17, 2016 (from Saturday, April 3, 2016)
|


=======================================================  ========================================================================  =======================================================================================
`register for news <http://numbbo.github.io/register>`_  `Coco quick start (scroll down a bit) <https://github.com/numbbo/coco>`_  `latest Coco release <https://github.com/numbbo/coco/releases/>`_
=======================================================  ========================================================================  =======================================================================================


Quantifying and comparing the performance of optimization algorithms
is a difficult and tedious task to achieve. Previously, the Coco
platform has provided tools to ease this process for *single-objective
problems* by: (1) an implemented, well-motivated benchmark function
testbed, (2) a simple and sound experimental set-up, (3) the generation
of output data and (4) the post-processing and presentation of the
results in graphs and tables. For the first time, this year, we provide
an extension of the Coco platform towards a *multi-objective testbed*
(with two objective functions) and with nearly the same procedure as in
previous BBOB workshops.

The remaining tasks for participants are therefore: run your favorite
multi-objective black-box optimizer (old or new) by using the wrappers
provided and run the post-processing procedure (provided as well) that
will generate automatically all the material for a workshop paper
(LaTeX templates are provided). A description of the algorithm and the
discussion of the results completes the paper writing.

We encourage particularly submissions related to multi-objective algorithms
for *expensive optimization* (with a limited budget) and also *algorithms
from outside the evolutionary computation community*. Please note that
submissions related to the existing *single-objective BBOB testbeds*
(noiseless and noisy) are still welcome although the focus will be on
the new bi-objective testbed.

During the workshop, algorithms and results will be presented by
the participants. An overall analysis and comparison will be
accomplished by the organizers and the overall process will be
critically reviewed. A plenary discussion on future improvements will,
among others, address the question, of how the testbeds should evolve.


Updates and News
----------------
Get updated about the latest news regarding the workshop and
releases and bugfixes of the supporting NumBBO/Coco plateform, by
registering at http://numbbo.github.io/register.


Supporting material
-------------------
Most likely, you want to read the `Coco quick start <https://github.com/numbbo/coco>`_
(scroll down a bit). This page also provides the code for the benchmark functions, for running the
experiments in C, Java, Matlab, Octave, and Python, and for postprocessing the experiment data
into plots, tables, html pages, and publisher-conform PDFs via provided LaTeX templates.
Please refer to http://numbbo.github.io/coco-doc/experimental-setup/
for more details on the general experimental set-up for black-box optimization benchmarking.

The latest (hopefully) stable release of the Coco software can be downloaded as a whole
`here <https://github.com/numbbo/coco/releases/>`_. Please use at least version v1.0 for
running your benchmarking experiments.

Documentation of the functions used in the `bbob-biobj` suite for BBOB 2016 are provided at
http://numbbo.github.io/coco-doc/bbob-biobj/functions/ .

Note that the current release of the new Coco platform does not contain the original noisy BBOB testbed,
such that you must use the old code at http://coco.gforge.inria.fr/doku.php?id=downloads for the time
being if you want to compare your algorithm on the noisy testbed.


Submissions
-----------
Submissions of benchmarking results of new or existing numerical optimization algorithms in terms
of bi-objective or single-objective optimization are welcome and should be done through the
following form at http://numbbo.github.io/submit.





Important Dates
---------------

* **01/20/2016** first version of the new Coco platform released as `0.5-beta <https://github.com/numbbo/coco/releases/>`_
* **01/30/2016** (planned: 01/29/2016) release `0.7-beta <https://github.com/numbbo/coco/releases/>`_ of the Coco software with the main functionality to run experiments
* (planned: 02/12/2016, replaced by 7 intermediate releases) first complete release `0.9 <https://github.com/numbbo/coco/releases/>`_ of the software
* **03/29/2016** (planned: 03/18/2016) final release `1.0 <https://github.com/numbbo/coco/releases/>`_ for producing the papers
* **04/17/2016** new *paper and data submission deadline* (extended from 04/02/2016)
* **04/20/2016** decision notification
* **05/04/2016** deadline camera-ready papers
* **07/20/2016** or **07/21/2016** workshop


Organizers
----------
* Anne Auger, Inria Saclay - Ile-de-France
* Dimo Brockhoff, Inria Lille - Nord Euruope
* Nikolaus Hansen, Inria Saclay - Ile-de-France
* Dejan Tusar, Inria Lille - Nord Europe
* Tea Tusar, Inria Lille - Nord Europe
* Tobias Wagner, TU Dortmund University
