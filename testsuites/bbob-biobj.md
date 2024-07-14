---
layout: default
permalink: /testsuites/bbob-biobj
parent: Test Suites
nav_order: 3
title: bbob-biobj
has_toc: false
---


# The bbob-biobj Test Suite

|   | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|---|---|
| The [biobjective bbob-biobj test suite](https://hal.inria.fr/hal-01296987) from 2016 is COCO's first multiobjective test suite with 55 noiseless, scalable bi-objective functions that utilize 10 of the original 24 <a href="bbob">bbob</a> test functions. Each function is provided in various dimensions (2, 3, 5, 10, 20, 40, and scalable to any dimension). Hypervolume reference values and thus the postprocessing by COCO are available for the first 15 function instances in the provided dimensions.| <img align="top" position="relative" src="https://numbbo.github.io/ppdata-archive/bbob-biobj/2016/pprldmany_10D_noiselessall.svg" alt="ECDF of runtimes for 16 algorithms on the bbob-biobj suite in dimension 10" width="100%"/>|

- The paper _Using Well-Understood Single-Objective Functions in Multiobjective Black-Box Optimization Test Suites_ ([_Evolutionary Computation_ (2022) 30 (2): 165–193][1]) describes the suite construction in detail.
- More details about the test functions, including visualizations of search and objective space, can be found on the <a href="https://numbbo.github.io/bbob-biobj/">supplementary material webpage</a>.
- Peculiarities of the performance assessment methodology for the bi-objective testbeds are documented <a href="https://arxiv.org/abs/1605.01746">here</a> and published in the paper _Anytime Performance Assessment in Blackbox Optimization Benchmarking_ ([IEEE TEC (2022) 26 (6): 1293-1305][2]).
- A list of all so-far benchmarked algorithms on the bbob-biobj suite together with their links to papers describing the experiment can be found in our <a href="https://numbbo.github.io/data-archive/bbob-biobj/">bbob-biobj data archive</a>.
- Postprocessed data can be found <a href="https://numbbo.github.io/ppdata-archive">here</a>.
- For detailed explanations of how to use the functions in a COCO benchmarking experiment, see the <a href="https://github.com/numbbo/coco">COCO code page</a> on Github. 

[1]: https://doi.org/10.1162/evco_a_00298
[2]: https://doi.org/10.1109/TEVC.2022.3210897

<link rel="stylesheet" href="{{ '/assets/css/custom.css' | relative_url }}"/>
