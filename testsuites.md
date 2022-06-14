---
layout: default
permalink: /testsuites
nav_order: 2
has_children: true
title: Test Suites
has_toc: false
---


# Available Test Suites in COCO

---

COCO provides different test suites with about 20..100 test functions which are available in different dimensions and with an arbitrary number of instances. Below, you find the currently supported list with links to the function description, available data sets and postprocessed data.


| [bbob](bbob) <br /> standard single-objective BBOB benchmark suite with 24 noiseless, scalable test functions             | [function definitions (pdf)](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf)<br />[data archive](https://numbbo.github.io/data-archive/bbob/)<br />[archive of postprocessed data](https://numbbo.github.io/ppdata-archive/)        |
| [bbob-noisy](testsuites/bbob-noisy)<br />single-objective suite with 30 noisy functions with different noise levels and noise types  | !  for now, experiments must be done with the [old code](oldcode/bboball15.03.tar.gz) <br/> [function definitions](https://hal.inria.fr/inria-00369466/document) <br /> [data archive](https://numbbo.github.io/data-archive/bbob-noisy/) <br /> [archive of postprocessed data](https://numbbo.github.io/ppdata-archive/) |
| [bbob-biobj](testsuites/bbob-biobj) <br /> a bi-objective benchmark suite, combining 10 selected functions from the bbob suite, resulting in 55 noiseless functions | [data archive](https://numbbo.github.io/data-archive/bbob-biobj) <br /> [function definitions, visualizations, and postprocessed data](https://numbbo.github.io/bbob-biobj/) |
| bbob-largescale <br /> a version of the bbob benchmark suite with dimensions 20 to 640<br />employing permuted block-diagonal matrices to reduce the execution time for function evaluations in higher dimension | [function definitions (pdf)](https://arxiv.org/pdf/1903.06396.pdf) <br /> [data archive](https://numbbo.github.io/data-archive/bbob-largescale/) <br /> [archive of postprocessed data](https://numbbo.github.io/ppdata-archive/) |
| bbob-mixint <br /> a mixed-integer version of the original bbob and bbob-largescale suites in which 80% of the variables have been discretized | [function definitions (pdf)](https://arxiv.org/pdf/1903.06396.pdf) <br /> [function definitions with level set plots](https://numbbo.github.io/gforge/preliminary-bbob-mixint-documentation/bbob-mixint-doc.pdf) |
| bbob-biobj-mixint <br /> a version of the (so far not supported) bbob-biobj-ext test suite with 92 functions with 80% discretized variables | |
| bbob-constrained <br /> a suite of 54 non-linearly constrained test functions with varying number of (active and inactive) constraints | |





<link rel="stylesheet" href="{{ '/assets/css/custom.css' | relative_url }}"/>
