---
layout: default
permalink: /testsuites
has_children: true
nav_order: 1
title: test suites
has_toc: false
---


# Available Test Suites in COCO

---

COCO provides different test suites with about 20..100 test functions which are available in different dimensions and with an arbitrary number of instances. Below, you find the currently supported list with links to the function description, available data sets and postprocessed data.

- [bbob](bbob): standard single-objective BBOB benchmark suite with 24 noiseless, scalable test functions
  - [function definitions](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf)
  - [data archive](https://numbbo.github.io/data-archive/bbob)
  - [archive of postprocessed data](https://numbbo.github.io/ppdata-archive/bbob)  
- [bbob-noisy](bbob-noisy): single-objective suite with 30 noisy functions with different noise levels and noise types
  - !  for now, experiments must be done with the [old code](oldcode/bboball15.03.tar.gz)
  - [function definitions](https://hal.inria.fr/inria-00369466/document)
  - [data archive](https://numbbo.github.io/data-archive/bbob-noisy)
  - [archive of postprocessed data](https://numbbo.github.io/ppdata-archive/bbob-noisy)  
- [bbob-biobj](bbob-biobj): a bi-objective benchmark suite, combining 10 selected functions from the bbob suite, resulting in 55 noiseless functions
  - [data archive](https://numbbo.github.io/data-archive/bbob-biobj)
  - [function definitions, visualizations, and postprocessed data](https://numbbo.github.io/bbob-biobj/)
- bbob-largescale: a version of the bbob benchmark suite with dimensions 20 to 640, employing permuted block-diagonal matrices to reduce the execution time for function evaluations in higher dimension
- bbob-mixint: a mixed-integer version of the original bbob and bbob-largescale suites in which 80% of the variables have been discretized
- bbob-biobj-mixint: a version of the (so far not supported) bbob-biobj-ext test suite with 92 functions with 80% discretized variables
- bbob-constrained: a first suite of 54 non-linearly constrained test functions with varying number of (active and inactive) constraints





<link rel="stylesheet" href="{{ '/assets/css/custom.css' | relative_url }}"/>
