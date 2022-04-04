---
layout: default
nav_exclude: true
title: ecdf-bbob-biobj
has_toc: false
---


# Comparison of 16 algorithms from BBOB-2016 on the bbob-biobj suite
---------------------------------------------------------------------

<p style="text-align:center">
   <a href="https://numbbo.github.io/ppdata-archive/bbob-biobj/2016/pprldmany_10D_noiselessall.svg"><img src="https://numbbo.github.io/ppdata-archive/bbob-biobj/2016/pprldmany_10D_noiselessall.svg" alt="ECDF of runtimes for 16 algorithms on the bbob-biobj suite" width="50%"/></a>
</p>

Empirical runtime distributions (runtime in number of function evaluations divided by dimension) on all 55 bbob-biobj functions with target values in {1, ..., 1e-5, 0, -1e-5, -1e-4.8, ..., -1e-4} in dimension 10. The cross indicates the maximum number of function evaluations. A decline in steepness right after the cross (e.g. for HMO-CMA-ES) indicates that the maximum number of function evaluations should have been chosen larger. A steep increase right after the cross (e.g. for SMS-EMOA-DE) indicates that a restart should have been invoked earlier. 

For a more detailed performance assessment of all so-far [benchmarked algorithms on bbob-biobj](https://numbbo.github.io/data-archive/bbob-biobj/), see the [ppdata archive](https://numbbo.github.io/ppdata-archive).


<link rel="stylesheet" href="{{ '/assets/css/custom.css' | relative_url }}"/>
