---
layout: default
nav_exclude: true
title: ecdf-bbob
has_toc: false
---


# Comparison of 31 algorithms from BBOB-2009 on bbob suite
-----------------------------------------------------------

<p style="text-align:center">
   <a href="pprldmany_default.svg"><img src="pprldmany_default.svg" alt="ECDF of runtimes for 31 algorithms on the bbob suite" width="90%"/></a>
</p>

Empirical runtime distributions (runtime in number of function evaluations divided by dimension) on all functions with target values in {100, … , 1e-8} in dimension 10. The cross indicates the maximum number of function evaluations. A decline in steepness right after the cross (e.g. for IPOP-SEP-CMA-ES) indicates that the maximum number of function evaluations should have been chosen larger. A steep increase right after the cross (e.g. for simple GA) indicates that a restart should have been invoked earlier. 

The plot stems from the BBOB-2010 paper <a href="https://hal.archives-ouvertes.fr/hal-00545727/file/ws1p34.pdf">&quot;Comparing Results of 31 Algorithms from the Black-Box
Optimization Benchmarking BBOB-2009&quot;</a>.
For a more detailed performance assessment of all so-far [benchmarked algorithms on bbob](https://numbbo.github.io/data-archive/bbob/), see the [ppdata archive](https://numbbo.github.io/ppdata-archive).



<link rel="stylesheet" href="{{ '/assets/css/custom.css' | relative_url }}"/>
