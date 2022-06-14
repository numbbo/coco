---
layout: default
permalink: /testsuites/bbob-biobj
parent: test suites
nav_order: 3
title: bbob-biobj
has_toc: false
---


# The bbob-biobj Test Suite

---

<table>
	<tr>
		<td>
			The biobjective bbob-biobj test suite from 2016 is COCO's first multiobjective test suite with 55 noiseless, scalable bi-objective functions that utilize 10 of the original 24 [bbob](../testsuites/bbob) test functions. Each function is provided in various dimension (2, 3, 5, 10, 20, 40). For now, hypervolume reference values and thus the postprocessing of COCO are available for 15 function instances.

            More details about the test functions, including visualizations of search and objective space, can be found on <a href="https://numbbo.github.io/bbob-biobj/">this supplementary material webpage</a>. A list of all so-far benchmarked algorithms on the bbob-biobj suite together with their links to papers describing the experiment can be found in our <a href="https://numbbo.github.io/data-archive/bbob-biobj/">bbob-biobj data archive</a>. Postprocessed data can be found <a href="https://numbbo.github.io/ppdata-archive">here</a>. For detailed explanations of how to use the functions in a COCO benchmarking experiment, please go to the <a href="https://github.com/numbbo/coco">COCO code page</a> on Github.
		</td>
		<td style="width=40%">
			<a href="ecdf-bbob-biobj.html"><img src="https://numbbo.github.io/ppdata-archive/bbob-biobj/2016/pprldmany_10D_noiselessall.svg" alt="ECDF of runtimes for 16 algorithms on the bbob-biobj suite in dimension 10" width="100%"/></a>
		</td>
	</tr>
</table>



<link rel="stylesheet" href="{{ '/assets/css/custom.css' | relative_url }}"/>
