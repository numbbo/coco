#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
These comparisons are based on computing the ratio between an ERT value and a
reference (best) ERT value (or the inverse)

ERT loss ratio of an algorithm A for comparison to BBOB-2009. This works
only as comparison to a set of algorithms that reach at least the same
target values. Let f=f_A(EVALS) be the smallest target value such that the
expected running time of algorithm A was smaller than or equal to EVALS.
Let ERT_A=EVALS, if ERT_best(next difficult f) < EVALS and
ERT_A=ERT_A(f_A(EVALS)) otherwise (we have ERT_A(f_A(EVALS)) <= EVALS).
The ERT loss ratio for algorithm A is defined as:
    Loss_A = stat_fcts(exp(CrE_A) * ERT_A / ERT_best(f))

    + where f is a function of EVALS and stat_fcts is the desired statistics
      over the values from all functions (or a subgroup of functions), for
      example the geometric mean, min, max or any quantile. More specific: we
      plot versus 'the budget EVALS' the geometric mean (line) and Box-Whisker
      error bars at EVALS=2*D, 10*D, 100*D,...: a box between 25% and 75% with
      the median as additional symbol, a line with "T" as end-marker between
      10% and 90% (the box covers the line) and a single point for min, max.
      For a function subgroup the Box-Whisker is replaced with the four or five
      actual points with the function number written.
      Caption: ERT loss ratio: expected running time, ERT (measured in number
      of function evaluations), divided by the best ERT seen in BBOB-2009 for
      the respectively same function and target function value, plotted versus
      number of function evaluations for the functions $f_1$--$f_{24}$ in
      dimension $D=XXX$, corrected by the parameter-crafting-effort
      $\exp(CrE)==YYY$. Line: geometric mean over all functions. Box-Whisker
      error bars: 25-75\%-percentile range with median (box),
      10-90\%-percentile range (line), and minimum and maximum ERT loss ratio
      (points). Alternative Box-Whisker sentence: Points: ERT loss ratio for
      each function
    + The problem: how to find out CrE_A? Possible solution: ask for input in
      the script and put the given number into the caption and put exp(CrE_A)
      as small symbol on the y-axis of the figure for cross-checking.
    + This should make a collection of graphs for all functions and all
      subgroups which gives an additional page in the 'single algorithm'
      template. Respective tables could be side-by-side the graphs.
    + Example for how to read the graph: a loss ratio of 4 for ERT=20D means,
      that the function value reached with ERT=20D could be reached with the
      respective best algorithm in ERT_best=5D function evaluations on average.
      Therefore, given a budget of 20*D function evaluations, the best
      algorithm could have further improved the function value using the
      remaining 15*D ($75\%=1-1/4$) function evaluations.
"""
