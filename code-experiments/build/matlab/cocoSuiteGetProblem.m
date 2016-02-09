# Returns the problem of the suite defined by problem_index.
#
# Example usage:
#
#   >> suite_name = 'bbob-biobj'; % works for 'bbob' as well
#   >> suite = cocoSuite(suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
#   >> problem = cocoSuiteGetProblem(suite, 10);
#   >> cocoProblemGetDimension(problem)
#   ans = 2
#   >> cocoEvaluateFunction(problem, [2, 10])
#   ans =
#
#      5.9543e+002  1.6626e+008
#   >> cocoProblemFree(problem)
#   >> cocoSuiteFree(suite)
#