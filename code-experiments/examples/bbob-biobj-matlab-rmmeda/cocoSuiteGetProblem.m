% Returns the problem of the suite defined by problem_index.
%
% Parameters:
%   suite          The given suite.
%   problem_index  The index of the problem to be returned.
%
% Example usage:
%
%   >> suite_name = 'bbob-biobj'; % works for 'bbob' as well
%   >> suite = cocoCall('cocoSuite', suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
%   >> problem = cocoCall('cocoSuiteGetProblem', suite, 10);
%   >> cocoCall('cocoProblemGetDimension', problem)
%   ans = 2
%   >> cocoCall('cocoEvaluateFunction', problem, [2, 10])
%   ans =
%
%      5.9543e+002  1.6626e+008
%   >> cocoCall('cocoProblemFree', problem);
%   >> cocoCall('cocoSuiteFree', suite);
function problem = cocoSuiteGetProblem(suite, problem_index)
problem = cocoCall('cocoSuiteGetProblem', suite, problem_index);