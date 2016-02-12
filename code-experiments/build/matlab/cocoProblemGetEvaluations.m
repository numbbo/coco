% Returns the number of evaluations done on the given problem. 
%
% Parameters:
%   problem  The given problem.
%
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
%     5.9543e+002  1.6626e+008
%
%   >> cocoCall('cocoProblemGetEvaluations', problem)
%   ans = 1
%   >> cocoCall('cocoEvaluateFunction', problem, [4, 10])
%   ans =
%
%    6.2303e+002  1.6626e+008
%
%   >> cocoCall('cocoProblemGetEvaluations', problem)
%   ans = 2
%   >> cocoCall('cocoProblemFree', problem);
%   >> cocoCall('cocoSuiteFree', suite);
%
