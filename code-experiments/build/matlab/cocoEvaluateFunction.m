% Evaluates the problem function in point x and saves the result in y.
%
% Evaluates the problem function, increases the number of evaluations, and
% updates the best observed value and the best observed evaluation number.
%
% Note:
%    Both x and y must point to correctly sized allocated memory regions.
%
% Parameters:
%    problem  The given COCO problem.
%    x        The decision vector.
%    y        The objective vector that is the result of the evaluation (in
%             single-objective problems only the first vector item is being
%             set). 
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
