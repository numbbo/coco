% Returns the number of variables i.e. the dimension of the problem. 
%
% Parameters:
%   problem  The given problem.
%
%
% Example usage:
%
%   >> suite_name = 'bbob-biobj'; % works for 'bbob' as well
%   >> suite = cocoCall('cocoSuite', suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
%   >> problem = cocoCall('cocoSuiteGetProblem', suite, 666);
%   >> cocoCall('cocoProblemGetDimension', problem)
%   ans = 3
%   >> cocoCall('cocoProblemFree', problem);
%   >> cocoCall('cocoSuiteFree', suite);
%
