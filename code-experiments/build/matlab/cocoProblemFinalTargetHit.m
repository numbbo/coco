% Returns 1 if the final target was hit on given problem, 0 otherwise.
%
% Note:
%   Can be used to prevent unnessary burning of CPU time. 
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
%   >> cocoCall('cocoProblemFinalTargetHit', problem)
%   ans = 0
%   >> cocoCall('cocoProblemFree', problem);
%   >> cocoCall('cocoSuiteFree', suite);
%