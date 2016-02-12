% Returns 1 if the given problem is a valid Coco problem, 0 otherwise.
%
% Note:
%   Can be used to decide whether a given suite has been exhaustively solved
%   (see for example how it is used in the exampleexperiment.m).
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
%   >> cocoCall('cocoProblemIsValid', problem)
%   ans = 1
%   >> cocoCall('cocoProblemFree', problem);
%   >> cocoCall('cocoSuiteFree', suite);
%