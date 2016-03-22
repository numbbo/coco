% Returns a vector of size 'dimension' with upper bounds of the region of
% interest in the decision space. 
%
% Parameters:
%    problem  The given problem.
%
%
% Example usage:
%
%   >> suite = cocoCall('cocoSuite', 'bbob-biobj', 'year: 2016', 'dimensions: 2,3,5,10,20,40');
%   >> observer = cocoCall('cocoObserver', 'bbob-biobj', 'result_folder: test');
%   COCO INFO: Results will be output to folder exdata\test
%   >> problem = cocoCall('cocoSuiteGetNextProblem', suite, observer);
%
%   COCO INFO: 09.02.16 16:31:16, d=2, running: f01.>> cocoCall('cocoProblemGetLargestValuesOfInterest', problem)
%   ans = 
%
%     5  5
%
%   >> cocoCall('cocoObserverFree', observer);
%   >> cocoCall('cocoSuiteFree', suite);
%