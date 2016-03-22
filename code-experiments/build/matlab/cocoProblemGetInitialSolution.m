% Returns an initial solution, i.e. a feasible variable setting, to the problem.
%
% If a special method for setting an initial solution to the problem does not
% exist, the center of the problem's region of interest is the initial solution.
%
% Parameters:
%   problem  The given COCO problem.
%
%
% Example usage:
%
%   >> suite = cocoCall('cocoSuite', 'bbob-biobj', 'year: 2016', 'dimensions: 2,3,5,10,20,40');
%   >> observer = cocoCall('cocoObserver', 'bbob-biobj', 'result_folder: test');
%   COCO INFO: Results will be output to folder exdata\test
%   >> problem = cocoCall('cocoSuiteGetNextProblem', suite, observer);
%
%   COCO INFO: 09.02.16 16:31:16, d=2, running: f01.>> cocoCall('cocoProblemGetInitialSolution', problem)
%   ans = 
%
%     0  0
%
%   >> cocoCall('cocoObserverFree', observer);
%   >> cocoCall('cocoSuiteFree', suite);
%