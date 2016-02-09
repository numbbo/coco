% Returns the next (observed) problem of the suite or NULL if there is no next
% problem left. 
%
% Iterates through the suite first by instances, then by functions and finally
% by dimensions. The instances/functions/dimensions that have been filtered out
% using the suite_options of the coco_suite function are skipped. Outputs some
% information regarding the current place in the iteration. The returned problem
% is wrapped with the observer. If the observer is NULL, the returned problem is
% unobserved.
%
% Parameters:
%    suite     The given suite.
%    observer  The observer used to wrap the problem. If NULL, the problem is
%              returned unobserved.
%
% Example usage:
%
%   >> suite = cocoCall('cocoSuite', 'bbob-biobj', 'year: 2016', 'dimensions: 2,3,5,10,20,40');
%   >> observer = cocoCall('cocoObserver', 'bbob-biobj', 'result_folder: test');
%   COCO INFO: Results will be output to folder exdata\test
%   >> problem = cocoCall('cocoSuiteGetNextProblem', suite, observer);
%   COCO INFO: 09.02.16 16:31:16, d=2, running: f01.>> cocoCall('cocoObserverFree', suite);
%   >> cocoCall('cocoSuiteFree', suite);
%