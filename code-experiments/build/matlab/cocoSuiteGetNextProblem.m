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
function problem = cocoSuiteGetNextProblem(suite, observer)
problem = cocoCall('cocoSuiteGetNextProblem', suite, observer);