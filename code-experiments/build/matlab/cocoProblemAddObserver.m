% Adds an observer to the given problem.
%
% Wraps the observer's logger around the problem if the observer is not NULL and
% invokes the initialization of this logger.
%
% Parameters:
%   problem   The given COCO problem.
%   observer  The COCO observer, whose logger will wrap the problem.
%
% Returns:
%   The observed problem in the form of a new COCO problem instance or the same
%   problem if the observer is NULL.
function obsproblem = cocoProblemAddObserver(problem, observer)
obsproblem = cocoCall('cocoProblemAddObserver', problem, observer);