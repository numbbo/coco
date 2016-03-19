% Removes an observer from the given problem.
%
% Frees the observer's logger and returns the inner problem.
%
% Parameters:
%   problem   The observed COCO problem.
%   observer  The COCO observer, whose logger was wrapping the problem.
%
% Returns:
%   The unobserved problem as a pointer to the inner problem or the same
%   problem if the problem was not observed.
function unobsproblem = cocoProblemRemoveObserver(problem, observer)
unobsproblem = cocoCall('cocoProblemRemoveObserver', problem, observer);