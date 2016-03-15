% Returns the name of the given problem.
%
% Parameters:
%   problem  The given problem.
function name = cocoProblemGetName(problem)
name = cocoCall('cocoProblemGetName', problem);