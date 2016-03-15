% Returns the id of the given problem.
%
% Parameters:
%   problem  The given problem.
function id = cocoProblemGetId(problem)
id = cocoCall('cocoProblemGetId', problem);