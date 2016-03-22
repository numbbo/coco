% Returns the number of objectives of the given problem.
%
% Parameters:
%   problem  The given problem.
function nObj = cocoProblemGetNumberOfObjectives(problem)
nObj = cocoCall('cocoProblemGetNumberOfObjectives', problem);