% Returns the number of constraints of the given problem.
%
% Parameters:
%   problem  The given problem.
function nObj = cocoProblemGetNumberOfConstraints(problem)
nObj = cocoCall('cocoProblemGetNumberOfConstraints', problem);
