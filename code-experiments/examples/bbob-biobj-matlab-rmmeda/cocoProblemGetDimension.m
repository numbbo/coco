% Returns the number of variables i.e. the dimension of the problem.
%
% Parameters:
%   problem  The given problem.
function dim = cocoProblemGetDimension(problem)
dim = cocoCall('cocoProblemGetDimension', problem);