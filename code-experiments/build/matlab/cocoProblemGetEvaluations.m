% Returns the number of evaluations done on the given problem.
%
% Parameters:
%   problem  The given problem.
function count = cocoProblemGetEvaluations(problem)
count = cocoCall('cocoProblemGetEvaluations', problem);