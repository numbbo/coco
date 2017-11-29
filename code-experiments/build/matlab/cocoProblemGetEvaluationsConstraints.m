% Returns the number of constraint evaluations done on the given problem.
%
% Parameters:
%   problem  The given problem.
function count = cocoProblemGetEvaluationsConstraints(problem)
count = cocoCall('cocoProblemGetEvaluationsConstraints', problem);
