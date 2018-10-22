% Returns the number of integer variables of the given problem (if > 0, all 
% integer variables come before any continuous ones.
%
% Parameters:
%   problem  The given problem.
function nIntVar = cocoProblemGetNumberOfIntegerVariables(problem)
nIntVar = cocoCall('cocoProblemGetNumberOfIntegerVariables', problem);
