% Returns an initial solution, i.e. a feasible variable setting, to the problem.
%
% If a special method for setting an initial solution to the problem does not
% exist, the center of the problem's region of interest is the initial solution.
%
% Parameters:
%   problem  The given COCO problem.
function x = cocoProblemGetInitialSolution(problem)
x = cocoCall('cocoProblemGetInitialSolution', problem);