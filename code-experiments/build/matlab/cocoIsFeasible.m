% Checks whether the point x is feasible by calling cocoEvaluateConstraint with x as input.
%
% Note:
%    x must point to correctly sized allocated memory region.
%
% Parameters:
%    problem  The given COCO problem.
%    x        The decision vector.
%
% Returns:
%    is_feasible   1 if x is feasible and 0 otherwise
%    y             Vector of constraints values resulting from the evaluation
function [is_feasible, y] = cocoIsFeasible(problem, x)
[is_feasible, y] = cocoCall('cocoIsFeasible', problem, x);
