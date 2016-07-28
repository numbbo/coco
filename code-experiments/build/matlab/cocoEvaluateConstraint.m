% Evaluates the problem constraints at the point x, saves the result in y.
% and increases the number of constraint evaluations.
%
% Note:
%    Both x and y must point to correctly sized allocated memory regions.
%
% Parameters:
%    problem  The given COCO problem.
%    x        The decision vector.
%
% Returns:
%    y        Vector of constraints values resulting from the evaluation
function y = cocoEvaluateConstraint(problem, x)
y = cocoCall('cocoEvaluateConstraint', problem, x);
