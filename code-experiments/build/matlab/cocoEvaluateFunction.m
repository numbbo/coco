% Evaluates the problem function in point x and saves the result in y.
%
% Evaluates the problem function, increases the number of evaluations, and
% updates the best observed value and the best observed evaluation number.
%
% Note:
%    Both x and y must point to correctly sized allocated memory regions.
%
% Parameters:
%    problem  The given COCO problem.
%    x        The decision vector.
%
% Returns:
%    y        The objective vector that is the result of the evaluation (in
%             single-objective problems only the first vector item is being
%             set).
function y = cocoEvaluateFunction(problem, x)
y = cocoCall('cocoEvaluateFunction', problem, x);
