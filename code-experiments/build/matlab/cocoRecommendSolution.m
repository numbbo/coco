% Recommends solution x (evaluates it and logs it, but does not return the
% function values).
%
% Note:
%    x must point to correctly sized allocated memory region.
%
% Parameters:
%    problem  The given COCO problem.
%    x        The decision vector.
%
function cocoRecommendSolution(problem, x)
cocoCall('cocoRecommendSolution', problem, x);
