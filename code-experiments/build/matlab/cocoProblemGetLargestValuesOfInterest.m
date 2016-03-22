% Returns a vector of size 'dimension' with upper bounds of the region of
% interest in the decision space.
%
% Parameters:
%    problem  The given problem.
function ub = cocoProblemGetLargestValuesOfInterest(problem)
ub = cocoCall('cocoProblemGetLargestValuesOfInterest', problem);