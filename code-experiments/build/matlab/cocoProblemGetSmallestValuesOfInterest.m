% Returns a vector of size 'dimension' with lower bounds of the region of
% interest in the decision space.
%
% Parameters:
%    problem  The given problem.
function lb = cocoProblemGetSmallestValuesOfInterest(problem)
lb = cocoCall('cocoProblemGetSmallestValuesOfInterest', problem);