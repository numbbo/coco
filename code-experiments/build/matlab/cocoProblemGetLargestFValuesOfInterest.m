% For multi-objective problems, returns a vector of largest values of 
% interest in each objective. Currently, this equals the nadir point. 
% For single-objective problems it raises an error..
%
% Parameters:
%    problem  The given problem.
function ub = cocoProblemGetLargestFValuesOfInterest(problem)
ub = cocoCall('cocoProblemGetLargestFValuesOfInterest', problem);