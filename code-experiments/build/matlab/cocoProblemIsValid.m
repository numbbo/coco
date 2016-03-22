% Returns 1 if the given problem is a valid Coco problem, 0 otherwise.
%
% Note:
%   Can be used to decide whether a given suite has been exhaustively solved
%   (see for example how it is used in the exampleexperiment.m).
%
% Parameters:
%   problem  The given problem.
function flag = cocoProblemIsValid(problem)
flag = cocoCall('cocoProblemIsValid', problem);