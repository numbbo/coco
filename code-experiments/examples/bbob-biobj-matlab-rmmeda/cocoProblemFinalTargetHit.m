% Returns 1 if the final target was hit on given problem, 0 otherwise.
%
% Note:
%   Can be used to prevent unnessary burning of CPU time.
%
% Parameters:
%   problem  The given problem.
function flag = cocoProblemFinalTargetHit(problem)
flag = cocoCall('cocoProblemFinalTargetHit', problem);