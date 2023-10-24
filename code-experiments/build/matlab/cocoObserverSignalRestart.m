% Signals the restart of the algorithm.
%
% Parameters:
%   observer  The given observer.
%   problem   The given problem.
function cocoObserverSignalRestart(observer, problem)
cocoCall('cocoObserverSignalRestart', observer, problem);