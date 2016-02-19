% Adds an observer to the given problem.
%
% Wraps the observer's logger around the problem if the observer is not NULL and
% invokes the initialization of this logger.
%
% Parameters:
%   problem   The given COCO problem.
%   observer  The COCO observer, whose logger will wrap the problem.
% 
% Returns:
%   The observed problem in the form of a new COCO problem instance or the same
%   problem if the observer is NULL. 
%
% Example usage:
%
%   >> suite_name = 'bbob-biobj'; % works for 'bbob' as well
%   >> suite = cocoCall('cocoSuite', suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
%   >> problem = cocoCall('cocoSuiteGetProblem', suite, 10);
%   >> cocoCall('cocoProblemGetDimension', problem)
%   ans = 2
%   >> cocoCall('cocoEvaluateFunction', problem, [2, 10])
%   ans =
%
%      5.9543e+002  1.6626e+008
%   >> observer = cocoCall('cocoObserver', suite_name, 'result_folder: test');
%   COCO INFO: Results will be output to folder exdata\test
%   >> observedproblem = cocoCall('cocoProblemAddObserver', problem, observer);
%   >> cocoCall('cocoEvaluateFunction', problem, [7, 7])
%   ans =
%
%      6.1108e+002  7.7216e+007
%   % all files in test/ still empty because obervedproblem has not been called
%   >> cocoCall('cocoEvaluateFunction', observedproblem, [7, 7])
%   ans =
%
%      6.1108e+002  7.7216e+007
%   % now, solution [2, 10] is recorded in the folder test/
%   >> unobservedproblem = cocoCall('cocoProblemRemoveObserver', ...
%           observedproblem, observer); % unobservedproblem and problem now
%                                       % the same
%   >> cocoCall('cocoObserverFree', observer);
%   >> cocoCall('cocoSuiteFree', suite);
