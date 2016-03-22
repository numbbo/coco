% Removes an observer from the given problem.
%
% Frees the observer's logger and returns the inner problem.
%
% Parameters:
%   problem   The observed COCO problem.
%   observer  The COCO observer, whose logger was wrapping the problem.
%
% Returns:
%   The unobserved problem as a pointer to the inner problem or the same
%   problem if the problem was not observed. 
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