% This script runs random search for 2*DIM function evaluations on
% the biobjective 'bbob-biobj' suite.
%
% An example experiment on the single-objective 'bbob' suite can be started
% by renaming the suite_name below.

more off; % to get immediate output in Octave


BUDGET_MULTIPLIER = 2; % algorithm runs for BUDGET*dimension funevals
suite_name = 'bbob-biobj'; % works for 'bbob' as well
observer_name = suite_name;

observer_options = strcat('result_folder: RS_on_', ...
    suite_name, ...
    [' algorithm_name: RS '...
    ' algorithm_info: A_simple_random_search ']);
    
% dimension 40 is optional:
suite = cocoCall('cocoSuite', suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
observer = cocoCall('cocoObserver', observer_name, observer_options);

disp(['Running random search on the ', suite_name, ' suite...']);
while true
    problem = cocoCall('cocoSuiteGetNextProblem', suite, observer);
    if (~cocoCall('cocoProblemIsValid', problem))
        break;
    end
    dimension = cocoCall('cocoProblemGetDimension', problem);
    while BUDGET_MULTIPLIER*dimension > cocoCall('cocoProblemGetEvaluations', problem)
        my_optimizer(problem,...
            cocoCall('cocoProblemGetSmallestValuesOfInterest', problem),...
            cocoCall('cocoProblemGetLargestValuesOfInterest', problem),...
            BUDGET_MULTIPLIER*dimension-cocoCall('cocoProblemGetEvaluations', problem));
    end;
end
disp(['Done with ', suite_name, ' suite.']);
cocoCall('cocoObserverFree', observer);
cocoCall('cocoSuiteFree', suite);
