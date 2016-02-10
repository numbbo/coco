% Runs the SMS-EMOA on the bbob-biobj test suite.

more off; % to get immediate output in Octave

BUDGET_MULTIPLIER = 2; % algorithm runs for BUDGET_MULTIPLIER*dimension funevals
suite_name = 'bbob-biobj';
observer_name = suite_name;
observer_options = ['result_folder: SMSEMOA_on_bbob-biobj \',...
                    'algorithm_name: SMS-EMOA \',...
                    'algorithm_info: "SMS-EMOA without restarts" \', ...
                    'log_level: COCO_INFO'];      
                
% dimension 40 is optional:
suite = cocoCall('cocoSuite', suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
observer = cocoCall('cocoObserver', observer_name, observer_options);

while true
    problem = cocoCall('cocoSuiteGetNextProblem', suite, observer);
    if (~cocoCall('cocoProblemIsValid', problem))
        break;
    end
    dimension = cocoCall('cocoProblemGetDimension', problem);
    while BUDGET_MULTIPLIER*dimension > cocoCall('cocoProblemGetEvaluations', problem)
        my_smsemoa(problem, ...
            cocoCall('cocoProblemGetSmallestValuesOfInterest', problem), ...
            cocoCall('cocoProblemGetLargestValuesOfInterest', problem), ...
            BUDGET_MULTIPLIER*dimension-cocoCall('cocoProblemGetEvaluations', problem));
    end;
end

cocoCall('cocoObserverFree', observer);
cocoCall('cocoSuiteFree', suite);