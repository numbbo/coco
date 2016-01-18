% This script runs random search for 100 function evaluations on both 
% the single-objective, noiseless 'bbob' suite and on the biobjective
% 'bbob-biobj' suite.

BUDGET = 10; % algorithm runs for BUDGET*dimension funevals
suiteAndObserver_names = {'bbob', 'bbob-biobj'};
suite_instance = '';


for i = 1:length(suiteAndObserver_names)
    suite_name = suiteAndObserver_names{i};
    disp(['Running random search on the ', suite_name, ' suite...']);
    
    % dimension 40 is optional:
    suite_options = 'dimensions: 2,3,5,10,20,40 instance_idx: 1-5';
    observer_name = suite_name;
    switch suite_name
        case 'bbob'
            observer_options = ['result_folder: RS_on_bbob \',...
                    'algorithm_name: RS \',...
                    'algorithm_info: "A simple random search algorithm"'];
        case 'bbob-biobj'
            observer_options = ['result_folder: RS_on_bbob-biobj \',...
                    'algorithm_name: RS \',...
                    'algorithm_info: "A simple random search algorithm" \',...
                    'log_decision_variables: low_dim \',...
                    'compute_indicators: 1 \',...
                    'log_nondominated: all'];
    end
    suite = cocoSuite(suite_name, suite_instance, suite_options);
    observer = cocoObserver(observer_name, observer_options);
    while true
        problem = cocoSuiteGetNextProblem(suite, observer);
        if (~cocoProblemIsValid(problem))
            break;
        end
        disp(['Optimizing ', cocoProblemGetId(problem)]);
        dimension = cocoProblemGetDimension(problem);
        my_optimizer(problem, cocoProblemGetSmallestValuesOfInterest(problem), cocoProblemGetLargestValuesOfInterest(problem), BUDGET*dimension);
        disp(['Done with problem ', cocoProblemGetId(problem), '...']);
    end
    cocoObserverFree(observer);
    cocoSuiteFree(suite);
end