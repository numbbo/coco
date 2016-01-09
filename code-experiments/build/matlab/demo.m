MAX_EVALS = 100;
suite_name = 'bbob-biobj';
suite_instance = '';
% dimension 40 is optional:
suite_options = 'dimensions: 2,3,5,10,20 instance_idx: 1-5';
observer_name = 'bbob-biobj';
observer_options = ['result_folder: RS_on_bbob-biobj \',...
                    'algorithm_name: RS \',...
                    'algorithm_info: "A simple random search algorithm" \',...
                    'log_decision_variables: low_dim \',...
                    'compute_indicators: log_nondominated: all'];
suite = cocoSuite(suite_name, suite_instance, suite_options);
observer = cocoObserver(observer_name, observer_options);
while true
    problem = cocoSuiteGetNextProblem(suite, observer);
    if (~cocoProblemIsValid(problem))
        break;
    end
    disp(['Optimizing ', cocoProblemGetId(problem)]);
    my_optimizer(problem, cocoProblemGetSmallestValuesOfInterest(problem), cocoProblemGetLargestValuesOfInterest(problem), MAX_EVALS);
    disp(['Done with problem ', cocoProblemGetId(problem), '...']);
end
cocoObserverFree(observer);
cocoSuiteFree(suite);