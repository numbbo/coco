% Runs the SMS-EMOA on the bbob-biobj test suite.
%
% Note that the whole experiment, as defined below, takes about
% 14 minutes on an Intel Core2 Duo T9600 machine with 2.8GHz and
% 4GB of memory.


MAX_EVALS = 200;
suite_name = 'bbob-biobj';
suite_instance = '';
% dimension 40 is optional:
suite_options = 'dimensions: 2,3,5,10,20 instance_idx: 1-3';
observer_name = 'bbob-biobj';
observer_options = ['result_folder: SMSEMOA_on_bbob-biobj \',...
                    'algorithm_name: SMS-EMOA \',...
                    'algorithm_info: "SMS-EMOA without restarts" \',...
                    'log_decision_variables: low_dim \',...
                    'compute_indicators: 1\',...
                    'log_nondominated: all'];             
                
suite = cocoSuite(suite_name, suite_instance, suite_options);
observer = cocoObserver(observer_name, observer_options);

while true
    problem = cocoSuiteGetNextProblem(suite, observer);
    if (~cocoProblemIsValid(problem))
        break;
    end
    disp(['Optimizing ', cocoProblemGetId(problem)]);
    my_smsemoa(problem, cocoProblemGetSmallestValuesOfInterest(problem), cocoProblemGetLargestValuesOfInterest(problem), MAX_EVALS);
    disp(['Done with problem ', cocoProblemGetId(problem), '...']);
end
cocoObserverFree(observer);
cocoSuiteFree(suite);