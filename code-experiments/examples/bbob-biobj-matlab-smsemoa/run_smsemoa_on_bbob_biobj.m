% Runs the SMS-EMOA on the bbob-biobj test suite.

more off; % to get immediate output in Octave


BUDGET_MULTIPLIER = 10000; % algorithm runs for BUDGET*dimension funevals
suite_name = 'bbob-biobj'; % works for 'bbob' as well
observer_name = suite_name;

observer_options = strcat('result_folder: SMSEMOA_on_', suite_name, ...
    [' algorithm_name: SMSEMOA '...
    ' algorithm_info: "SMSEMOA_without_restarts ']);

% dimension 40 is optional:
suite = cocoSuite(suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
observer = cocoObserver(observer_name, observer_options);

disp(['"SMSEMOA on the ', suite_name, ' suite...']);

while true
    problem = cocoSuiteGetNextProblem(suite, observer);
    if (~cocoProblemIsValid(problem))
        break;
    end
    disp(['Optimizing ', cocoProblemGetId(problem)]);
    dimension = cocoProblemGetDimension(problem);
    my_smsemoa(problem, cocoProblemGetSmallestValuesOfInterest(problem),...
        cocoProblemGetLargestValuesOfInterest(problem),...
        BUDGET_MULTIPLIER*dimension);
    disp(['Done with problem ', cocoProblemGetId(problem), '...']);
end
disp(['Done with ', suite_name, ' suite.']);
cocoObserverFree(observer);
cocoSuiteFree(suite);