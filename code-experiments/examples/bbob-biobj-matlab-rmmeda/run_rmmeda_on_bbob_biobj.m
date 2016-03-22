% Runs the RM-MEDA on the bbob-biobj test suite.

more off; % to get immediate output in Octave

BUDGET_MULTIPLIER = 10000; % algorithm runs for BUDGET*dimension funevals
suite_name = 'bbob-biobj'; % works for 'bbob' as well
observer_name = suite_name;

observer_options = strcat('result_folder: rmmeda_on_', suite_name, ...
    [' algorithm_name: RM-MEDA '...
    ' algorithm_info: "RM-MEDA_without_restarts ']);
    
% dimension 40 is optional:
suite = cocoSuite(suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
observer = cocoObserver(observer_name, observer_options);

disp(['"RM-MEDA on the ', suite_name, ' suite...']);

while true
    problem = cocoSuiteGetNextProblem(suite, observer);
    if (~cocoProblemIsValid(problem))
        break;
    end
    disp(['Optimizing ', cocoProblemGetId(problem)]);
    dimension = cocoProblemGetDimension(problem);
    while BUDGET_MULTIPLIER*dimension > cocoProblemGetEvaluations(problem)
    my_rmmeda(problem, cocoProblemGetSmallestValuesOfInterest(problem),...
      cocoProblemGetLargestValuesOfInterest(problem),...
      BUDGET_MULTIPLIER*dimension-cocoProblemGetEvaluations(problem));
    end;
    disp(['Done with problem ', cocoProblemGetId(problem), '...']);
end

cocoObserverFree(observer);
cocoSuiteFree(suite);