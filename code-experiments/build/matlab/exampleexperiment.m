% This script runs random search for 2*DIM function evaluations on 
% the biobjective 'bbob-biobj' suite.
% 
% An example experiment on the single-objective 'bbob' suite can be started
% by renaming the suite_name below.

more on; % to get immediate output in Octave


BUDGET = 2; % algorithm runs for BUDGET*dimension funevals
suite_name = 'bbob-biobj'; % works for 'bbob' as well
observer_name = suite_name; 
suite_instance = '';

observer_options = sprintf("result_folder: RS_on_%s \
                            algorithm_name: RS \
                            algorithm_info: \"A simple random search algorithm\" \
                            log_level: info", suite_name);
       

disp(['Running random search on the ', suite_name, ' suite...']);
    
% dimension 40 is optional:
suite = cocoSuite(suite_name, "year: 2016", "dimensions: 2,3,5,10,20,40");
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
