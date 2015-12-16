MAX_EVALS = 100;
number_of_batches = 1;
current_batch = 1;
suite_name = 'suite_biobj_300';
observer_name = 'observer_biobj';
observer_options = 'result_folder: RS_on_suite_biobj_300 include_decision_variables: 0 log_nondominated: final';
observer = cocoObserver(observer_name, observer_options);
problem_index = -1;
while true
    problem_index = cocoSuiteGetNextProblemIndex(suite_name, problem_index, '');
    if (problem_index < 0)
        break;
    end
    if (mod(problem_index + current_batch - 1, number_of_batches)~= 0)
        continue;
    end
    problem = cocoSuiteGetProblem(suite_name, problem_index);
    problem = cocoProblemAddObserver(problem, observer);
    disp(['Optimizing ', cocoProblemGetId(problem)]);
    my_optimizer(problem, cocoProblemGetSmallestValuesOfInterest(problem), cocoProblemGetLargestValuesOfInterest(problem), MAX_EVALS);
    disp(['Done with problem ', cocoProblemGetId(problem), '...']);
    cocoProblemFree(problem);
end
cocoObserverFree(observer);