MAX_EVALS = 100;
number_of_batches = 1;
current_batch = 1;
observer_options = 'result_folder: RS_on_suite_biobj_300 include_decision_variables: 0 log_nondominated: final'
my_benchmark = Benchmark('suite_biobj_300', '', 'observer_biobj', observer_options);
problem_index = -1;
while true
    try
        problem_index = getNextProblemIndex(my_benchmark, problem_index);
        if (problem_index < 0)
            break;
        end
        if (mod(problem_index + current_batch - 1, number_of_batches)~= 0)
            continue;
        end
        problem = getProblem(my_benchmark, problem_index);
        disp(['Optimizing ', problem.toString()]);
        my_optimizer(problem, problem.lower_bounds, problem.upper_bounds, MAX_EVALS);
        disp(['Done with problem ', problem.toString(), '...']);
        freeProblem(problem);
    catch e
        disp(e.message);
        return
    end
end
freeObserver(problem);