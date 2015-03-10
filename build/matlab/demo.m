MAX_EVALS = 100;
number_of_batches = 1;
current_batch = 1;
my_benchmark = Benchmark('bbob2009', '', 'bbob2009_observer', 'random_search');
problem_index = -1;
while true
    try
        problem_index = nextProblemIndex(my_benchmark, problem_index);
        if (problem_index < 0)
            break;
        end
        if (mod(problem_index + current_batch - 1, number_of_batches)~= 0)
            continue;
        end
        problem = getProblem(my_benchmark, problem_index);
        %disp(['Optimizing ', problem.toString()]);
        my_optimizer(problem, problem.lower_bounds, problem.upper_bounds, MAX_EVALS);
        disp(['Done with problem ', problem.toString(), '...']);
        free(problem);
    catch e
        disp(e.message);
        return
    end
end