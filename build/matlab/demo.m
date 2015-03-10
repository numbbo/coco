MAX_EVALUATIONS = 100;
my_benchmark = Benchmark('bbob2009', '', 'bbob2009_observer', 'random_search');
problem_index = -1
while true
    try
        problem_index = nextProblemIndex(my_benchmark, problem_index);
        problem = getProblem(my_benchmark, problem_index);
        disp(['Optimizing ', problem.toString()]);
        my_optimizer(problem, problem.lower_bounds, problem.upper_bounds, MAX_EVALUATIONS);
    catch e
        disp(e.message);
        return
    end
end