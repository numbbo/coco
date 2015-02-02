my_benchmark = benchmark('bbob2009', 'bbob2009_observer', 'random_search');
while (TRUE)
    problem = next_problem(my_benchmark);
    my_optimizer(problem, problem.lower_bounds, problem.upper_bounds, 1000);
    free(problem);
end