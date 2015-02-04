my_benchmark = Benchmark('bbob2009', 'bbob2009_observer', 'random_search');
while true
    problem = nextProblem(my_benchmark);
    disp(['Optimizing ', problem.toString()]);
    my_optimizer(problem, cocoGetSmallestValuesOfInterest(problem), cocoGetLargestValuesOfInterest(problem), 100);
    %free(problem);
end