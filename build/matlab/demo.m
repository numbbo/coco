MAX_EVALUATIONS = 1000;
my_benchmark = Benchmark('bbob2009', 'bbob2009_observer', 'random_search');
while true
    try
        problem = nextProblem(my_benchmark);
        disp(['Optimizing ', problem.toString()]);
        my_optimizer(problem, cocoGetSmallestValuesOfInterest(problem), cocoGetLargestValuesOfInterest(problem), MAX_EVALUATIONS);
    catch e
        disp(e.message);
        return
    end
end