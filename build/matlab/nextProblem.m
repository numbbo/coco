function Pr = nextProblem(benchmark)
    Pr = Problem(benchmark.problem_suit, benchmark.function_index, benchmark.observer, benchmark.options);
    benchmark.function_index = benchmark.function_index + 1;
end

