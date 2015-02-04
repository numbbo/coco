function Pr = nextProblem(benchmark)
    % Handle exceptions : validProblem
    if ~validProblem(benchmark.problem_suit, benchmark.function_index)
        disp(['Problem suit ', benchmark.problem_suit, ' lacks a function with function id ', benchmark.function_index]);
        return
    end
    Pr = Problem(benchmark.problem_suit, benchmark.function_index, benchmark.observer, benchmark.options);
    benchmark.function_index = benchmark.function_index + 1;
    
end

