function Pr = nextProblem(benchmark)
    if ~validProblem(benchmark.problem_suit, benchmark.function_index)
        msgID = 'nextProblem:NoSuchProblem';
        msg = ['Problem suit ', benchmark.problem_suit, ' lacks a function with function id ', num2str(benchmark.function_index)];
        baseException = MException(msgID, msg);
        throw(baseException)
    end
    Pr = Problem(benchmark.problem_suit, benchmark.function_index, benchmark.observer, benchmark.options);
    benchmark.function_index = benchmark.function_index + 1;
    
end

