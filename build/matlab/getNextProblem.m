function Pr = getNextProblem(benchmark)
    if ~problemIsValid(benchmark.problem_suite, benchmark.function_index)
        msgID = 'getNextProblem:NoSuchProblem';
        msg = ['Problem suite ', benchmark.problem_suite, ' lacks a function with function id ', num2str(benchmark.function_index)];
        baseException = MException(msgID, msg);
        throw(baseException)
    end
    Pr = Problem(benchmark.problem_suite, benchmark.function_index, benchmark.observer, benchmark.options);
    % FIXME: this should use the coco_suite_get_next_problem_index function from coco.h
    benchmark.function_index = benchmark.function_index + 1;
    
end

