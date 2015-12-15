classdef Benchmark < handle
  
    properties
        problem_suite
        problem_suite_options
        observer
        observer_options
        len
        dimensions
        objectives
        current_problem_index
    end
    
    methods
        function B = Benchmark(problem_suite, problem_suite_options, observer, observer_options)
            B.problem_suite = problem_suite;
            B.problem_suite_options = problem_suite_options;
            B.observer = observer;
            B.observer_options = observer_options;
            B.len = 0;
            B.dimensions = 0;
            B.objectives = 0;
            B.current_problem_index = -1;
        end
        
        function Pr = getProblemUnobserved(B, problem_index) % handle exceptions
            try
                Pr = Problem(B.problem_suite, problem_index);
            catch e
                disp(['Benchmark.getProblemUnobserved: ', e.message]);
                throw(e);
            end
        end
        
        function Pr = getProblem(B, problem_index) % handle exceptions
            try
                Pr = getProblemUnobserved(B, problem_index);
                addObserver(Pr, B.observer, B.observer_options);
            catch e
                disp(['Benchmark.getProblem: ', e.message]);
                throw(e);
            end
        end
        
        function index = getNextProblemIndex(B, problem_index)
            index = cocoSuiteGetNextProblemIndex(B.problem_suite, problem_index, B.problem_suite_options);
        end
        
        function Pr = getNextProblem(B) % handle exceptions
            try
                B.current_problem_index = getNextProblemIndex(B, B.current_problem_index);
                Pr = getProblem(B, B.current_problem_index);
            catch e
                disp(['Benchmark.getNextProblem: ', e.message]);
                throw(e);
            end
        end
    end
    
end

