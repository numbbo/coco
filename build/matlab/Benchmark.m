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
            B.problem_suite_options = problem_suite_options
            B.observer = observer;
            B.observer_options = observer_options;
            B.len = 0;
            B.dimensions = 0;
            B.objectives = 0;
            B.current_problem_index = -1;
        end
        
        function Pr = getProblemUnobserved(B, problem_index) % handle exceptions
            Pr = Problem(B.problem_suite, problem_index);
        end
        
        function Pr = getProblem(B, problem_index) % handle exceptions
            Pr = getProblemUnobserved(B, problem_index);
            addObserver(Pr, B.observer, B.observer_options);
        end
        
        function index = nextProblemIndex(B, problem_index)
            index = cocoNextProblemIndex(B.problem_suite, problem_index, B.problem_suite_options);
        end
        
        function Pr = nextProblem(B) % handle exceptions
            B.current_problem_index = nextProblemIndex(B, B.current_problem_index);
            Pr = getProblem(B, B.current_problem_index);
        end
    end
    
end

