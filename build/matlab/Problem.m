classdef Problem < handle
    
    properties
        problem
        number_of_variables
        number_of_objectives
        lower_bounds
        upper_bounds
        problem_suite
        function_index
    end
    
    methods
        function Pr = Problem(problem_suite, function_index)
            Pr.problem = cocoGetProblem(problem_suite, function_index);
            Pr.problem_suite = problem_suite;
            Pr.function_index = function_index;
            Pr.lower_bounds = cocoGetSmallestValuesOfInterest(Pr.problem);
            Pr.upper_bounds = cocoGetLargestValuesOfInterest(Pr.problem);
            Pr.number_of_variables = cocoGetNumberOfVariables(Pr.problem);
            Pr.number_of_objectives = cocoGetNumberOfObjectives(Pr.problem);
        end
        
        function addObserver(Pr, observer, options)
            Pr.problem = cocoObserveProblem(observer, Pr.problem, options);
        end
        
        function free(Pr)
            cocoFreeProblem(Pr.problem);
        end
        
        function S = id(Pr)
            S = cocoGetProblemId(Pr.problem);
        end
        
        function S = name(Pr)
            S = cocoGetProblemName(Pr.problem); % TODO: to be defined
        end
        
        function eval = evaluations(Pr)
            eval = cocoGetEvaluations(Pr.problem); % TODO: to be defined
        end
        
        % TODO: remove toString
        function S = toString(Pr)
            S = cocoGetProblemId(Pr.problem);
            if isempty(S)
                S = 'finalized/invalid problem';
            end
        end

    end
    
end

