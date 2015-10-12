classdef Problem < handle
    
    properties
        problem
        dimension
        number_of_objectives
        lower_bounds
        upper_bounds
        problem_suite
        function_index
    end
    
    methods
        function Pr = Problem(problem_suite, function_index)
            Pr.problem = cocoGetProblem(problem_suite, function_index);
            if ~problemIsValid(Pr.problem)
                msgID = 'Problem:NoSuchProblem';
                msg = ['Problem.Problem: Problem suite ', problem_suite, ' lacks a function with function id ', num2str(function_index)];
                baseException = MException(msgID, msg);
                throw(baseException)
            end
            Pr.problem_suite = problem_suite;
            Pr.function_index = function_index;
            Pr.lower_bounds = cocoProblemGetSmallestValuesOfInterest(Pr.problem);
            Pr.upper_bounds = cocoProblemGetLargestValuesOfInterest(Pr.problem);
            Pr.dimension = cocoProblemGetDimension(Pr.problem);
            Pr.number_of_objectives = cocoProblemGetNumberOfObjectives(Pr.problem);
        end
        
        function addObserver(Pr, observer, options)
            Pr.problem = cocoProblemAddObserver(observer, Pr.problem, options);
        end
        
        function free(Pr)
            cocoProblemFree(Pr.problem);
        end
        
        function S = id(Pr)
            S = cocoProblemGetId(Pr.problem);
        end
        
        function S = name(Pr)
            S = cocoProblemGetName(Pr.problem); % TODO: to be defined
        end
        
        function eval = evaluations(Pr)
            eval = cocoGetEvaluations(Pr.problem); % TODO: to be defined
        end
        
        % TODO: remove toString
        function S = toString(Pr)
            S = cocoProblemGetId(Pr.problem);
            if isempty(S)
                S = 'finalized/invalid problem';
            end
        end

    end
    
end

