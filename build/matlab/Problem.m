classdef Problem
    
    properties
      number_of_variables
      number_of_objectives
      number_of_constraints
      smallest_values_of_interest
      largest_values_of_interest
      best_value
      best_parameter
      problem_name
      problem_id
      problem_suit
      function_index
      observer_name
      options
    end
    
    methods
        function Pr = Problem(problem_suit, function_index)
            Pr.problem_suit = problem_suit;
            Pr.function_index = function_index;
            Pr.observer_name = '';
            Pr.options = '';
        end
        
        function cocoEvaluateFunction(Pr, x)
        end
        
        function cocoGetNumberOfVariables(Pr)
        end
        
        function cocoGetNumberOfObjectives(Pr)
        end
        
        function cocoGetSmallestValuesOfInterest(Pr)
        end
        
        function cocoGetLargestValuesOfInterest(Pr)
        end
        
        function cocoGetProblemId(Pr)
        end
    end
    
end

