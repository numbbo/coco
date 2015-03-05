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
        function Pr = Problem(varargin)
            if nargin == 2
                Pr.problem_suit = varargin{1};
                Pr.function_index = varargin{2};
                Pr.observer_name = '';
                Pr.options = '';
            else
                if nargin == 4
                    Pr.problem_suit = varargin{1};
                    Pr.function_index = varargin{2};
                    Pr.observer_name = varargin{3};
                    Pr.options = varargin{4};
                end
            end
        end
        
        function S = toString(Pr)
            S = cocoGetProblemId(Pr);
            if isempty(S)
                S = 'finalized/invalid problem';
            end
        end

    end
    
end

