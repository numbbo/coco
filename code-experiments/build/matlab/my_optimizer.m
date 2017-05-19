function my_optimizer (problem, lower_bounds, upper_bounds, budget)
n = length(lower_bounds);
delta = upper_bounds - lower_bounds;
for i= 1:budget
    x = lower_bounds + rand(1,n) .* delta;
    cocoEvaluateFunction(problem, x);
    if cocoProblemGetNumberOfConstraints(problem) > 0
        cocoEvaluateConstraint(problem, x)
        cocoProblemGetName(problem)
    end
end

%
% asma: The code below calls MATLAB solver, fmincon, on bbob-constrained testbed
% cons is defined in cons.m
%

% function my_optimizer (problem, lower_bounds, upper_bounds, budget)
% n = length(lower_bounds);
% delta = upper_bounds - lower_bounds;
% x0 = lower_bounds + rand(1,n) .* delta;
% options = optimset('Display', 'off', 'MaxFunEvals', budget);
% cocoProblemGetId(problem) % print id of currently minimized problem
% x = fmincon(@(x)cocoEvaluateFunction(problem, x), x0, [], [], [], [], [], [], @(x)cons(problem, x), options);
% end
