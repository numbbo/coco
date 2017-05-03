%function my_optimizer (problem, lower_bounds, upper_bounds, budget)
%n = length(lower_bounds);
%delta = upper_bounds - lower_bounds;
%for i= 1:budget
%    x = lower_bounds + rand(1,n) .* delta;
%    cocoEvaluateFunction(problem, x);
%    if cocoProblemGetNumberOfConstraints(problem) > 0
%        cocoEvaluateConstraint(problem, x)
%    end
%end


function my_optimizer (problem, lower_bounds, upper_bounds, budget)
n = length(lower_bounds);
delta = upper_bounds - lower_bounds;
x0 = lower_bounds + rand(1,n) .* delta;
options = optimoptions('fmincon', 'Display', 'iter', 'MaxFunctionEvaluations', budget);
x = fmincon(@(x)cocoEvaluateFunction(problem, x), x0, [], [], [], [], lower_bound, upper_bound, @(x)cocoEvaluateConstraint(problem, x), options);
end
