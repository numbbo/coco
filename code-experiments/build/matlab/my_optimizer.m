function my_optimizer (problem, lower_bounds, upper_bounds, budget)
n = length(lower_bounds);
delta = upper_bounds - lower_bounds;
for i= 1:budget
    x = lower_bounds + rand(1,n) .* delta;
    cocoEvaluateFunction(problem, x);
    if cocoProblemGetNumberOfConstraints(problem) > 0
        cocoEvaluateConstraint(problem, x)
    end
end
