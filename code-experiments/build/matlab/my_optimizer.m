function my_optimizer (problem, lower_bounds, upper_bounds, num_integer_vars, num_constraints, budget)
n = length(lower_bounds);
delta = upper_bounds - lower_bounds;
for i= 1:budget
    x = lower_bounds + rand(1,n) .* delta;
    % Round the variable values that need to be integer (not really needed)
    if num_integer_vars > 0
        x(1:num_integer_vars) = round(x(1:num_integer_vars));
    end
    if num_constraints > 0
        cocoEvaluateConstraint(problem, x);
    end
    cocoEvaluateFunction(problem, x);
end
