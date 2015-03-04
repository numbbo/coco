function my_optimizer (f, lower_bounds, upper_bounds, budget)
    n = length(lower_bounds);
    delta = upper_bounds - lower_bounds;
    for i= 1:budget
        x = lower_bounds + normrnd(zeros(1, n), 1) .* delta;
        y = cocoEvaluateFunction(f, x);
    end
end

