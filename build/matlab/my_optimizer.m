function my_optimizer (f, lower_bounds, upper_bounds, budget)
    n = length(lower_bounds);
    delta = upper_bounds - lower_bounds;
    x = lower_bounds + normrnd(zeros(1, n), 1) * delta;
    for i= 1:budget
        y = f(x);
    end


end

