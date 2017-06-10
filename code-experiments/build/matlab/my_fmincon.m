%
% The code below calls MATLAB solver, fmincon, on bbob-constrained testbed
% cons is defined in cons.m
%

function my_fmincon (problem, lower_bounds, upper_bounds, budget)
n = length(lower_bounds);
delta = upper_bounds - lower_bounds;
x0 = lower_bounds + rand(1,n) .* delta;
options = optimset('Display', 'off', 'MaxFunEvals', budget);
x = fmincon(@(x)cocoEvaluateFunction(problem, x), x0, [], [], [], [], [], [], @(x)cons(problem, x), options);
end
