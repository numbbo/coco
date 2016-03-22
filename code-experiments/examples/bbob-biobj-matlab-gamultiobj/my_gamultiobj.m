function my_gamultiobj (f, lower_bounds, upper_bounds, budget)
n = length(lower_bounds);
options = gaoptimset(@gamultiobj);
options = gaoptimset(options, 'Display', 'off', 'PopulationSize',...
    double(min(100,budget)), 'Generations', double(ceil(budget./100)));
gamultiobj(@(x)cocoEvaluateFunction(f, x), n, [], [], [], [],...
    lower_bounds, upper_bounds, options);