function my_smsemoa (f, lower_bounds, upper_bounds, budget)
my_f = @(x)cocoEvaluateFunction(f, x);
opts = SMSEMOA;
opts.useOCD = false;
opts.maxEval = budget;
SMSEMOA(my_f, lower_bounds, upper_bounds, opts);