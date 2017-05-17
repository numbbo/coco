function [c,ceq] = cons(problem, x)
c = cocoEvaluateConstraint(problem, x);
ceq = [];
end