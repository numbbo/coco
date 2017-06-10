%
% asma: 'cons' for constraint function
% This is a wrapper around cocoEvaluateConstraint(...) so that it can be
% used with fmincon
% The function cons evaluates the constraint functions of problem
% at x and returns:
%   c: vector of inequality constraint values
%   ceq: vector of equality constraint values
% Only inequality constraints in bbob-constrained testbed
%

function [c, ceq] = cons(problem, x)
c = cocoEvaluateConstraint(problem, x);
ceq = [];
end