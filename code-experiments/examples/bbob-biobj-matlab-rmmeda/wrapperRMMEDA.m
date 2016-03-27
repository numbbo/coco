function [F, V] = wrapperRMMEDA (f, X)
% Assumes a multi-objective problem of the following form
% Min   F(X) = (f1(X),f2(X),...,fm(X))
%   s.t: hi(X) =  0 , i=1,2,...,k
%        hj(X) <= 0 , j=k+1,...,p
% X is decision vector,
% F is objective vector,
% V = |h1(X)|+ |h2(X)| + ... + |hk(X)| + max(0,hk+1(X)) + ... + max(0,hp(X)
% is constraint violation
x = X';
F = cocoEvaluateFunction(f, x);
V = 0;