function [] = paretofront(varargin)
% PARETOFRONT returns the logical Pareto Front of a set of points.
% 
%       synopsis:   front = paretofront(M)
%
%
%   INPUT ARGUMENT
%
%       - M         n x m array, of which (i,j) element is the j-th objective
%                   value of the i-th point;
%
%   OUTPUT ARGUMENT
%
%       - front     n x 1 logical vector to indicate if the corresponding 
%                   points are belong to the front (true) or not (false).
%
% By Yi Cao at Cranfield University, 31 October 2007
%
% Example 1: Find the Pareto Front of a circumference
% 
% alpha = [0:.1:2*pi]';
% x = cos(alpha);
% y = sin(alpha);
% front = paretofront([x y]);
% 
% hold on;
% plot(x,y);
% plot(x(front), y(front) , 'r');
% hold off
% grid on
% xlabel('x');
% ylabel('y');
% title('Pareto Front of a circumference');
% 
% Example 2:  Find the Pareto Front of a set of 3D random points
% X = rand(100,3);
% front = paretofront(X);
% 
% hold on;
% plot3(X(:,1),X(:,2),X(:,3),'.');
% plot3(X(front, 1) , X(front, 2) , X(front, 3) , 'r.');
% hold off
% grid on
% view(-37.5, 30)
% xlabel('X_1');
% ylabel('X_2');
% zlabel('X_3');
% title('Pareto Front of a set of random points in 3D');
% 
% 
% Example 3: Find the Pareto set of a set of 1000000 random points in 4D
%            The machine performing the calculations was a 
%            Intel(R) Core(TM)2 CPU T2500 @ 2.0GHz, 2.0 GB of RAM
%            
% X = rand(1000000,4);
% t = cputime;
% paretofront(X);
% cputime - t
% 
% ans =
% 
%    1.473529

error('mex file absent, type ''mex paretofront.c'' to compile');
