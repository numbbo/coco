%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compiles the Coco functions in the Matlab/Octave wrapper  %
% to be able to run the exampleexperiment.m                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
more off; % turn off page-wise output

fprintf('compiling cocoCall.c...');
mex('-Dchar16_t=uint16_t', 'cocoCall.c');
fprintf('Done\n');
fprintf('Preparation of all mex files finished.\n');