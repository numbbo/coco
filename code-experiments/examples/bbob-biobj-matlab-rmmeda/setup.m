%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compiles the Coco functions in the Matlab/Octave wrapper       %
% to be able to run the SMS-EMOA via run_smsemoa_on_bbob_biobj.m %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
more off; % turn off page-wise output

fprintf('compiling cocoCall.c...');
mex('-Dchar16_t=uint16_t', 'cocoCall.c');

fprintf('compiling MOSelector.cpp...');
mex MOSelector.cpp;
fprintf('Done\n');

fprintf('compiling ParetoFilter.cpp...');
mex ParetoFilter.cpp;
fprintf('Done\n');

fprintf('Preparation of all mex files finished.\n');