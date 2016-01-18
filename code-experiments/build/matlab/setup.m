%%%%%%%%%%%%%%%%%
% Octave syntax %
%%%%%%%%%%%%%%%%%

%mkoctfile --mex -Dchar16_t=uint16_t cocoEvaluateFunction.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemAddObserver.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemFree.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoObserver.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoObserverFree.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemGetDimension.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemGetEvaluations.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemGetId.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemGetLargestValuesOfInterest.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemGetName.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemGetNumberOfObjectives.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemGetSmallestValuesOfInterest.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoProblemIsValid.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoSuiteGetProblem.c
%mkoctfile --mex -Dchar16_t=uint16_t cocoSuiteGetNextProblemIndex.c




%%%%%%%%%%%%%%%%%
% MATLAB syntax %
%%%%%%%%%%%%%%%%%
more off; % turn off page-wise output
tocompilecoco = {'cocoSuite.c', ...
                 'cocoSuiteFree.c', ...
                 'cocoSuiteGetNextProblem.c', ...
                 'cocoEvaluateFunction.c', ...
                 'cocoObserver.c', ...
                 'cocoObserverFree.c', ...
                 'cocoProblemGetDimension.c', ...
                 'cocoProblemGetEvaluations.c', ...
                 'cocoProblemGetId.c', ...
                 'cocoProblemGetLargestValuesOfInterest.c', ...
                 'cocoProblemGetName.c', ...
                 'cocoProblemGetNumberOfObjectives.c', ...
                 'cocoProblemGetSmallestValuesOfInterest.c', ...
                 'cocoProblemIsValid.c'};

for i = 1:length(tocompilecoco)
    printf('compiling %s...', tocompilecoco{i});
    mex('-Dchar16_t=uint16_t', tocompilecoco{i});
    printf('Done\n');
end