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

mex -Dchar16_t=uint16_t cocoEvaluateFunction.c
mex -Dchar16_t=uint16_t cocoSuite.c
mex -Dchar16_t=uint16_t cocoSuiteFree.c
mex -Dchar16_t=uint16_t cocoObserver.c
mex -Dchar16_t=uint16_t cocoObserverFree.c
mex -Dchar16_t=uint16_t cocoSuiteGetNextProblem.c
mex -Dchar16_t=uint16_t cocoProblemGetDimension.c
mex -Dchar16_t=uint16_t cocoProblemGetEvaluations.c
mex -Dchar16_t=uint16_t cocoProblemGetId.c
mex -Dchar16_t=uint16_t cocoProblemGetLargestValuesOfInterest.c
mex -Dchar16_t=uint16_t cocoProblemGetName.c
mex -Dchar16_t=uint16_t cocoProblemGetNumberOfObjectives.c
mex -Dchar16_t=uint16_t cocoProblemGetSmallestValuesOfInterest.c
mex -Dchar16_t=uint16_t cocoProblemIsValid.c

mex -I. hv.cpp Hypervolume.cpp
mex paretofront.c