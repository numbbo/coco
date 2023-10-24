% The main entry point of the Coco platform from Matlab/GNU Octave. 
%
% Provides most of the basic functionality of the COCO C code via
% a cocoCall function that takes a string, naming the C function to be called
% (without underscores), as the first argument.
%
% Usage:
%
%   > res = cocoCall('COCOFUNCTION', varargs);
%
% where COCOFUNCTION and the variable argument list are one of the following:
%
%   * cocoEvaluateFunction: problem, x, y 
%       Evaluates the objective function at point x and saves the result in y. 
%   * cocoEvaluateConstraint: problem, x, y 
%       Evaluates the constraints at point x and saves the result in y. 
%   * cocoRecommendSolution: problem, x
%       Recommends solution x (logs the function values, but does not return them).
%   * cocoObserver: observer_name, observer_options
%       Returns a new COCO observer. 
%   * cocoObserverFree: observer
%       Frees the given observer.
%   * cocoObserverSignalRestart: observer, problem
%       Signals a restart by the algoirthm.
%   * cocoProblemAddObserver: problem, observer
%       Adds an observer to the given problem and returns the resulting problem.
%   * cocoProblemFinalTargetHit: problem
%       Returns 1 if the final target was hit on given problem, 0 otherwise.
%   * cocoProblemFree: problem
%       Frees the given problem.
%   * cocoProblemGetDimension: problem
%       Returns the number of variables i.e. the dimension of the problem.
%   * cocoProblemGetEvaluations: problem
%       Returns the number of evaluations done on the problem.
%   * cocoProblemGetId: problem
%       Returns the ID of the problem. 
%   * cocoProblemGetInitialSolution: problem
%       Returns an initial solution (ie a feasible variable setting) to problem.
%   * cocoProblemGetLargestFValuesOfInterest: problem
%       For multi-objective problems, returns a vector of largest values of 
%         interest in each objective. Currently, this equals the nadir point. 
%         For single-objective problems it raises an error.
%   * cocoProblemGetLargestValuesOfInterest: problem
%       Returns a vector of size 'dimension' with upper bounds of the region
%         of interest in the decision space for the given problem.
%   * cocoProblemGetName: problem
%       Returns the name of the problem. 
%   * cocoProblemGetNumberOfObjectives: problem
%       Returns the number of objectives of the problem.
%   * cocoProblemGetNumberOfConstraints: problem
%       Returns the number of constraints of the problem.
%   * cocoProblemGetNumberOfIntegerVariables: problem
%       Returns the number of integer variables of the problem.
%   * cocoProblemGetSmallestValuesOfInterest: problem
%       Returns a vector of size 'dimension' with lower bounds of the region
%         of interest in the decision space for the given problem.
%   * cocoProblemIsValid: problem
%       Returns 1 if the given problem is a valid Coco problem, 0 otherwise.
%   * cocoProblemRemoveObserver: problem, observer
%       Removes an observer from the given problem and returns the inner problem.
%   * cocoSetLogLevel: log_level 	
%       Sets the COCO log level to the given value (a string) and returns the
%         previous value. 
%   * cocoSuite: suite_name, suite_instance, suite_options 
%       Returns a new suite.
%   * cocoSuiteFree: suite
%       Frees the given suite (no return value).
%   * cocoSuiteGetNextProblem: suite, observer 
%       Returns the next (observed) problem of the suite or NULL if there is
%         no next problem left. 
%   * cocoSuiteGetProblem: suite, problem_index
%       Returns the problem of the suite defined by problem_index. 
%
% For a more detailed help, type 'help COCOFUNCTION' or 'doc COCOFUNCTION'.
% 
% For more information on the Coco C functions, see <a href="matlab: 
% web('http://numbbo.github.io/coco-doc/C')">the online Coco C documentation</a>.
