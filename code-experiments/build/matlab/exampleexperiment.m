%
% This script runs random search for BUDGET_MULTIPLIER*DIM function
% evaluations on the 'bbob-constrained' suite.
%
% An example experiment on the single-objective 'bbob' suite can be started
% by renaming the suite_name below.
%
% This example experiment allows also for easy implementation of independent
% restarts by simply increasing NUM_OF_INDEPENDENT_RESTARTS. To make this
% effective, the algorithm should have at least one more stopping criterium
% than just a maximal budget.
%
more off; % to get immediate output in Octave

%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%
BUDGET_MULTIPLIER = 2; % algorithm runs for BUDGET_MULTIPLIER*dimension funevals
NUM_OF_INDEPENDENT_RESTARTS = 1e9; % max. number of independent algorithm
% restarts; if >0, make sure that the
% algorithm is not always doing the same thing
% in each run (which is typically trivial for
% randomized algorithms)

%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare Experiment    %
%%%%%%%%%%%%%%%%%%%%%%%%%
suite_name = 'bbob-constrained'; % works for 'bbob' as well
observer_name = 'bbob';
observer_options = strcat('result_folder: RS_on_', ...
    suite_name, ...
    [' algorithm_name: RS '...
    ' algorithm_info: A_simple_random_search ']);

% dimension 40 is optional:
suite = cocoSuite(suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
observer = cocoObserver(observer_name, observer_options);

% set log level depending on how much output you want to see, e.g. 'warning'
% for fewer output than 'info'.
cocoSetLogLevel('info');

% keep track of problem dimension and #funevals to print timing information:
printeddim = 1;
doneEvalsAfter = 0; % summed function evaluations for a single problem
doneEvalsTotal = 0; % summed function evaluations per dimension
printstring = '\n'; % store strings to be printed until experiment is finished

%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Experiment        %
%%%%%%%%%%%%%%%%%%%%%%%%%
while true
    % get next problem and dimension from the chosen suite:
    problem = cocoSuiteGetNextProblem(suite, observer);
    if ~cocoProblemIsValid(problem)
        break;
    end
    dimension = cocoProblemGetDimension(problem);
    
    % printing
    if printeddim < dimension
      if printeddim > 1
        elapsedtime = toc;
        printstring = strcat(printstring, ...
            sprintf('   COCO TIMING: dimension %d finished in %e seconds/evaluation\n', ...
            printeddim, elapsedtime/double(doneEvalsTotal)));
        tic;
      end
      doneEvalsTotal = 0;
      printeddim = dimension;
      tic;
    end  
    
    % restart functionality: do at most NUM_OF_INDEPENDENT_RESTARTS+1
    % independent runs until budget is used:
    i = -1; % count number of independent restarts
    while (BUDGET_MULTIPLIER*dimension > (cocoProblemGetEvaluations(problem) + ...
                                          cocoProblemGetEvaluationsConstraints(problem)))
        i = i+1;
        if (i > 0)
            fprintf('INFO: algorithm restarted\n');
        end
        doneEvalsBefore = cocoProblemGetEvaluations(problem) + ...
                          cocoProblemGetEvaluationsConstraints(problem);
        
        % start algorithm with remaining number of function evaluations:
        my_optimizer(problem,...
            cocoProblemGetSmallestValuesOfInterest(problem),...
            cocoProblemGetLargestValuesOfInterest(problem),...
            BUDGET_MULTIPLIER*dimension - doneEvalsBefore);
        
        % check whether things went wrong or whether experiment is over:
        doneEvalsAfter = cocoProblemGetEvaluations(problem) + ...
                         cocoProblemGetEvaluationsConstraints(problem);
        if cocoProblemFinalTargetHit(problem) == 1 ||...
                doneEvalsAfter >= BUDGET_MULTIPLIER * dimension
            break;
        end
        if (doneEvalsAfter == doneEvalsBefore)
            fprintf('WARNING: Budget has not been exhausted (%d/%d evaluations done)!\n', ....
                doneEvalsBefore, BUDGET_MULTIPLIER * dimension);
            break;
        end
        if (doneEvalsAfter < doneEvalsBefore)
            fprintf('ERROR: Something weird happened here which should not happen: f-evaluations decreased');
        end
        if (i >= NUM_OF_INDEPENDENT_RESTARTS)
            break;
        end
    end
    
    doneEvalsTotal = doneEvalsTotal + doneEvalsAfter;
end

elapsedtime = toc;
printstring = strcat(printstring, ...
    sprintf('   COCO TIMING: dimension %d finished in %e seconds/evaluation\n', ...
    printeddim, elapsedtime/double(doneEvalsTotal)));
fprintf(printstring);

cocoObserverFree(observer);
cocoSuiteFree(suite);
