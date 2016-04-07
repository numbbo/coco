% Runs the SMS-EMOA on the bbob-biobj test suite.
%
% This example code allows also for easy implementation of independent
% restarts by simply increasing NUM_OF_INDEPENDENT_RESTARTS and setting
% opts.useOCD = true within my_smsemoa.m.
%
more off; % to get immediate output in Octave

%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%
BUDGET_MULTIPLIER = 2; % algorithm runs for BUDGET_MULTIPLIER*dimension funevals
NUM_OF_INDEPENDENT_RESTARTS = 1e9; % number of independent algorithm restarts;
                                   % if >0, make sure that the algorithm is not
                                   % always doing the same thing in each run
                                   % (typically trivial for randomized algorithms)

%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare Experiment    %
%%%%%%%%%%%%%%%%%%%%%%%%%                           
suite_name = 'bbob-biobj';
observer_name = suite_name;
observer_options = ['result_folder: SMSEMOA_on_bbob-biobj \',...
                    'algorithm_name: SMS-EMOA \',...
                    'algorithm_info: "SMS-EMOA without restarts" \', ...
                    'log_level: COCO_INFO'];      
                
% dimension 40 is optional:
suite = cocoCall('cocoSuite', suite_name, 'year: 2016', 'dimensions: 2,3,5,10,20,40');
observer = cocoCall('cocoObserver', observer_name, observer_options);

% set log level depending on how much output you want to see, e.g. 'warning'
% for fewer output than 'info'.
cocoCall('cocoSetLogLevel', 'info');

% keep track of problem dimension and #funevals to print timing information:
printeddim = 1;
doneEvalsAfter = 0; % summed function evaluations for a single problem
doneEvalsTotal = 0; % summed function evaluations per dimension


%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Experiment        %
%%%%%%%%%%%%%%%%%%%%%%%%%
while true
    % get next problem and dimension from the chosen suite:
    problem = cocoCall('cocoSuiteGetNextProblem', suite, observer);
    if (~cocoCall('cocoProblemIsValid', problem))
        break;
    end
    dimension = cocoCall('cocoProblemGetDimension', problem);
    
    % printing timing information
    if printeddim < dimension
      if printeddim > 1
        elapsedtime = toc;
        fprintf('\n   COCO TIMING: dimension %d finished in %e seconds/evaluation\n', ...
                printeddim, elapsedtime/double(doneEvalsTotal));
        tic;
      end
      doneEvalsTotal = 0;
      printeddim = dimension;
      tic;
    end  
    
    
    % restart functionality: do at most NUM_OF_INDEPENDENT_RESTARTS+1
    % independent runs until budget is used:
    i = -1; % count number of independent restarts
    while BUDGET_MULTIPLIER*dimension > cocoCall('cocoProblemGetEvaluations', problem)
        i = i+1;
        if (i > 0)
            fprintf('INFO: algorithm restarted');
        end
        doneEvalsBefore = cocoCall('cocoProblemGetEvaluations', problem);
        
        % start algorithm with remaining number of function evaluations:
        my_smsemoa(problem, ...
            cocoCall('cocoProblemGetSmallestValuesOfInterest', problem), ...
            cocoCall('cocoProblemGetLargestValuesOfInterest', problem), ...
            BUDGET_MULTIPLIER*dimension - doneEvalsBefore);
            
        % check whether things went wrong or whether experiment is over:
        doneEvalsAfter = cocoCall('cocoProblemGetEvaluations', problem);
        if (cocoCall('cocoProblemFinalTargetHit', problem) == 1) || (doneEvalsAfter >= BUDGET_MULTIPLIER * dimension)
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
end

elapsedtime = toc;
fprintf('\n   COCO TIMING: dimension %d finished in %e seconds/evaluation\n', ...
        printeddim, elapsedtime/double(doneEvalsTotal));

cocoCall('cocoObserverFree', observer);
cocoCall('cocoSuiteFree', suite);
