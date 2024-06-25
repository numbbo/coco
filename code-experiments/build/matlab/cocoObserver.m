% Returns a new COCO observer.
%
% Currently, four observers are supported:
%
%    "bbob" is the observer for single-objective (both noisy and noiseless)
%                 problems with known optima, which creates *.info, *.dat,
%                 *.tdat and *.rdat files and logs the distance to the optimum.
%    "bbob-biobj" is the observer for bi-objective problems, which creates
%                 *.info, *.dat and *.tdat files for the given indicators, as
%                 well as an archive folder with *.adat files containing
%                 nondominated solutions.
%    "rw" is an observer for single- and bi-objective real-world problems that
%         logs all information (can be configured to long only some information)
%         and produces *.txt files (not readable by post-processing).
%    "toy" is a simple observer that logs when a target has been hit.
%
% Parameters:
%    observer_name     A string containing the name of the observer. Currently
%                      supported observer names are "bbob", "bbob-biobj", "rw",
%                      and "toy".
%                      Strings "no_observer", "" or NULL return NULL.
%    observer_options  A string of pairs "key: value" used to pass the options
%                      to the observer. Some observer options are general, while
%                      others are specific to some observers. Here we list only
%                      the general options, see observer_bbob, observer_biobj,
%                      and observer_toy for options of the specific observers.
%
%                      "outer_folder: NAME" determines the outer folder for the
%                              experiment. The default value is "exdata".
%                      "result_folder: NAME" determines the folder within the
%                              "exdata" folder into which the results will be
%                              output. If the folder with the given name already
%                              exists, first NAME_001 will be tried, then
%                              NAME_002 and so on. The default value is
%                              "default".
%                      "algorithm_name: NAME", where NAME is a short name of the
%                              algorithm that will be used in plots (no spaces
%                              are allowed). The default value is "ALG".
%                      "algorithm_info: STRING" stores the description of the
%                              algorithm. If it contains spaces, it must be
%                              surrounded by double quotes. The default value is
%                              "" (no description).
%                      "number_target_triggers: VALUE" defines the number of
%                              targets between each 10**i and 10**(i+1) (equally
%                              spaced in the logarithmic scale) that trigger
%                              logging. The default value is 10.
%                      "log_target_precision: VALUE" defines the precision used
%                              for logarithmic targets (there are no targets for
%                              abs(values) < log_target_precision). The default
%                              value is 1e-8.
%                      ""lin_target_precision: VALUE" defines the precision used
%                              for linear targets. The default value is 1e-5.
%                      "number_evaluation_triggers: VALUE" defines the number of
%                              evaluations to be logged between each 10**i and
%                              10**(i+1). The default value is 20.
%                      "base_evaluation_triggers: VALUES" defines the base
%                              evaluations used to produce an additional
%                              evaluation-based logging. The numbers of
%                              evaluations that trigger logging are every
%                              base_evaluation * dimension * (10**i). For
%                              example, if base_evaluation_triggers = "1,2,5",
%                              the logger will be triggered by evaluations
%                              dim*1, dim*2, dim*5, 10*dim*1, 10*dim*2,
%                              10*dim*5, 100*dim*1, 100*dim*2, 100*dim*5, ...
%                              The default value is "1,2,5".
%                      "precision_x: VALUE" defines the precision used when
%                              outputting variables and corresponds to the
%                              number of digits to be printed after the decimal
%                              point. The default value is 8.
%                      "precision_f: VALUE" defines the precision used when
%                              outputting f values and corresponds to the number
%                              of digits to be printed after the decimal point.
%                              The default value is 15.
%                      "log_discrete_as_int: VALUE" determines whether the values 
%                              of integer variables (in mixed-integer problems)
%                              are logged as integers (1) or not (0 - in this case 
%                              they are logged as doubles). The default value is 0.
%
% Returns:
%   The constructed observer object or NULL if observer_name equals NULL, "" or
%   "no_observer".
function observer = cocoObserver(observer_name, observer_options)
observer = cocoCall('cocoObserver', observer_name, observer_options);