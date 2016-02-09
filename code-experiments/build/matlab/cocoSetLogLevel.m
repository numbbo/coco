% Sets the COCO log level to the given value and returns its previous value.
%
% Parameters:
%    log_level	Denotes the level of information given to the user. Can take on
%               the string values:
%               "error"   (only error messages are output),
%               "warning" (only error and warning messages are output),
%               "info"    (only error, warning and info messages are output) and
%               "debug"   (all messages are output).
%               "" does not set a new value. The default value is info.
%
% Returns:
%    The previous coco_log_level value as an immutable string. 
%
%
% Example usage:
%
%   >> suite = cocoCall('cocoSuite', 'bbob-biobj', 'year: 2016', 'dimensions: 2,3,5,10,20,40');
%   >> observer = cocoCall('cocoObserver', 'bbob-biobj', 'result_folder: test');
%   COCO INFO: Results will be output to folder exdata\test
%   >> problem = cocoCall('cocoSuiteGetNextProblem', suite, observer);
%   COCO INFO: 09.02.16 16:31:16, d=2, running: f01.>> cocoCall('cocoSetLogLevel', 'warning')
%   ans = info
%   >> problem = cocoCall('cocoSuiteGetNextProblem', suite, observer);
%   >> cocoCall('cocoSetLogLevel', 'info')
%   ans = warning
%   .>> cocoCall('cocoObserverFree', suite);
%   >> cocoCall('cocoSuiteFree', suite);
%
%
