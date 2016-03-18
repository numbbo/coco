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
function ll = cocoSetLogLevel(log_level)
ll = cocoCall('cocoSetLogLevel', log_level);