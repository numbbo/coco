% Returns a new COCO suite.
%
% Currently, six suites are supported:
%
%    "bbob" contains 24 single-objective functions in 6 dimensions (2, 3, 5, 10, 20, 40)
%    "bbob-biobj" contains 55 bi-objective functions in 6 dimensions (2, 3, 5, 10, 20, 40)
%    "bbob-largescale" contains 24 single-objective functions in 6 large dimensions (40, 80, 160, 320, 640, 1280)
%    "bbob-mixint" contains 24 mixed-integer single-objective functions in 6 dimensions (2, 3, 5, 10, 20, 40)
%    "bbob-biobj-mixint" contains 92 mixed-integer bi-objective functions in 6 dimensions (2, 3, 5, 10, 20, 40)
%    "toy" contains 6 single-objective functions in 5 dimensions (2, 3, 5, 10, 20)
%
% Only the suite_name parameter needs to be non-empty. The suite_instance and
% suite_options can be "" or NULL. In this case, default values are taken
% (default instances of a suite are those used in the last year and the suite is
% not filtered by default).
%
% Parameters
%   suite_name      A string containing the name of the suite. Currently
%                   supported suite names are "bbob", "bbob-biobj",
%                   "bbob-largescale" and "toy".
%   suite_instance  A string used for defining the suite instances. Two ways are
%                   supported:
%                   "year: YEAR", where YEAR is the year of the BBOB workshop,
%                                 includes the instances (to be) used in that
%                                 year's workshop;
%                   "instances: VALUES", where VALUES are instance numbers from
%                                 1 on written as a comma-separated list or a
%                                 range m-n.
%   suite_options   A string of pairs "key: value" used to filter the suite
%                   (especially useful for parallelizing the experiments).
%                   Supported options:
%                   "dimensions: LIST", where LIST is the list of dimensions to
%                                 keep in the suite (range-style syntax is not
%                                 allowed here),
%                   "dimension_indices: VALUES", where VALUES is a list or a
%                                 range of dimension indices (starting from 1)
%                                 to keep in the suite,
%                   "function_indices: VALUES", where VALUES is a list or a
%                                 range of function indices (starting from 1) to
%                                 keep in the suite, and
%                   "instance_indices: VALUES", where VALUES is a list or a
%                                 range of instance indices (starting from 1) to
%                                 keep in the suite.
function suite = cocoSuite(suite_name, suite_instance, suite_options)
suite = cocoCall('cocoSuite', suite_name, suite_instance, suite_options);
