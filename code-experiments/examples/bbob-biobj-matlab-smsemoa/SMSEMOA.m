function [paretoFront, ...   % objectives
    paretoSet]... % parameters
    = SMSEMOA(...
    problem, ...             % function handle to the objective function
    rngMin, ...              % lower bound of decision variables
    rngMax, ...              % upper bound of decision variables
    inopts, ...              % struct with options (optional)
    initPop)                 % initial population (optional)
% smsemoa.m, Version 1.0, last change: August, 14, 2008
% SMS-EMOA implements the S-Metric-Section-based Evolutionary
% Multi-Objective Algorithm for nonlinear vector minimization.
%
% OPTS = SMSEMOA returns default options.
% OPTS = SMSEMOA('defaults') returns default options quietly.
% OPTS = SMSEMOA('displayoptions') displays options.
% OPTS = SMSEMOA('defaults', OPTS) supplements options OPTS with default
% options.
%
% function call:
% [PARETOFRONT, PARETOSET] = SMSEMOA(PROBLEM[, OPTS])
%
% Input arguments:
%  PROBLEM is a function handle or a string function name like 'Sympart'.
%  PROBLEM.m takes as argument a row vector of parameters and returns
%  a row vector of objectives
%  OPTS (an optional argument) is a struct holding additional input
%     options. Valid field names and a short documentation can be
%     discovered by looking at the default options (type 'smsemoa'
%     without arguments, see above). Empty or missing fields in OPTS
%     invoke the default value, i.e. OPTS needs not to have all valid
%     field names.  Capitalization does not matter and unambiguous
%     abbreviations can be used for the field names. If a string is
%     given where a numerical value is needed, the string is evaluated
%     by eval, where
%     'nVar' expands to the problem dimension
%     'nObj' expands to the objectives dimension
%     'nPop' expands to the population size
%     'countEval' expands to the number of the recent evaluation
%     'nPV' expands to the number paretofronts
%
% Output:
%  PARETOFRONT is a struct holding the objectives in rows. Each row holds
%     the results of the objective function of one solution
%  PARETOSET is a struct holding the parameters. Each row holds one
%     solution.
%
% This software is Copyright (C) 2008
% Tobias Wagner, Fabian Kretzschmar
% ISF, TU Dortmund
% February 3, 2016
%
% This program is free software (software libre); you can redistribute it
% and/or modify it under the terms of the GNU General Public License as
% published by the Free Software Foundation; either version 2 of the
% License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
% Public License for more details.
%
% implementation based on [1][2] using
% *  Computation of the Hypervolume Indicator based on [3]
%    http://sbe.napier.ac.uk/~manuel/hypervolume
% *  Pareto Front Algorithms
%    http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?object
%    Id=17251&objectType=file
% *  coding-fragments from NSGA - II
%    http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?object
%    Id=10429&objectType=file
%
% [1] Michael Emmerich, Nicola Beume, and Boris Naujoks. An EMO algorithm
% using the hypervolume measure as selection criterion. In C. A. Coello
% Coello et al., Eds., Proc. Evolutionary Multi-Criterion Optimization,
% 3rd Int'l Conf. (EMO 2005), LNCS 3410, pp. 62-76. Springer, Berlin, 2005.
%
% [2] Boris Naujoks, Nicola Beume, and Michael Emmerich. Multi-objective
% optimisation using S-metric selection: Application to three-dimensional
% solution spaces. In B. McKay et al., Eds., Proc. of the 2005 Congress on
% Evolutionary Computation (CEC 2005), Edinburgh, Band 2, pp. 1282-1289.
% IEEE Press, Piscataway NJ, 2005.
%
% [3] Carlos M. Fonseca, Luís Paquete, and Manuel López-Ibáñez. An improved
% dimension-sweep algorithm for the hypervolume indicator.  In IEEE
% Congress on Evolutionary Computation, pages 3973-3979, Vancouver, Canada,
% July 2006.

% ----------- Set Defaults for Options ---------------------------------
% options: general - these are evaluated once
defopts.nPop              = '100           % size of the population';
defopts.maxEval           = 'inf           % maximum number of evaluations';
defopts.useOCD            = 'true          % use OCD to detect convergence';
defopts.OCD_VarLimit      = '1e-10         % variance limit of OCD';
defopts.OCD_nPreGen       = '15            % number of preceding generations used in OCD';
defopts.nPFevalHV         = 'inf           % evaluate 1st to this number paretoFronts with HV';
defopts.outputGen         = 'inf           % rate of writing output files';
defopts.outputType        = '0             % type of output (0 none, 1 population, 2 archive)';

% options: generation of offsprings - these are evaluated each run
defopts.var_crossover_prob= '0.9           % [0.8, 1] % variable crossover probability';
defopts.var_crossover_dist= '15            % distribution index for crossover';
defopts.var_mutation_prob = '1./nVar       % variable mutation probability';
defopts.var_mutation_dist = '20            % distribution index for mutation';
defopts.var_swap_prob     = '0.5           % variable swap probability';
defopts.DE_F              = '0.2+rand(1).*0.6% difference weight for DE';
defopts.DE_CR             = '0.9           % crossover probability for differential evo';
defopts.DE_CombinedCR     = 'true          % crossover of blocks instead of single variables';
defopts.useDE             = 'false         % perform differential evo instead of SBX&PM';
defopts.refPoint          = '0             % refPoint for HV; if 0, max(obj)+1 is used';

% ---------------------- Handling Input Parameters ----------------------

if nargin < 1 || isequal(problem, 'defaults') % pass default options
    paretoFront = defopts;
    if nargin > 1 % supplement second argument with default options
        paretoFront = getoptions(inopts, defopts);
    end
    return;
end

if isequal(problem, 'displayoptions')
    names = fieldnames(defopts);
    for name = names'
        disp([name{:} repmat(' ', 1, 20-length(name{:})) ': ''' defopts.(name{:}) '''']);
    end
    return;
end

if ~ischar(problem) && ~isa(problem, 'function_handle')
    error('first argument ''problem'' must be a string or a fhandle');
end

if nargin < 3
    error('problem, rngMin, and rngMax are required');
end;

% compose options opts
if nargin < 4 || isempty(inopts) % no input options available
    opts = defopts;
else
    opts = getoptions(inopts, defopts);
end

% initialize init pop param
if nargin < 5
    initPop = '';
end;

% ------------------------ Initialization -------------------------------

% reset the random number generator to a different state each restart
[v, d] = version;
if str2double(d(end-4:end)) > 2011
    rng('default');
    ropt = rng('shuffle');
    seed = ropt.Seed;
else
    d = clock;
    seed = double(ceil(d(end)*1e9));
    rand('seed', seed);
end;

% initialize auxiliary parameters
nVar = length(rngMin);
nObj = length(feval(problem, rngMin));

% get parameters for initialization
nPop = myeval(opts.nPop);
nPV = ceil((1/(2^(nObj-1)))*nPop); % guess number of Pareto-Ranks
maxEval = myeval(opts.maxEval);
useOCD = myeval(opts.useOCD);
OCD_VarLimit = myeval(opts.OCD_VarLimit);
OCD_nPreGen = myeval(opts.OCD_nPreGen);
nPFevalHV = myeval(opts.nPFevalHV);
outputGen = myeval(opts.outputGen);
outputType = myeval(opts.outputType);

% prepare save directories
if isa(problem, 'function_handle')
    nameProblem = func2str(problem);
    i = strfind(nameProblem, ')');
    k = strfind(nameProblem, '(');
    nameProblem = nameProblem(i(1)+1:k(2)-1);
else
    nameProblem = problem;
end;
outdir = sprintf('SMSEMOA_%s/%u/', nameProblem, seed);
if outputType>0 && ~isinf(outputGen) && ...
        ~isdir(sprintf('SMSEMOA_%s', nameProblem))
    mkdir(sprintf('SMSEMOA_%s', nameProblem));
end;
if outputType>0 && ~isinf(outputGen) && ~isdir(outdir)
    mkdir(outdir);
end;

% calculate initial sampling
ranks = inf(nPop+1,1);
population = initialize_variables(nPop, nObj, nVar, rngMin, ...
    rngMax, problem, initPop);
countEval = nPop;

% initialize new Element position
elementInd = nPop+1;

% write output files
if mod(countEval, outputGen)==0 && outputType > 0
    writeToFile(population, nPop, elementInd, nVar, nObj, ranks,...
        countEval, outdir)
end;

% OCD data structures
if useOCD
    PF = cell(OCD_nPreGen,1);
    PF{1} = population(paretofront(population(:,nVar+1:nVar+nObj)),...
        nVar+1:nVar+nObj);
end;
terminationCriterion = false;

% initialize archive
if outputType == 2
    archive = nan(maxEval, nVar+nObj);
    archive(1:countEval,:) = population(1:countEval,:);
end;

% evolutionary loop
while ~terminationCriterion && (countEval < maxEval)
    
    % evaluate parameters
    variable_crossover_prob = myeval(opts.var_crossover_prob);
    variable_crossover_dist = myeval(opts.var_crossover_dist);
    variable_mutation_prob = myeval(opts.var_mutation_prob);
    variable_mutation_dist =myeval(opts.var_mutation_dist);
    variable_swap_prob = myeval(opts.var_swap_prob);
    DE_F = myeval(opts.DE_F);
    DE_CR = myeval(opts.DE_CR);
    DE_CombinedCR = myeval(opts.DE_CombinedCR);
    useDE = myeval(opts.useDE);
    refPoint = myeval(opts.refPoint);
    
    % generate and add offspring
    population(elementInd,:) = generate_offspring(population, ...
        nObj, nVar, rngMin, rngMax, problem, ranks, ...
        variable_crossover_prob, variable_crossover_dist, ...
        variable_mutation_prob, variable_mutation_dist,...
        variable_swap_prob, useDE, DE_CombinedCR, DE_F, DE_CR);
    countEval = countEval+1;
    
    % update archive
    if outputType == 2
        archive(countEval,:) = population(elementInd,:);
    end;
    
    % environmental selection
    ranks = paretoRank(population(:,nVar+1:nVar+nObj));
    nPV = max(ranks);
    elementInd = select_element_to_remove(population, nObj, nVar, nPV,...
        ranks, nPFevalHV, refPoint);
    
    % perform OCD
    if useOCD && mod(countEval, nPop)==0
        iteration = int16(round(countEval./nPop));
        if iteration > OCD_nPreGen+1
            % shift reference fronts
            for i = 2:OCD_nPreGen+1
                PF{i-1} = PF{i};
            end;
            active = (1:nPop+1)'~=elementInd & ranks==1;
            PF{OCD_nPreGen+1} = population(active,nVar+1:nVar+nObj);
            [terminationCriterion, p] = OCD(PF, OCD_VarLimit, 0.05,...
                refPoint, p);
        else
            active = (1:nPop+1)'~=elementInd & ranks==1;
            PF{iteration} = population(active,nVar+1:nVar+nObj);
            if iteration == OCD_nPreGen+1
                [terminationCriterion, p] = OCD(PF, OCD_VarLimit, 0.05,...
                    refPoint);
            end;
        end;
        if terminationCriterion
            disp('OCD detected convergence due to the variance test');
        end;
    end;
    if mod(countEval, outputGen)==0 && outputType > 0
        if outputType == 1
            if useOCD && exist('p', 'var')
                writeToFile(population, nPop, elementInd, nVar, nObj,...
                    ranks, countEval, outdir, p);
            else
                writeToFile(population, nPop, elementInd, nVar, nObj,...
                    ranks, countEval, outdir);
            end;
        elseif outputType == 2
            if useOCD && exist('p', 'var')
                writeArchiveToFile(archive, nVar, nObj, countEval,...
                    outdir, p);
            else
                writeArchiveToFile(archive, nVar, nObj, countEval, outdir);
            end;
        end;
    end;
end;
if outputType > 0 &&  mod(countEval, outputGen)~=0
    if outputType == 1
        if useOCD
            writeToFile(population, nPop, elementInd, nVar, nObj, ranks,...
                countEval, outdir);
        else
            writeToFile(population, nPop, elementInd, nVar, nObj, ranks,...
                countEval, outdir, p);
        end;
    elseif outputType == 2
        if useOCD
            writeArchiveToFile(archive, nVar, nObj, countEval,...
                outdir, p);
        else
            writeArchiveToFile(archive, nVar, nObj, countEval, outdir);
        end;
    end;
end;
population(elementInd,:) = [];
paretoFront = population(:,nVar+1:nVar+nObj);
paretoSet = population(:,1:nVar);
end

%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------
function opts=getoptions(inopts, defopts)
% OPTS = GETOPTIONS(INOPTS, DEFOPTS) handles an arbitrary number of
% optional arguments to a function. The given arguments are collected
% in the struct INOPTS.  GETOPTIONS matches INOPTS with a default
% options struct DEFOPTS and returns the merge OPTS.  Empty or missing
% fields in INOPTS invoke the default value.  Fieldnames in INOPTS can
% be abbreviated.
if nargin < 2 || isempty(defopts) % no default options available
    opts=inopts;
    return;
elseif isempty(inopts) % empty inopts invoke default options
    opts = defopts;
    return;
elseif ~isstruct(defopts) % handle a single option value
    if isempty(inopts)
        opts = defopts;
    elseif ~isstruct(inopts)
        opts = inopts;
    else
        error('Input options are a struct, while default options are not');
    end
    return;
elseif ~isstruct(inopts) % no valid input options
    error('The options need to be a struct or empty');
end

opts = defopts; % start from defopts
% if necessary overwrite opts fields by inopts values
defnames = fieldnames(defopts);
idxmatched = []; % indices of defopts that already matched
for name = fieldnames(inopts)'
    name = name{1}; % name of i-th inopts-field
    idx = strncmpi(defnames, name, length(name));
    if sum(idx) > 1
        error(['option "' name '" is not an unambigous abbreviation. ' ...
            'Use opts=RMFIELD(opts, ''' name, ...
            ''') to remove the field from the struct.']);
    end
    if sum(idx) == 1
        defname  = defnames{find(idx)};
        if ismember(find(idx), idxmatched)
            error(['input options match more than ones with "' ...
                defname '". ' ...
                'Use opts=RMFIELD(opts, ''' name, ...
                ''') to remove the field from the struct.']);
        end
        idxmatched = [idxmatched find(idx)];
        val = getfield(inopts, name);
        % next line can replace previous line from MATLAB version 6.5.0 on and in octave
        % val = inopts.(name);
        if isstruct(val) % valid syntax only from version 6.5.0
            opts = setfield(opts, defname, ...
                getoptions(val, getfield(defopts, defname)));
        elseif isstruct(getfield(defopts, defname))
            % next three lines can replace previous three lines from MATLAB
            % version 6.5.0 on
            %   opts.(defname) = ...
            %      getoptions(val, defopts.(defname));
            % elseif isstruct(defopts.(defname))
            warning(['option "' name '" disregarded (must be struct)']);
        elseif ~isempty(val) % empty value: do nothing, i.e. stick to default
            opts = setfield(opts, defnames{find(idx)}, val);
            % next line can replace previous line from MATLAB version 6.5.0 on
            % opts.(defname) = inopts.(name);
        end
    else
        warning(['option "' name '" disregarded (unknown field name)']);
    end
end
end
%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------
function res=myeval(s)
if ischar(s)
    res = evalin('caller', s);
else
    res = s;
end
end
%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------
function f = initialize_variables(nPop, nObj, nVar,...
    min_range, max_range, problem, initPop)
% function f = initialize_variables(nPop, nObj, nVar, min_range, max_range,
% problem)
% This function initializes the chromosomes. Each chromosome has the
% following at this stage
%       * set of decision variables
%       * objective function values
% where,
% nPop - Population size
% nObj - Number of objective functions
% nVar - Number of decision variables
% min_range - A vector of decimal values which indicates the minimum value
%             for each decision variable.
% max_range - Vector of maximum possible values for decision variables.
% initPop - file with initial population, if empty a random initialization is performed

f = inf(nPop+1,nVar+nObj); %preallocation
if exist(initPop, 'file')
    % load variables from file
    temp = load(initPop);
    if size(temp,1) ~= nPop
        error('data in file initPop has to be of size nPop');
    else
        f(1:nPop,1:nVar) = temp;
    end;
    if all(min(f(1:nPop,1:nVar))>=0) && all(max(f(1:nPop,1:nVar))<=1)
        % normalized designs have to be transformed
        f(1:nPop,1:nVar) = repmat(min_range,nPop,1) + ...
            repmat((max_range - min_range),nPop,1).*f(1:nPop,1:nVar);
    end;
    f(nPop+1,1:nVar) = min_range + (max_range - min_range).*rand(1, nVar);
else
    % Initialize the decision variables based on the minimum and maximum
    % possible values. nVar is the number of decision variable. A random
    % number is picked between the minimum and maximum possible values for
    % the each decision variable.
    f(:,1:nVar) = repmat(min_range,nPop+1,1) + ...
        repmat((max_range - min_range),nPop+1,1).*rand(nPop+1, nVar);
end;

% Evaluate each chromosome:
for i = 1 : nPop
    % For ease of computation and handling data the chromosome also has the
    % value of the objective function concatenated at the end.
    f(i,nVar+1:nVar+nObj) = feval(problem, f(i,1:nVar));
end;
end
%%-------------------------------------------------------------------------
%%-------------------------------------------------------------------------
function elementInd = select_element_to_remove(population, nObj, nVar,...
    nPV, ranks, nPFevalHV, refPoint)
elementsInd = find(ranks==nPV);
frontsize = size(elementsInd,1);
if frontsize==1
    elementInd = 1;
elseif nPV > nPFevalHV
    % current front is higher than the threshold, select index randomly
    elementInd = int16(max(ceil(rand(1)*frontsize),1));
else
    frontObjectives = population(elementsInd,nVar+1:nVar+nObj);
    if refPoint==0
        % adaptive reference point
        refPoint = max(frontObjectives)+1;
    else
        % filter solutions not dominating the predefined reference point
        index = false(frontsize,1);
        for i = 1:frontsize
            if any(frontObjectives(i,:) >= refPoint)
                index(i) = true;
            end;
        end;
        if sum(index) > 0
            % enough infeasible solutions, remove the one with the
            % strongest individual violation
            [maxVal, IX] = max(max(frontObjectives-...
                repmat(refPoint,frontsize,1), [], 2));
            elementInd = elementsInd(IX(1));
            return;
        end;
    end
    deltaHV = zeros(1,frontsize);
    if nObj == 2
        % use fast calculation of HV contributions
        [frontObjectives, IX] = sortrows(frontObjectives, 1);
        deltaHV(IX(1)) = ...
            (frontObjectives(2,1) - frontObjectives(1,1)) .* ...
            (refPoint(2) - frontObjectives(1,2));
        for i = 2:frontsize-1
            deltaHV(IX(i)) = ...
                (frontObjectives(i+1,1) - frontObjectives(i,1))...
                .* ...
                (frontObjectives(i-1,2) - frontObjectives(i,2));
        end;
        deltaHV(IX(frontsize)) = ...
            (refPoint(1) - frontObjectives(frontsize,1)) .* ...
            (frontObjectives(frontsize-1,2) - ...
            frontObjectives(frontsize,2));
    else
        % resort to general HV code for arbitrary dimension
        currentHV = hv(frontObjectives', refPoint);
        for i=1:frontsize
            myObjectives = frontObjectives;
            myObjectives(i,:)=[];
            myHV = hv(myObjectives', refPoint);
            deltaHV(i) = currentHV - myHV;
        end
    end
    [minVal, IX]=min(deltaHV);
    elementInd = IX(1);
end;
elementInd = elementsInd(elementInd);
end
%-------------------------------------------------------------------------
%-------------------------------------------------------------------------
function offspring = generate_offspring(population, ...
    nObj, nVar, rngMin, rngMax, problem, ranks, ...
    variable_crossover_prob, variable_crossover_dist, ...
    variable_mutation_prob, variable_mutation_dist, variable_swap_prob, ...
    useDE, DE_CombinedCR, DE_F, DE_CR)
% function child  = generate_offspring(population, ...
%    nObj, nVar, rngMin, rngMax, problem, ranks, ...
%    variable_crossover_prob, variable_crossover_dist, ...
%    variable_mutation_prob, variable_mutation_dist, variable_swap_prob, ...
%    useDE, DE_CombinedCR, DE_F, DE_CR)
%
% population - all possible parents
% nObj - number of objective functions
% nVar - number of decision varaiables
% rngMin - a vector of lower limit for the corresponding decsion variables
% rngMax - a vector of upper limit for the corresponding decsion variables
% problem - problem string
% ranks - ranks of the population
% variable_crossover_prob - probability for crossover
% variable_crossover_dist - distribution index for crossover
% variable_mutation_prob - probability for mutation
% variable_mutation_dist - distribution index for mutation
% variable_swap_prob - probability for swapping variables after crossover
% useDE - use differential evolution instead of SBX & PM (true/false)
% DE_CombinedCR - Crossover in blocks or bits
% DE_F - difference weight for differential evolution
% DE_CR - crossover probability for DE
%
% The genetic operation is performed only on the decision variables, that
% are the first V elements in the chromosome vector.

nPop = size(population,1);
% Pre-Allocation
parent = zeros(4,nVar);
offspring = zeros(1,nVar+nObj);
if useDE
    % use differential evolution
    % we need four different parents
    mypermutation = randperm(nPop);
    parent(1,:) = population(mypermutation(1),1:nVar);
    parent(2,:) = population(mypermutation(2),1:nVar);
    switch nPop
        case 2
            parent(3,:) = population(mypermutation(1),1:nVar);
            parent(4,:) = population(mypermutation(2),1:nVar);
        case 3
            parent(3,:) = population(mypermutation(3),1:nVar);
            parent(4,:) = population(mypermutation(1),1:nVar);
        otherwise
            parent(3,:) = population(mypermutation(3),1:nVar);
            parent(4,:) = population(mypermutation(4),1:nVar);
    end;
    % build help_child
    child_1 = parent(2,:) + DE_F.*(parent(3,:)-parent(4,:));
    %combine child_1 & parent_1
    l_index = ceil(nVar*rand(1));
    if l_index == 0
        l_index = 1;
    end
    if DE_CombinedCR
        l_index_add = 0;
        while (rand(1) < DE_CR) && (l_index_add < nVar-1)
            l_index_add = l_index_add + 1;
        end;
        if l_index+l_index_add > nVar
            r_index = l_index+l_index_add-nVar;
            for j=1:nVar
                if (j<=r_index) || (j>=l_index)
                    offspring(j)=child_1(j);
                else
                    offspring(j)=parent(1,j);
                end
            end
        else
            r_index = l_index+l_index_add;
            for j=1:nVar
                if (j>=l_index)&&(j<=r_index)
                    offspring(j)=child_1(j);
                else
                    offspring(j)=parent(1,j);
                end
            end
        end
    else
        for j=1:nVar
            if (j == l_index) || (rand(1) < DE_CR)
                offspring(j)=child_1(j);
            else
                offspring(j)=parent(1,j);
            end
        end;
    end;
else
    % use SBX & PM
    % Initialize the parents for SBX
    % two binary tournaments
    randomindices = ceil(rand(1,4)*nPop);
    randomindices(randomindices==0)=1;
    parent(1,:) = population(randomindices(1),1:nVar);
    parent(2,:) = population(randomindices(2),1:nVar);
    parent(3,:) = population(randomindices(3),1:nVar);
    parent(4,:) = population(randomindices(4),1:nVar);
    if ranks(randomindices(1)) < ranks(randomindices(2))
        parent_1 = parent(1,:);
    elseif ranks(randomindices(1)) > ranks(randomindices(2))
        parent_1 = parent(2,:);
    elseif rand(1) > 0.5
        parent_1 = parent(1,:);
    else
        parent_1 = parent(2,:);
    end
    if ranks(randomindices(3)) < ranks(randomindices(4))
        parent_2 = parent(3,:);
    elseif ranks(randomindices(3)) > ranks(randomindices(4))
        parent_2 = parent(4,:);
    elseif rand(1) > 0.5
        parent_2 = parent(3,:);
    else
        parent_2 = parent(4,:);
    end
    % Perform crossover for each decision variable.
    child_1 = zeros(1,nVar);
    child_2 = zeros(1,nVar);
    for j = 1 : nVar
        if rand(1) < variable_crossover_prob
            % SBX (Simulated Binary Crossover)
            u = rand(1);
            if u <= 0.5
                bq = (2*u)^(1/(variable_crossover_dist+1));
            else
                bq = (1/(2*(1 - u)))^(1/(variable_crossover_dist+1));
            end
            % Generate the jth element of first child
            child_1(j) = ...
                0.5*(((1 + bq)*parent_1(j)) + (1 - bq)*parent_2(j));
            % Generate the jth element of second child
            child_2(j) = ...
                0.5*(((1 - bq)*parent_1(j)) + (1 + bq)*parent_2(j));
        else
            child_1(j) = parent_1(j);
            child_2(j) = parent_2(j);
        end
        if rand(1) < variable_swap_prob
            swap = child_1(j);
            child_1(j) = child_2(j);
            child_2(j) = swap;
        end;
    end
    if rand(1) < 0.5
        offspring(1:nVar) = child_1;
    else
        offspring(1:nVar) = child_2;
    end;
    % perform mutation. Mutation is based on polynomial mutation.
    % Perform mutation on each element of the selected parent.
    deltaMax = rngMax - rngMin;
    for j = 1 : nVar
        if rand(1) < variable_mutation_prob
            r = rand(1);
            if r < 0.5
                delta = (2*r)^(1/(variable_mutation_dist+1)) - 1;
            else
                delta = 1 - (2*(1 - r))^(1/(variable_mutation_dist+1));
            end
            % Generate the corresponding child element.
            offspring(j) = offspring(j) + delta.*deltaMax(j);
        end
    end
end
% Make sure that the generated element is within the decision space.
offspring(1:nVar) = min([rngMax; offspring(1:nVar)]);
offspring(1:nVar) = max([rngMin; offspring(1:nVar)]);
% Evaluate the objective functions
offspring(nVar + 1: nVar + nObj) = feval(problem, offspring(1:nVar));
end

function [stopFlag, pNew] = OCD(PF, varLimit, alpha, ref, p)
% Determination of convergence by means of statistical tests on the
% variance of the internally optimized HV indicator
%
% Call: [stopFlag, pNew] = OCD(PF, varLimit, alpha, ref, p)
%
% Input arguments:
% PF        is a 1xnPreGen+1 vector of cell arrays holding the current and
%           the last nPreGen Pareto front approximations
% varLimit  is the minimum variance limit (default: 1e-3)
% alpha     is the significance level of the statistical tests
%           (default: 0.05)
% ref       is a 1xd vector with the reference point in the d-dimensional
%           objective space (default: -inf(1,d))
% pNew         is the p-value of the Chi^2 variance test in the last iteration
%           (default: 1)
%
% Output arguments:
% stopFlag  is a boolean indicating whether the test detect convergence
% p         is the p-value of the variance test in the current iteration
%
% A detailed description of the procedure and the variables used in the
% code can be found in:
% Wagner, T.; Trautmann, H.: Online Convergence Detection for Evolutionary
% Multi-Objective Algorithms Revisited. In: Proceedings of the 2010 IEEE
% Congress on Evolutionary Computation (IEEE CEC 2010), July 18-23, 2010,
% Barcelona, Spain, G. Fogel, H. Ishibuchi (eds.), pp. 3554-3561
%
% Author: Tobias Wagner, Institute of Machining Technology, TU Dortmund
% License: GPLv2
% Last Revision: 2016-02-03
if nargin < 1
    error('OCD requires Pareto front approximations to detect convergence');
end;

% check input and initialize variables
nPreGen = length(PF)-1;
PI = zeros(1,nPreGen);
PFi = PF{nPreGen+1};
d = size(PFi,2);
if nargin < 5 || isempty(p)
    p = 1;
end
if nargin < 4 || isempty(ref) || ref == 0
    % determine ub from the data
    ref = -inf(1,d);
    for i = 1:nPreGen+1
        ref = max([ref; PF{i}]);
    end;
    ref = ref+1;
end;
if nargin < 3 || isempty(alpha)
    alpha = 0.05;
end;
if nargin < 2 || isempty(varLimit)
    varLimit = 1e-3;
end;

% compute hypervolume of the reference set
refValue = hv(PFi', ref);

for k = 1:nPreGen
    % compute indicator values
    PI(k) = refValue-hv(PF{k}', ref);
end;
pNew = Chi2(PI, varLimit); % perform Chi^2 test
% evaluate test-based termination criteria
stopFlag = (pNew <= alpha) && (p <= alpha);
end

function p = Chi2(PI, VarLimit) % One-sided Chi^2 variance test
N = size(PI,2)-1; % determine degrees of freedom
Chi = (var(PI).*N)./VarLimit; % compute test statistic
% look up p-value from Chi^2 distribution with N degrees of freedom
p = chi2cdf(Chi, N);
end

function writeToFile(population, nPop, elementInd, nVar, nObj, ranks,...
    countEval, outdir, p)
active = setdiff(1:nPop+1,elementInd);
PS = population(active,1:nVar);
PF = population(active,nVar+1:nVar+nObj);
dlmwrite(sprintf('%spar_%03d.txt', outdir, countEval), PS, ' ');
dlmwrite(sprintf('%sobj_%03d.txt', outdir, countEval), PF, ' ');
dlmwrite(sprintf('%sps_%03d.txt', outdir, countEval),...
    PS(ranks(active)==1,:), ' ');
dlmwrite(sprintf('%spf_%03d.txt', outdir, countEval),...
    PF(ranks(active)==1,:), ' ');
if nargin > 8
    dlmwrite(sprintf('%spvalue_%03d.txt', outdir, countEval), p, ' ');
end;
end

function writeArchiveToFile(archive, nVar, nObj, countEval, outdir, p)
X = archive(1:countEval,1:nVar);
Y = archive(1:countEval,nVar+1:nVar+nObj);
PF = Y(paretofront(Y),:);
dlmwrite(sprintf('%spar_%03d.txt', outdir, countEval), X, ' ');
dlmwrite(sprintf('%sobj_%03d.txt', outdir, countEval), Y, ' ');
dlmwrite(sprintf('%spf_%03d.txt', outdir, countEval), PF, ' ');
if nargin > 5
    dlmwrite(sprintf('%spvalue_%03d.txt', outdir, countEval), p, ' ');
end;
end