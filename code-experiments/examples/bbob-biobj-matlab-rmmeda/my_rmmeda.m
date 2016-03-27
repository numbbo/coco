function my_rmmeda(f, lower_bounds, upper_bounds, budget)
% set parameters
Problem.Name    = 'COCO';     % name of test problem
Problem.NObj    = 2;            % number of objectives
Problem.XLow    = lower_bounds';  % lower boundary of decision variables, it also defines the number of decision variables
Problem.XUpp    = upper_bounds';   % upper boundary of decision variables
Problem.FObj    = @(x)wrapperRMMEDA(f, x); % Objective function, please read the definition

Generator.Name  = 'LPCA';       % name of generator
Generator.NClu  = 5;            % parameter of generator, the number of clusters(default)
Generator.Iter  = 50;           % maximum trainning steps in LPCA
Generator.Exte  = 0.25;         % parameter of generator, extension rate(default)

NIni            = min(100,budget);          % population size
IterMax         = floor((budget-NIni)./NIni); % number of generations

% run
RMMEDA( Problem, Generator, NIni, IterMax );