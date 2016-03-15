function Model = LPCA( X, NClu, DLat, MaxIter )

% Model = LPCA( X, NClu, DLat ) partitions the set X into NClu clusters by
% Local PCA method
%
%   Please refer Nandakishore Kambhatla and Todd K. Leen. 'Dimension
%   Reduction by Local Principal Component Analysis' for more details.
%
%     Parameters:
%     X         - data set
%     NClu      - number of clusters
%     DLat      - dimension of latent sapce
%     MaxIter   - maximum iterations
%     Returns:
%     Model
%       - Model.Mean    mean vector of each cluster
%       - Model.PI      matrix PI to each cluster
%       - Model.Eve     eigenvectors 
%       - Model.Eva     eigenvaluse
%       - Model.PMin    min values in principal vectors
%       - Model.PMax    max values in principal vectors
%       - Model.Index   index to which cluster
%
%     Aimin Zhou
%     Department of Computer Science
%     University of Essex
%     Colchester, U.K, CO4 3SQ
%     amzhou@gmail.com
%
% History:
%     6/10/2006 create
%    27/10/2006 remove the part to optimize reference vector
%               the reference vector is set to be the mean of each cluster
%    10/11/2006 optimize the codes


%% Step 1: define and set algorithm parameters
DX          = size(X,1);        % dimension of decision variables
NX          = size(X,2);        % size of data
index       = randperm(NX);
Model.Mean  = X(:,index(1:NClu));           % reference vector to each cluster
Model.PI    = repmat(eye(DX) , [1 1 NClu]); % matrix PI to each cluster
Model.Eve   = repmat(eye(DX) , [1 1 NClu]); % eigenvectors
Model.Eva   = zeros(DX,NClu);               % eigenvalues
Model.PMin  = zeros(DLat,NClu);             % min projection in each principal directions
Model.PMax  = zeros(DLat,NClu);             % max projection in each principal directions

%% Step 2: main iterations
iter        = 1;
update      = NClu;
repartion   = 1;
while((update>0 && iter <= MaxIter) ||repartion>0)
    % Step 2.1 partition the population
    dis     = zeros(NX,NClu);
    for c=1:1:NClu
        dis(:,c) = sum((X-repmat(Model.Mean(:,c), [1,NX]))'*Model.PI(:,:,c).*(X-repmat(Model.Mean(:,c), [1,NX]))',2);
    end
    [mins, Model.Index] = min(dis,[],2);
    id                  = eye(NClu);
    cindex              = id(Model.Index,:);
    
    % Step 2.2 update the reference vectors
    update      = NClu;
    repartion   = 0;
    for c=1:1:NClu
        mean_old    = Model.Mean(:,c);
        if sum(cindex(:,c))<1
            index           = randperm(NX);
            Model.Mean(:,c) = X(:,index(1));
            Model.PI(:,:,c) = eye(DX);
            repartion       = 1;
        elseif sum(cindex(:,c))==1
            Model.Mean(:,c) = X(:,cindex(:,c)'>0);
            Model.PI(:,:,c) = eye(DX);
        elseif sum(cindex(:,c))>1
            cx                  = X(:,cindex(:,c)'>0);      % data in this cluster
            Model.Mean(:,c)     = mean(cx,2);               % mean 
            cy                  = cx - repmat(Model.Mean(:,c),[1 size(cx,2)]);
            % eigens and sort eigens
            [eve,eva]   = eig(cov(cy'));
            eva         = diag(eva);
            [eva, perm] = sort(-eva);
            eva         = -eva;
            eve         = eve(:,perm);
            % set values
            Model.Eve(:,:,c)    = eve;
            Model.Eva(:,c)      = eva;
            Model.PI(:,:,c)     = eve(:,DLat+1:DX) * eve(:,DLat+1:DX)';
        end
        % check whether new center point is found
        err     = sqrt((mean_old-Model.Mean(:,c))'*(mean_old-Model.Mean(:,c)));
        if err<1.0e-5 
            update = update - 1; 
        end
    end
    iter = iter + 1;
end

%% Step 3: make all parameters feasible
Model.Eve = real(Model.Eve);
Model.Eva = real(Model.Eva);    Model.Eva(Model.Eva<realmin) = realmin;
Model.Mean= real(Model.Mean);
Model.PI  = real(Model.PI);

%% Step 4: calculate the projection
for c=1:1:NClu
    if sum(cindex(:,c))>1
        cx                  = X(:,cindex(:,c)'>0);      % data in this cluster
        cy                  = cx - repmat(Model.Mean(:,c),[1 size(cx,2)]);
        proj                = cy'*eve(:,1:DLat);
        Model.PMin(:,c)     = min(proj,[],1)';
        Model.PMax(:,c)     = max(proj,[],1)';
    end
end

clear eve eva X cindex;