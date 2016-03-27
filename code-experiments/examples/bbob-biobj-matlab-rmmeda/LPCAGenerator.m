function TriX = LPCAGenerator(PopX, XLow, XUpp, NIni, DX, DLat, NClu, MaxIter, Exten)

%%
% TriX = LPCAGenerator(PopX, XLow, XUpp, NIni, DX, DLat, NClu,
% MaxIter, Exten) generate a set of new trial solutions from previous
% population
%
%     Regularity Model based Multiobjective Estimation of Distribution
%     Algorithm (RM-MEDA), see Q. Zhang, A. Zhou, Y. Jin. 'Modelling the
%     Regularity in an Estimation of Distribution Algorithm for Continuous 
%     Multiobjective Optimisation with Variable Linkages'
%     (http://cswww.essex.ac.uk/staff/qzhang/mypublication.htm) for more
%     details.
%
%     Parameters:
%     PopX - previous population
%     XLow - lower boundary of search space
%     XUpp - upper boundary of search space
%     NIni - number of initial population defined by experimental design
%     DX   - dimension of search space
%     DLat - dimension of latent sapce
%     NClu - number of clusters
%     MaxIter - maximum iterations of Local PCA
%     Exten- extension rate of sampling
%     Returns:
%     TriX - new trial solutions
%
%	Copyright (c) Aimin Zhou (2006)
%     Department of Computer Science
%     University of Essex
%     Colchester, U.K, CO4 3SQ
%     amzhou@gmail.com
%
% History:
%     30/10/2006 create

%% Step M.1: build probability model by Local PCA 
Model  = LPCA(PopX, NClu, DLat, MaxIter);

%% Step M.2: find the number of groups with more than one points
exist   = zeros(NClu,1);    % indicator which cluster exist (more than one point)
for i=1:1:NClu
    exist(i) = sum(Model.Index == i)>1;
end

%% Step M.3: calculate probability of each cluster
prob    = prod(Model.PMax - Model.PMin, 1)';    % probability vector
prob    = prob.*exist;
tvol    = sum(prob(exist>0,:));                 % total volume
prob    = prob/tvol;
cn      = 2*ones(NClu,1)+floor((NIni-2*NClu)*prob);% each cluster creates at least 2 points
while sum(cn)<NIni
    num     = min(cn(cn > 2,:));
    id      = find(cn==num,1);
    cn(id)  = cn(id) + 1;
end

%% Step M.4: creat new trial solutions in each cluster
tn  = 0;
ast = 0;
for k=1:1:NClu
    if exist(k)
        LInd    = ones(DLat,cn(k));
        for i=1:1:DLat
            LInd(i,:)   = randperm(cn(k));
        end
        cmax    = Model.PMax(:,k) + Exten*(Model.PMax(:,k)-Model.PMin(:,k));
        cmin    = Model.PMin(:,k) - Exten*(Model.PMax(:,k)-Model.PMin(:,k));
        T       = (LInd - rand(DLat,cn(k)))/cn(k) .* ((cmax-cmin)*ones(1,cn(k))) + cmin*ones(1,cn(k));
        st      = sqrt(sum(abs(Model.Eva(DLat+1:DX,k)))/(DX-DLat));
        ast     = st;
        eve     = Model.Eve(:,:,k);
        TriX(:,tn+1:tn+cn(k)) = repmat(Model.Mean(:,k),[1 cn(k)]) + eve(:,1:DLat)*T + normrnd(0.0,st,[DX,cn(k)]);
        
%         Model.PMax(:,k)
%         Model.PMin(:,k)
%         sqrt(Model.Eva(:,k))
    else
        TriX(:,tn+1:tn+cn(k)) = repmat(Model.Mean(:,k),[1 cn(k)]) + normrnd(0.0,ast,[DX,cn(k)]);
    end 
    tn = tn + cn(k);
end

%% Step M.5 check the boundaries
low     = repmat(XLow,[1 NIni]);
upp     = repmat(XUpp,[1 NIni]);
lbnd    = TriX < low;
ubnd    = TriX > upp;
TriX(lbnd) = 0.5*(PopX(lbnd) + low(lbnd)); 
TriX(ubnd) = 0.5*(PopX(ubnd) + upp(ubnd));

clear LInd;