function ranks = paretoRank(objectives)
%paretoRank calculates pareto-ranks for all elements of objectives
nPop = size(objectives,1);
ranks = zeros(nPop,1);
% init select-vector filled with logical ones
popInd = true(nPop,1);
nPV = 1;
while any(popInd)
    %get next paretofront
    frontInd = popInd;
    frontInd(popInd) = paretofront(objectives(popInd,:));
    ranks(frontInd) = nPV;
    % remove ones of next paretofront from select-vector
    popInd = xor(popInd,frontInd);
    nPV = nPV+1;
end;