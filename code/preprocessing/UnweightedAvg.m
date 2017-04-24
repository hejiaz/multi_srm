% Perform unweighted averaging of vectors from wordvecs generated from the
% method WordVec()

function uavgvecs = UnweightedAvg(wordvecs)
    length = size(wordvecs,1);
    uavgvecs = zeros(300, length);
    for i = 1:length
        vecs = wordvecs{i};
        if (~isempty(vecs) == 1)
            avg = mean(vecs,2);
        else
            avg = zeros(1,300);
        end
        uavgvecs(:,i) = avg;   
    end
end

