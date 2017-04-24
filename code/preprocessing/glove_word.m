% Create a hash table to easily look up the vector form of a word
vocabhash = containers.Map;
fid = fopen('glove.840B.300d.txt');
inFile = textscan(fid, '%s','EndOfLine','\n','Delimiter','\n');
fclose(fid);
vocab = inFile{:};
clear inFile
for i = 1:length(vocab)
    vocabhash(strtok(vocab{i})) = i;
end
clear vocab
% Use WordVec to get the vector of each word in a TR
textfile = 'sherlock_text_TRs.txt';
numTR = 1976;
wordvecs = WordVec(textfile, numTR, vocabhash);
save('wordvecs_sherlock.mat','wordvecs')

% Use UnweightedAvg to get the unweighted average of the word vectors at each TR
uavgvecs = UnweightedAvg(wordvecs);

% save results
save('uavgvecs_sherlock.mat','uavgvecs')
