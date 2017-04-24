% Creates word vectors at each TR
% Depends on VecLookUp.m, and vocabhash created by HashVocab.m
% Input: textfile = transcript file (already separated by lines according
% to TR), numTR = number of TRs, vocabhash = % hash table of 300 dimensional
% word vectors created from HashVocab
function wordvecs = WordVec(textfile, numTR, vocabhash)
    vechash = containers.Map; % save already read words
    fid = fopen(textfile); % open the file
    inFile = textscan(fid, '%s','EndOfLine','\n','Delimiter','\n');
    fclose(fid);    
    tlines = inFile{:}; % read lines
    wordvecs = cell(numTR, 1); % store word vectors
    
    % read line in text one by one, create word vectors
    for i = 1:numTR
        tline = tlines{i};
        words = strsplit(tline); % split words in each line by white space
        wordnum = size(words,2); % get number of words in line
        TRvecs = zeros(300,wordnum); % 300 hundred dimensional word vectors for this specific TR
        todelete = zeros(wordnum, 1); % delete in case VecLookup returns NaN
        % look up vector representation of each word, put all together in
        % matrix, delete NaN columns
        for j = 1:wordnum
            if (isKey(vechash, words{j}))
                vec = vechash(words{j});
            else
                vec = VecLookup(words{j}, vocabhash);
                vechash(words{j}) = vec;
            end
            if isnan(vec)
                todelete(j) = 1;
            else
                TRvecs(:,j) = vec;
            end
        end
        % delete the NaN columns
        todelete_index = find(todelete);
        TRvecs(:,todelete_index) = [];
        
        % put matrix into wordvecs cell
        wordvecs{i} = TRvecs;
                
    end
end