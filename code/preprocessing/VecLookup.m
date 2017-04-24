function vec = VecLookup(word, vocabhash)
inpfile = fopen('glove.840B.300d.txt');

if (isKey(vocabhash, word))
    linenum = vocabhash(word);
    C = textscan(inpfile,['%s',repmat('%f',[1,300])],1,'CollectOutput',1,'HeaderLines',linenum-1);
    vec = C{2};
    vec = vec';
else
    vec = NaN;
end
fclose(inpfile);
end