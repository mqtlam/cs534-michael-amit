function [ vocabMap ] = vocabTransform( vocabulary, wordLengthThreshold )
%VOCABTRANSFORM Summary of this function goes here
%   vocabulary:             cells of vocabulary words
%   wordLengthThreshold:    only keep words that have length greater than
%                           this threshold
%   vocabMap:               mapping from original vocabulary to 
%                           new vocabulary:
%                               index -> new index
%                               index -> 0 if word omitted

%% default parameters
if nargin < 2
    wordLengthThreshold = 3;
end

%% initialization
originalVocabSize = length(vocabulary);
vocabMap = zeros(originalVocabSize, 1);

%% construct new vocabulary by creating a mapping from old to new
newId = 1;
for wordId = 1:originalVocabSize
    if (length(vocabulary{wordId}) > wordLengthThreshold)
        vocabMap(wordId) = newId;
        newId = newId + 1;
    end
end

end

