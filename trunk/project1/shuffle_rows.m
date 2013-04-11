function [ shuffledArray ] = shuffle_rows( orderedArray )
%SHUFFLE_ROWS Shuffles the rows of a matrix

shuffledArray = orderedArray(randperm(size(orderedArray,1)),:);

end

