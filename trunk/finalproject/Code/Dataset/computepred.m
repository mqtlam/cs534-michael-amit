function [ ] = computepred( )
%COMPUTEPRED Summary of this function goes here
%   Detailed explanation goes here

correct = importdata('groundTruth');
facc = fopen('Results/accuracy.txt','at');

xbound = 0.7;
while (xbound < 0.71)
    ybound = 0.1;
    while(ybound < 0.11)
        incorrect=0;
        k=1;
        fname = sprintf('Results/Predictions_x_%.2f_y_%.2fd', xbound, ybound);
        fprintf('\n %s', fname);
        fid = fopen(fname, 'at');
        for i=127:200
            name = 'Imgvideo0';
            j = int2str(i);
            name = strcat(name, j);
            name = strcat(name, '.mat');
            startplot(name);
            %fprintf('\n %s', name);
            pred = makeprediction(name, xbound, ybound);
            if(pred ~= 0)
                name = '0';
                j = int2str(i);
                name = strcat(name, j);
                
                while(correct(k,1) < i)
                    k=k+1;
                end
                while(correct(k,1) > i)
                    k=k-1;
                end
                
                
                if(correct(k,2) == 0)
                    k=k+1;
                    continue;
                end
                
                if(correct(k,2) ~= pred)
                    incorrect = incorrect+1;
                    fprintf(fid, '%s\t%d <= Incorrect\n',name, pred);
                else
                    fprintf(fid, '%s\t%d\n',name, pred);
                end
                k=k+1;
            end 
        
        end
        fprintf(facc, '%s\t%.2f\n',fname, (incorrect/k));
        ybound = ybound + 0.05;
    end
    xbound = xbound + 0.05;
end
end