function [ pData, names ] = computepred( )
%COMPUTEPRED Summary of this function goes here
%   Detailed explanation goes here

correct1 = importdata('wideGT_game2');
d = correct1.data;
correct = zeros(175,2);

for i=1:size(d)
    correct(i,2) = d(i);
end

ans1 = correct1.textdata;
ans1 = (ans1(:,1));
for i = 1:size(d)
    temp = ans1{i};
    correct(i,1) = str2num(temp);
end

names = [];
l=1;
predData = zeros(175,61)
%facc = fopen('Results/accuracy.txt','at');
%count =0;
xbound = 0.6;
while (xbound < 0.86)
    ybound = 0.0;
    while(ybound < 0.21)
        %incorrect=0;
        k=1;
        
        %fname = sprintf('Results/Predictions_x_%.2f_y_%.2fd', xbound, ybound);
        fprintf('\n\n %d %d', xbound, ybound);
        %fid = fopen(fname, 'at');
        for i=1:175
            name = 'video0';
            if(i<10)
                name = strcat(name, '00');
            elseif i<100
                name = strcat(name, '0');
            end
            j = int2str(i);
            name = strcat(name, j);
            %fprintf('\n %s', name);
            [pred1 pred2]= makeprediction(name, xbound, ybound);
            if(pred1 ~= 0 && pred2~=0)
%                 name = '0';
%                 if(i<10)
%                     name = strcat(name, '00');
%                 elseif i<100
%                     name = strcat(name, '0');
%                 end
%                 j = int2str(i);
%                 name = strcat(name, j);
                    
                while(correct(k,1)>i)
                    k=k-1;
                end
                while(correct(k,1)<i)
                    k=k+1;
                end
                
                if(correct(k,2) == 0 || correct(k,1) ~= i)
                    k=k+1;
                    continue;
                end
 
                predData(i,l) = pred1;
                predData(i,l+1) = pred2;
                predData(i,61) = correct(k,2);
                fprintf('\n %s', name);
                names = cat(1,names, name);
                
%                 if(correct(k,2) ~= pred)
%                     incorrect = incorrect+1;
%                     fprintf(fid, '%s\t%d <= Incorrect\n',name, pred);
%                 else
%                     count = count+1;
%                     fprintf(fid, '%s\t%d\n',name, pred);
%                 end
                %k=k+1;
            end 
        
        end
        %fprintf(facc, '%s\t%.2f\n',fname, (incorrect/k));
        ybound = ybound + 0.05;
        l=l+2;
    end
    xbound = xbound + 0.05;
end

%predData(:,61) = correct(:,2);
k=1;
for i=1:175
    if (predData(i,:) == 0)
        continue;
    else
        pData(k,:) = predData(i,:);        
    end
    k=k+1;
end

pData = uint8 (pData);
save('dataset2', 'pData');
save('vidnames', 'names');
end