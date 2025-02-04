function [ pred ] = makeprediction( filePath, xbound, ybound )
%MAKEPREDICTION Summary of this function goes here
%   Detailed explanation goes here
%   Region beyond the xbound and more than ybound will only be considered 
%   for the weighted voting.
%   Right side, col is one and weight is positive.
%   Weight = linear combination of the 2 features. f1+f2.

pred =0;
if exist(filePath, 'file')
    
    data = importdata(filePath);
    f1 = data(:,8);     %length
    f2 = data(:,10);    %distance
    f3 = data(:,11);    %direction
    
    wt(:,1) = data(:,8) + data(:,10);
    finalwt=0;
    for i=1:size(f1,1)
        
        if ((f1(i) > ybound) && (f2(i) > xbound))
            if f3(i) == 1
                finalwt = finalwt + 1; % wt(i);  %right
            else
                finalwt = finalwt - 1; %wt(i);  %left
            end
        end
    end
    
    if(finalwt > 0)
        pred = 1;
    else
        pred = 2;
    end

    %%
    for i=1:size(f1)
        %if ((f1(i) > ybound) && (f2(i) > xbound))
            
            if(f3(i)==1) 
                plot(f2(i),f1(i), '*', 'color', 'r'), xlim([0 1]), ylim([0 1]);hold on
            else
                plot(f2(i),f1(i), '*', 'color', 'b'), xlim([0 1]), ylim([0 1]);hold on
            end
        %end
    end
    line([0.7 0.7 1], [1 0.1 0.1]);
    f = getframe();
    f = f.cdata();
    name = strcat(filePath,'_pred.png');
    imwrite(f, name);
    hndl = gcf();
    close(hndl);
    hold off;
end


end

