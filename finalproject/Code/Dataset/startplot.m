function [ ] = startplot( filePath )
%MAKEPREDICTION Summary of this function goes here
%   Detailed explanation goes here
%   Region beyond the xbound and more than ybound will only be considered 
%   for the weighted voting.
%   Right side, col is one and weight is positive.
%   Weight = linear combination of the 2 features. f1+f2.


if exist(filePath, 'file')
    
    data = importdata(filePath);
    %f1 = data(:,8);     %length
    %f2 = data(:,10);    %distance
    f3 = data(:,11);    %direction
    f4 = data(:,2);
    f5 = 480-data(:,3);
    

    %%
    for i=1:size(f4)
        %if ((f1(i) > ybound) && (f2(i) > xbound))
            
            if(f3(i)==1) 
                plot(f4(i),f5(i), '*', 'color', 'r');hold on
            else
                plot(f4(i),f5(i), '*', 'color', 'b');hold on
            end
        %end
    end
    
    %line([0.7 0.7 1], [1 0.1 0.1]);
    f = getframe();
    f = f.cdata();
    name = strcat(filePath,'_start.png');
    imwrite(f, name);
    hndl = gcf();
    close(hndl);
    hold off;
end

