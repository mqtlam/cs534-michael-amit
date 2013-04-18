function scatter_plot( X, y, w )
%SCATTER_PLOT function to display the scatter plot

plot(0,0);hold on;
for i=1:size(X,1)
    if(y(i) == 1)
        plot(X(i,1),X(i,2), '*b');
    else
        plot(X(i,1),X(i,2), '*g');
    end
end

x0 = 0;
y0 = -w(1)/w(2);
x1 = -w(1)/w(3);
y1 = 0;
line([x0 y0], [x1 y1]);
hold off;
end

