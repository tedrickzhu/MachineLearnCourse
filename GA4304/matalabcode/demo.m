clear all
close all


[X,Y] = meshgrid(-1:.02:1);
Z = fun(X,Y);
figure(1)
surf(X,Y,Z)
figure(2)
contour(X,Y,Z)

%-----------------------------------------------------------
pop.size = 50;
pop.mxGen = 200;
pop.cr = 0.6;
pop.mr = 0.1;
pop.xNum = 2;
pop.xRange = [-1,1;-1,1];
pop.xAc = [0.01;0.002];

pop = setting(pop);

%-----------------------------------------------------------
pop = initialPop(pop);

figure(2)
contour(X,Y,Z)
hold on
plot(pop.solution(:,1),pop.solution(:,2),'ok')
plot(pop.bestS(1,1),pop.bestS(1,2),'ok','MarkerFaceColor','r')

%-----------------------------------------------------------

for i = 1:pop.mxGen
    pop = newPop(pop);
    figure(2)
    contour(X,Y,Z)
    hold on
    plot(pop.solution(:,1),pop.solution(:,2),'ok')
    plot(pop.bestS(end,1),pop.bestS(end,2),'ok','MarkerFaceColor','r')
    hold off
    figure(3)
    plot(pop.performance)
end





