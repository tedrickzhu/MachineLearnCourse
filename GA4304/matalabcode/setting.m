function pop = setting(pop)

pop.bn = zeros(pop.xNum,1);
for i = 1:pop.xNum
    r = (pop.xRange(i,2)-pop.xRange(i,1))/pop.xAc(i);
    pop.bn(i) = 1;
    while 2^pop.bn(i)-1 < r
        pop.bn(i)=pop.bn(i)+1;
    end
    pop.bl(i) = (pop.xRange(i,2)-pop.xRange(i,1))/(2^pop.bn(i)-1);
end




