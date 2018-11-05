function solution = decoding(pop)

solution = zeros(pop.size,pop.xNum);
p = [0;cumsum(pop.bn)];
for i = 1:pop.size
    for j = 1:pop.xNum
        solution(i,j) = converting(pop.chro(i,p(j)+1:p(j+1)),pop.xRange(j,:),pop.bl(j));
    end
end

%-----------------------------------------------------------
function s = converting(c,r,rr)

a = [0:1:length(c)-1];
b = 2.^a;
d = sum(c.*b);
s = r(1)+d*rr;













