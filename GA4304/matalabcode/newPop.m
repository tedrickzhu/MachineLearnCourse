function pop = newPop(pop)

pop = selection(pop);
pop = crossover(pop);
pop = mutation(pop);

pop.chro(ceil(rand*pop.size),:) = pop.bestC(end,:);

pop.solution = decoding(pop);

pop.obj = fun(pop.solution(:,1),pop.solution(:,2));

pop.prob = computeProb(pop.obj);

pop.bestC(end+1,:) = pop.chro(find(pop.obj == min(pop.obj),1),:);
pop.bestS(end+1,:) = pop.solution(find(pop.obj == min(pop.obj),1),:);
pop.performance(end+1,:) = [min(pop.obj),mean(pop.obj)];

%-----------------------------------------------------------
function pop = selection(pop)
cp = cumsum(pop.prob);

for i = 1:pop.size
    id(i) = find(cp>=rand,1);
end

pop.chro = pop.chro(id,:);

%-----------------------------------------------------------
function pop = crossover(pop)

for i = 1:floor(pop.size/2)
    if rand<=pop.cr
        c1 = pop.chro(i*2-1,:);
        c2 = pop.chro(i*2,:);
        r = ceil(rand*(length(c1)-1));
        pop.chro(i*2-1,:) = [c1(1:r),c2(r+1:end)];
        pop.chro(i*2,:) = [c2(1:r),c1(r+1:end)];
    end
end

%-----------------------------------------------------------
function pop = mutation(pop)

for i = 1:pop.size
    if rand<=pop.mr
        r = ceil(rand*(size(pop.chro,2)));
        pop.chro(i,r) = abs(pop.chro(i,r)-1);
    end
end






