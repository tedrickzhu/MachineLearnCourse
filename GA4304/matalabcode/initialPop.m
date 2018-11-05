function pop = initialPop(pop)

pop.chro = round(rand(pop.size,sum(pop.bn)));
pop.solution = decoding(pop);

pop.obj = fun(pop.solution(:,1),pop.solution(:,2));

pop.prob = computeProb(pop.obj);

pop.bestC = pop.chro(find(pop.obj == min(pop.obj),1),:);
pop.bestS = pop.solution(find(pop.obj == min(pop.obj),1),:);
pop.performance = [min(pop.obj),mean(pop.obj)];

