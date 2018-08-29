function prob = computeProb(obj)

fit = max(obj)-obj;

if max(fit) == 0
    fit(:) = 1;
    prob = fit/sum(fit);
else
    prob = fit/sum(fit);
end



