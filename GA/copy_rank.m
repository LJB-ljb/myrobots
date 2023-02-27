function number = copy_rank(fitness, range)
% 使用适应度排序法进行选择
%     fitness是一个50*1的行向量
    len = length(fitness)*range;
    number = zeros(size(fitness));
    for i = 1:len
        [value,index] = max(fitness);
        number(i) = index;
        fitness(index) = 0;
    end
    for i = len+1:length(fitness)
        number(i) = number(randi([1 len]));
    end
end