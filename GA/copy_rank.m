function number = copy_rank(fitness, range)
% ʹ����Ӧ�����򷨽���ѡ��
%     fitness��һ��50*1��������
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