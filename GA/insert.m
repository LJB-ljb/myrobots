function huanchong = insert(huanchong,p_nm,p_m)
% 执行插入操作
[r_huanchong , c_huanchong] = size(huanchong);
n = round(r_huanchong*p_nm);
for i = 1:n
% 选择染色体
index_c = randi([1 r_huanchong]);
temp = huanchong(index_c,:);
% 交换基因
    if rand(1)<p_m                              % 设置变异概率
        if rand(1)<0.5
            pos1=randi([2 c_huanchong/2]);                     % 随机产生两个位置
            pos2=randi([2 c_huanchong/2]);
        else
            pos1=randi([c_huanchong/2+1 c_huanchong-1]);                     % 随机产生两个位置
            pos2=randi([c_huanchong/2+1 c_huanchong-1]);
        end

        if pos1>pos2                           % 如果向前插
           Gene = temp(pos1);                  % 先保存位置1上的基因 
           for j = pos1:-1:pos2                % 将位置2和位置1之间（包括位置2）的基因向后移一个位置
               temp(j) = temp(j-1);
           end
           temp(pos2) = Gene;                  % 再将保存的位置1的基因赋值位置2
        elseif pos1<pos2                       % 如果向后插
            Gene = temp(pos1);                 % 先保存位置1上的基因
            for j = pos1:pos2                  % 将位置1和位置2之间（包括位置2）的基因向前移一个位置
                temp(j) = temp(j+1);
            end
            temp(pos2) = Gene;                 % 再将保存的位置1的基因赋值位置2
        end
    end
huanchong(index_c,:)= temp;
end
end



