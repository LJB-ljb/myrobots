function huanchong = insert(huanchong,p_nm,p_m)
% ִ�в������
[r_huanchong , c_huanchong] = size(huanchong);
n = round(r_huanchong*p_nm);
for i = 1:n
% ѡ��Ⱦɫ��
index_c = randi([1 r_huanchong]);
temp = huanchong(index_c,:);
% ��������
    if rand(1)<p_m                              % ���ñ������
        if rand(1)<0.5
            pos1=randi([2 c_huanchong/2]);                     % �����������λ��
            pos2=randi([2 c_huanchong/2]);
        else
            pos1=randi([c_huanchong/2+1 c_huanchong-1]);                     % �����������λ��
            pos2=randi([c_huanchong/2+1 c_huanchong-1]);
        end

        if pos1>pos2                           % �����ǰ��
           Gene = temp(pos1);                  % �ȱ���λ��1�ϵĻ��� 
           for j = pos1:-1:pos2                % ��λ��2��λ��1֮�䣨����λ��2���Ļ��������һ��λ��
               temp(j) = temp(j-1);
           end
           temp(pos2) = Gene;                  % �ٽ������λ��1�Ļ���ֵλ��2
        elseif pos1<pos2                       % �������
            Gene = temp(pos1);                 % �ȱ���λ��1�ϵĻ���
            for j = pos1:pos2                  % ��λ��1��λ��2֮�䣨����λ��2���Ļ�����ǰ��һ��λ��
                temp(j) = temp(j+1);
            end
            temp(pos2) = Gene;                 % �ٽ������λ��1�Ļ���ֵλ��2
        end
    end
huanchong(index_c,:)= temp;
end
end



