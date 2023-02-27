%% ��ʼ��
clear,clc;
% x = -3:0.01:3;
% y = -3:0.01:3;
% [X,Y] = meshgrid(x,y);
x_min = -3; x_max = 3;
y_min = -3; y_max = 3;

%��������
diedai = 300;
%Ⱦɫ��ĳ���Ϊ2*12
len = 12;
%��Ⱥ�ĳ�ʼ����
len_zhongqun = 50;
%Ⱦɫ�彻�����
p_c = 0.05;
cross_fanwei = 2; % 1������С��2�����1��
%Ⱦɫ��������
variation_p = 3; % 3���1��С
%Ⱦɫ�巢������ĸ���
p_nm = 0.1;
%����������
p_m = 0.1;
%����
range = 0.5;
%��һ������x���ڶ�������y,����������ֵ
fit_max = [0 0 0];
% XY_dec = zeros(2^len,2);
delta_x = (x_max-x_min)/(2^len-1);
delta_y = (y_max-y_min)/(2^len-1);
fitness = zeros(1,len_zhongqun);

%% ����
X_bin = decimal2bin(len_zhongqun,len);
Y_bin = decimal2bin(len_zhongqun,len);
chromosome =[X_bin ,Y_bin];
for i_diedai =1:diedai
%% ����
    X = bin2decimal( chromosome(:,(1:len)) );
    Y = bin2decimal( chromosome(:,(len+1:2*len)) );
    X_dec = x_min + X*delta_x;
    Y_dec = y_min + Y*delta_y;
    XY_dec  = [X_dec Y_dec];

%% ������Ӧ�Ⱥ����Լ�����ѡ��
    for i_fit=1:len_zhongqun
        fitness(i_fit) = fit(XY_dec(i_fit,1),XY_dec(i_fit,2))+3;%����Ϊ��ֹ���ָ�ֵ
    end

    [fit_max(3),max_position] = max(fitness);
    fit_max(1) = XY_dec(max_position,1); fit_max(2) = XY_dec(max_position,2);
    fit_mean = mean(fitness);
    fit_max_plot(i_diedai)=fit_max(3);
    fit_mean_plot(i_diedai)=fit_mean;
    plot(i_diedai,fit_max(3),'g')
    plot(i_diedai,fit_mean,'r')
    
    huanchong = chromosome;
    [r_huanchong , c_huanchong] = size(huanchong);
    
% %ʹ��ת�ַ�   
%     huanchong(1,:) = chromosome(max_position,:);
%     for i=2:r_huanchong
%         number = copy_zhuanlun(fitness);
%         huanchong(i,:) = chromosome(number,:);
%     end

%ʹ����Ӧ������
    number = copy_rank(fitness, range);
    for i=1:r_huanchong
        huanchong(i,:) = chromosome(number(i),:);
    end
    
%% ���н��桢���졢�������
%ִ�н������
    if ((i_diedai < 3*diedai/4)&(i_diedai >= diedai/2))
        cross_fanwei = 1;
    elseif(i_diedai > 3*diedai/4)
        continue;
    end
    huanchong = cross(huanchong,fit_max(3),p_c,x_min , delta_x , y_min , delta_y , cross_fanwei);
%ִ�б������
    if ( i_diedai < 4*diedai/5 && i_diedai >= diedai/2 )
        variation_p =2;
    elseif (i_diedai >= 4*diedai/5 )
        variation_p =1;
    end
    huanchong = variation(huanchong,fit_max(3),variation_p ,x_min , delta_x , y_min , delta_y);
    %ִ�в������
    huanchong = insert(huanchong,p_nm,p_m);
    chromosome = huanchong;
end
hold on
plot(1:diedai,fit_max_plot,'g')

plot(1:diedai,fit_mean_plot,'r')
hold off

