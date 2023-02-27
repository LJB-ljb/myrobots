%% 初始化
clear,clc;
% x = -3:0.01:3;
% y = -3:0.01:3;
% [X,Y] = meshgrid(x,y);
x_min = -3; x_max = 3;
y_min = -3; y_max = 3;

%迭代次数
diedai = 300;
%染色体的长度为2*12
len = 12;
%种群的初始数量
len_zhongqun = 50;
%染色体交叉概率
p_c = 0.05;
cross_fanwei = 2; % 1代表最小，2代表比1大
%染色体变异概率
variation_p = 3; % 3最大，1最小
%染色体发生插入的概率
p_nm = 0.1;
%基因插入概率
p_m = 0.1;
%代沟
range = 0.5;
%第一个代表x，第二个代表y,第三个代表值
fit_max = [0 0 0];
% XY_dec = zeros(2^len,2);
delta_x = (x_max-x_min)/(2^len-1);
delta_y = (y_max-y_min)/(2^len-1);
fitness = zeros(1,len_zhongqun);

%% 编码
X_bin = decimal2bin(len_zhongqun,len);
Y_bin = decimal2bin(len_zhongqun,len);
chromosome =[X_bin ,Y_bin];
for i_diedai =1:diedai
%% 解码
    X = bin2decimal( chromosome(:,(1:len)) );
    Y = bin2decimal( chromosome(:,(len+1:2*len)) );
    X_dec = x_min + X*delta_x;
    Y_dec = y_min + Y*delta_y;
    XY_dec  = [X_dec Y_dec];

%% 计算适应度函数以及进行选择
    for i_fit=1:len_zhongqun
        fitness(i_fit) = fit(XY_dec(i_fit,1),XY_dec(i_fit,2))+3;%加三为防止出现负值
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
    
% %使用转轮法   
%     huanchong(1,:) = chromosome(max_position,:);
%     for i=2:r_huanchong
%         number = copy_zhuanlun(fitness);
%         huanchong(i,:) = chromosome(number,:);
%     end

%使用适应度排序法
    number = copy_rank(fitness, range);
    for i=1:r_huanchong
        huanchong(i,:) = chromosome(number(i),:);
    end
    
%% 进行交叉、变异、插入操作
%执行交叉操作
    if ((i_diedai < 3*diedai/4)&(i_diedai >= diedai/2))
        cross_fanwei = 1;
    elseif(i_diedai > 3*diedai/4)
        continue;
    end
    huanchong = cross(huanchong,fit_max(3),p_c,x_min , delta_x , y_min , delta_y , cross_fanwei);
%执行变异操作
    if ( i_diedai < 4*diedai/5 && i_diedai >= diedai/2 )
        variation_p =2;
    elseif (i_diedai >= 4*diedai/5 )
        variation_p =1;
    end
    huanchong = variation(huanchong,fit_max(3),variation_p ,x_min , delta_x , y_min , delta_y);
    %执行插入操作
    huanchong = insert(huanchong,p_nm,p_m);
    chromosome = huanchong;
end
hold on
plot(1:diedai,fit_max_plot,'g')

plot(1:diedai,fit_mean_plot,'r')
hold off

