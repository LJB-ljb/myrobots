function X = decimal2bin(len_zhongqun,len)
% 把一个十进制数组转变成给定长度的二进制数
%     dleta = (x_max-x_min)/(2^len-1);
    X=zeros(len_zhongqun,len);
    x=zeros(1,len);
    for i=1:len_zhongqun
        x_bin = dec2bin(i,len);
        for j=1:len
            if (x_bin(j)=='0')
                x(j)=0;
            elseif(x_bin(j)=='1')
                x(j)=1;
            end
        end
        X(i,:)= x;
    end
end