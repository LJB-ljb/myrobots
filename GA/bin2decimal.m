function X = bin2decimal(X_bin)
% ��һ������������ת���ʮ��������
[r,c]=size(X_bin);    
X =zeros(1,r);
    for i=1:r
        for j=1:c
        X(i) = X(i) + X_bin(i,j)*2^(c-j);
        end
    end
    
    X  = X';
end