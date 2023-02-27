function func()
% ¿ÉÊÓ»¯
x = -3:0.01:3;
y = -3:0.01:3;
[X,Y] = meshgrid(x,y);
Z = (sin((2*pi/3)*X)).^2.*(sin((2*pi/3)*Y)).^2.*exp((X+Y)/(5*pi));
[r,c]=size(Z);

for i=1:r
    for j=1:c
        m1 = (x(i)>=-3)&(x(i)<=-1.5)&((y(j)>=-3)&(y(j)<=3));
        m2 = (x(i)>=-3)&(x(i)<=3)&((y(j)>=-3)&(y(j)<=-1.5));
        m3 = (x(i)>=1.5)&(x(i)<=3)&((y(j)>=1.5)&(y(j)<=3));
        if((m1|m2|m3))
            Z(i,j) = Z(i,j);
        else
            Z(i,j) = -Z(i,j);
        end
    end
end
Z = Z+ 3;
set(gca,'ZLim',[0 3]);
surf(X,Y,Z)
end