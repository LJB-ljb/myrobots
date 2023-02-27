function fit_xy = fit(x,y)
% ¼ÆËãÊÊÓ¦¶È
        m1 = (x>=-3)&(x<=-1.5)&((y>=-3)&(y<=3));
        m2 = (x>=-3)&(x<=3)&((y>=-3)&(y<=-1.5));
        m3 = (x>=1.5)&(x<=3)&((y>=1.5)&(y<=3));
        if((m1|m2|m3))
            fit_xy = (sin((2*pi/3)*x))^2*(sin((2*pi/3)*y))^2*exp((x+y)/(5*pi));
        else
            fit_xy = -(sin((2*pi/3)*x))^2*(sin((2*pi/3)*y))^2*exp((x+y)/(5*pi));
        end
end

