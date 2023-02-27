function huanchong = variation(huanchong,fit_max ,variation_p ,x_min , delta_x , y_min , delta_y)
% ±‰“Ï
[r,c] = size(huanchong);
if (variation_p == 3)
    p = 0.05;
elseif (variation_p == 2)
    p = 0.005;
elseif (variation_p == 1)
    p = 0.0001;
end
    
number = round(r*c*p);

    X = bin2decimal( huanchong(:,(1:c/2)) );
    Y = bin2decimal( huanchong(:,(c/2+1:c)) );
    X_dec = x_min + X*delta_x;
    Y_dec = y_min + Y*delta_y;
    
for i =1:number
    j = randi([2,r]);
    k = randi([1,c]);
    position = [j,k];
    
    if ( fit_max - fit(X_dec(position(1)),Y_dec(position(1)))>0.1*fit_max )
    %÷¥––±‰“Ï
        if (huanchong( position(1) , position(2) )==1)
            huanchong( position(1) , position(2) ) =0;
        elseif (huanchong( position(1) , position(2) )==0)
            huanchong( position(1) , position(2) )=1;
        end
    end
end
end