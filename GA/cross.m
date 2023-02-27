function huanchong = cross(huanchong ,fit_max, p_c ,x_min , delta_x , y_min , delta_y , cross_fanwei)
    [r,c] = size(huanchong);
    %染色体交叉率
    num = round(r*p_c/2);
    X = bin2decimal( huanchong(:,(1:c/2)) );
    Y = bin2decimal( huanchong(:,(c/2+1:c)) );
    X_dec = x_min + X*delta_x;
    Y_dec = y_min + Y*delta_y;
%     XY_dec  = [X_dec Y_dec];

    for i=1:num
        k = true;
        while k
          j = randi([2,r],1,2);
            if (j(1)~=j(2))
                k = false;
            end      
        end
        
        cross1 = huanchong(j(1),:);
        cross2 = huanchong(j(2),:);
        if ( fit_max - fit(X_dec(j(1)),Y_dec(j(1)))>0.1*fit_max & fit_max - fit(X_dec(j(2)),Y_dec(j(2)))>0.1*fit_max )
        
       %% 小范围交叉
        if ( cross_fanwei == '1' )
        %交换前半段，即对应x
        cross_for_change = cross1(1,(c/2-1:c/2));
        cross1(1,(c/2-1:c/2)) = cross2(1,(c/2-1:c/2));
        cross2(1,(c/2-1:c/2)) = cross_for_change;
        %交换后半段，即对应y
        cross_for_change = cross1(1,(c-1:c));
        cross1(1,(c-1:c)) = cross2(1,(c-1:c));
        cross2(1,(c-1:c)) = cross_for_change;
        huanchong(j(1),:)=cross1;
        huanchong(j(2),:)=cross2;
        end
        %% 大范围交叉
        if ( cross_fanwei == '2' )
        %交换前半段，即对应x
        cross_for_change = cross1(1,(1:c/4));
        cross1(1,(1:c/4)) = cross2(1,(1:c/4));
        cross2(1,(1:c/4)) = cross_for_change;
        %交换后半段，即对应y
        cross_for_change = cross1(1,(c/2+1:c/2+1+c/4));
        cross1(1,(c/2+1:c/2+1+c/4)) = cross2(1,(c/2+1:c/2+1+c/4));
        cross2(1,(c/2+1:c/2+1+c/4)) = cross_for_change;
        huanchong(j(1),:)=cross1;
        huanchong(j(2),:)=cross2;
        end
        end
        
    end
end