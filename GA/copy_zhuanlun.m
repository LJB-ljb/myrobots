function number = copy_zhuanlun(fitness)
%P_chrom应该作为一个行向量
P_all = sum(fitness);
P_chrom = fitness/P_all;
p_max = max(P_chrom);
size_p = length(P_chrom);
zhuanlun = zeros(1,size_p+1);
for i=2:size_p+1
    zhuanlun(i) = P_chrom(i-1)+zhuanlun(i-1);
end
p=rand;
number = 0;
for j=1:size_p
    if (P_chrom(j)==p_max)
        number = j;
    elseif (p>=zhuanlun(j) & p<zhuanlun(j+1))
        number = j ;
    end
end
end