% 决策树的实现
% SX2201203 李静波
clc;
clear;
%加载数据，数据集来源于课本表4.1,第一行为属性，最后一列为结果
load('watermelon.mat');
[r_w,c_w] = size(watermelon);
%创建训练集D与属性集A
D = watermelon(2:r_w,:); %D为训练集
A = watermelon(1,1:c_w-1);    %A为属性集
[r_D,c_D] = size(D);
%% 建立新的训练集数组D_train
D_train = zeros(size(D));
D_value = cell(1,c_D);%用于索引序号和D_train值的关系
% 将属性值存在D_value中，与D_train中的数字相对应
for j=1:c_D
    for i=1:r_D
    D_value{1,j} = unique(D(:,j));
            for k = 1:length(D_value{1,j})
            belong = strcmp( D_value{1,j}{k,1} , D{i,j});
                if (belong == 1)
                D_train(i,j) = k;
                end             
            end
    end
end

%设定0对应否，不是好瓜，1对应是，是好瓜
if ( D_value{1,end}{1,1} == '否' )
   for i=1:r_D
    if ( D_train(i,end) == 1)
        D_train(i,end) = 0 ;
    else
        D_train(i,end) = 1 ;
    end
   end
else
    for i=1:r_D
    if ( D_train(i,end) == 1)
        D_train(i,end) = 1 ;
    else
        D_train(i,end) = 0 ;
    end
    end   
end

 %% 使用ID3算法来生成决策树   
tree = ID3(D_train,A,D_value);
print_tree(tree);


%% ID3
function tree = ID3(D_train,A,D_value)
% D_train是训练集矩阵，A是属性集，D_value对应属性值
    [r,c] = size(D_train);%这里的列数包括最后的结果   
    %D_train中样本全属于同一类别,返回为叶节点
    if length( unique(D_train(:,end)) )==1
        if D_train(1,c) == 0
            tree = '否';
        elseif D_train(1,c) == 1
            tree = '是';
        end
        return
    end
    
    %A为空或者D为空，返回为叶节点
    if length(A)==0
    temp=tabulate(D_train(:,end));
    value=temp(:,1);            %属性值
    count=temp(:,2);  %不同属性值的各自数量
    index=find(max(count)==count);
    choose=index(randi(length(index)));
    if D_train(1,c) == 0
        tree = '否';
        elseif D_train(1,c) == 1
            tree = '是';
        end
        return
    end  
    
 

    D_pn = {}; %记录正反例情况
    %元胞数组D_pn每列对应每种属性，第一行对应正例数量，第二行对应反例编号，第三行对应反例数量，第四行对应反例编号
    k_max = max(D_train);% k_max为训练集每种属性的取值个数向量
        for j =1:c-1
            for i =1:k_max(j)
            D_pn{i,j}{1,1} = 0;
            D_pn{i,j}{2,1} = [];
            D_pn{i,j}{3,1} = 0;
            D_pn{i,j}{4,1} = [];
            end
        end
    gain = zeros(1,c-1);%初始化增益数组
    
    
    %从A中选择最优属性划分属性
    
      %确定每种属性的取值个数
      for j = 1:c-1   
          for i =1:r
              for k = 1:k_max(j)
              if ( D_train(i,j) == k )
                  if ( D_train(i,c) == 1 )%1对应是，0对应否
                    D_pn{k,j}{1,1}= D_pn{k,j}{1,1}+1;
                    D_pn{k,j}{2,1}(end+1) = i;
                  else
                    D_pn{k,j}{3,1}= D_pn{k,j}{3,1}+1;
                    D_pn{k,j}{4,1}(end+1) = i;
                  end
              end
              end
          end
      end
        
       %计算信息熵增益
       tbl_result = tabulate( D_train(:,end) );
       %ent_D为样本集合D的信息熵
       ent_D = -(tbl_result(1,3)/100 * log2(tbl_result(1,3)/100) + tbl_result(2,3)/100 * log2(tbl_result(2,3)/100));
        
       %计算信息熵增益
       %D_pn记录正反例情况，D_train为训练集，ent_D为训练集的增益，k_max为训练集每种属性的取值个数向量
        gain = Gain(D_pn,D_train,ent_D,k_max);
        
        %把信息增益最大的特征值挑选出来，并将其从标签中去除
        [max_v , max_n] = max(gain);
        bestFeatureLabel = A{max_n};%信息增益最大的特征值对应的标签
        featValues = D_train(:,max_n);
        bestFeatureValue = char(D_value{max_n}); %信息增益最大的特征值对应的属性
        uniqueVals = unique(featValues);%确定要划分的分支数
        [r_A,c_A]=size(A);
        %将标签和属性值去除
        A = [A(1:max_n-1) A(max_n+1:c_A)];        
        D_value = [D_value(1:max_n-1) D_value(max_n+1:c_A)];
        %生成树和叶节点
        tree = containers.Map;
        leaf = containers.Map;
        %递归循环
        for i = 1:length(uniqueVals)
            A_sub = A;
            value = uniqueVals(i);
            D_sub =[];
            for j = 1:r
                data = D_train(j,:);
                if data(max_n) == value
                D_sub = [D_sub; [data(1:max_n-1) data(max_n+1:length(data))]];  %取 该特征列 该属性 对应的数据
                end
            end
            leaf(bestFeatureValue(value,:)) = ID3(D_sub,A_sub,D_value);
            tree(bestFeatureLabel) = leaf;
        end
        
end
%% 计算增益
function gain = Gain(D_pn,D_train,ent_D,k_max)
%计算信息熵增益
% D_pn记录正反例情况，D_train为训练集
% ent_D为训练集的增益，k_max为训练集每种属性的取值个数向量
    [r , c] = size(D_train);
    gain =[];
    for j = 1:c-1
    ent = zeros(1,k_max(j));
    for i =1:k_max(j)
        % p_n为正例数，n_n为反例数
        p_n = D_pn{i,j}{1,1};
        n_n = D_pn{i,j}{3,1};
        if (p_n == 0 | n_n == 0)
            ent(i) = 0;
        else
        ent(i) = -( (p_n/(p_n+n_n)) * log2(p_n/(p_n+n_n)) + (n_n/(p_n+n_n)) * log2(n_n/(p_n+n_n)));
        end
        ent(i) = (p_n+n_n) * ent(i)/r;
    end
    gain(end+1) = ent_D - sum(ent);
    end
end

%% 把决策树画出来
function print_tree(tree)
% 层序遍历决策树，返回nodeids（节点关系），nodevalue（节点信息），branchvalue（枝干信息）
nodeids(1) = 0;
nodeid = 0;
nodevalue={};
branchvalue={};

queue = {tree} ;      %形成队列，一个一个进去
while ~isempty(queue)
    node = queue{1};
    queue(1) = [];                  %在队列中除去该节点
    if string(class(node))~="containers.Map" %叶节点
        nodeid = nodeid+1;
        nodevalue = [nodevalue,{node}];
    elseif length(node.keys)==1 %节点的话
        nodevalue = [nodevalue,node.keys];      %储存该节点名
        node_info = node(char(node.keys));      %储存该节点下的属性对应的map
        nodeid = nodeid+1;
        branchvalue = [branchvalue,node_info.keys];   %每个节点下的属性
        for i=1:length(node_info.keys)
            nodeids = [nodeids,nodeid];
        end

        
    end
    
    if string(class(node))=="containers.Map" 
        keys = node.keys();
        for i = 1:length(keys)
            key = keys{i};
            queue=[queue,{node(key)}];                  %队列变成该节点下面的节点
        end
    end
nodeids_=nodeids;
nodevalue_=nodevalue;
branchvalue_ = branchvalue;
end
[x,y,h] = treelayout(nodeids); %x:横坐标，y:纵坐标；h:树的深度
f = find(nodeids~=0); %非0节点
pp = nodeids(f); %非0值
X = [x(f); x(pp); NaN(size(f))];
Y = [y(f); y(pp); NaN(size(f))];

X = X(:);
Y = Y(:);

n = length(nodeids);
if n<500
    hold on;
    plot(x,y,'ro',X,Y,'r-')
    nodesize = length(x);
    for i=1:nodesize
        text(x(i)+0.01,y(i),nodevalue{1,i});      
    end
    for i=2:nodesize
        j = 3*i-5;
        text((X(j)+X(j+1))/2-length(char(branchvalue{1,i-1}))/200,(Y(j)+Y(j+1))/2,branchvalue{1,i-1})
    end
    hold off
else
    plot(X,Y,'r-');
end
xlabel(['height = ' int2str(h)]);
axis([0 1 0 1]);

end
