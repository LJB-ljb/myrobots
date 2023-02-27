% ��������ʵ��
% SX2201203 ���
clc;
clear;
%�������ݣ����ݼ���Դ�ڿα���4.1,��һ��Ϊ���ԣ����һ��Ϊ���
load('watermelon.mat');
[r_w,c_w] = size(watermelon);
%����ѵ����D�����Լ�A
D = watermelon(2:r_w,:); %DΪѵ����
A = watermelon(1,1:c_w-1);    %AΪ���Լ�
[r_D,c_D] = size(D);
%% �����µ�ѵ��������D_train
D_train = zeros(size(D));
D_value = cell(1,c_D);%����������ź�D_trainֵ�Ĺ�ϵ
% ������ֵ����D_value�У���D_train�е��������Ӧ
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

%�趨0��Ӧ�񣬲��Ǻùϣ�1��Ӧ�ǣ��Ǻù�
if ( D_value{1,end}{1,1} == '��' )
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

 %% ʹ��ID3�㷨�����ɾ�����   
tree = ID3(D_train,A,D_value);
print_tree(tree);


%% ID3
function tree = ID3(D_train,A,D_value)
% D_train��ѵ��������A�����Լ���D_value��Ӧ����ֵ
    [r,c] = size(D_train);%����������������Ľ��   
    %D_train������ȫ����ͬһ���,����ΪҶ�ڵ�
    if length( unique(D_train(:,end)) )==1
        if D_train(1,c) == 0
            tree = '��';
        elseif D_train(1,c) == 1
            tree = '��';
        end
        return
    end
    
    %AΪ�ջ���DΪ�գ�����ΪҶ�ڵ�
    if length(A)==0
    temp=tabulate(D_train(:,end));
    value=temp(:,1);            %����ֵ
    count=temp(:,2);  %��ͬ����ֵ�ĸ�������
    index=find(max(count)==count);
    choose=index(randi(length(index)));
    if D_train(1,c) == 0
        tree = '��';
        elseif D_train(1,c) == 1
            tree = '��';
        end
        return
    end  
    
 

    D_pn = {}; %��¼���������
    %Ԫ������D_pnÿ�ж�Ӧÿ�����ԣ���һ�ж�Ӧ�����������ڶ��ж�Ӧ������ţ������ж�Ӧ���������������ж�Ӧ�������
    k_max = max(D_train);% k_maxΪѵ����ÿ�����Ե�ȡֵ��������
        for j =1:c-1
            for i =1:k_max(j)
            D_pn{i,j}{1,1} = 0;
            D_pn{i,j}{2,1} = [];
            D_pn{i,j}{3,1} = 0;
            D_pn{i,j}{4,1} = [];
            end
        end
    gain = zeros(1,c-1);%��ʼ����������
    
    
    %��A��ѡ���������Ի�������
    
      %ȷ��ÿ�����Ե�ȡֵ����
      for j = 1:c-1   
          for i =1:r
              for k = 1:k_max(j)
              if ( D_train(i,j) == k )
                  if ( D_train(i,c) == 1 )%1��Ӧ�ǣ�0��Ӧ��
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
        
       %������Ϣ������
       tbl_result = tabulate( D_train(:,end) );
       %ent_DΪ��������D����Ϣ��
       ent_D = -(tbl_result(1,3)/100 * log2(tbl_result(1,3)/100) + tbl_result(2,3)/100 * log2(tbl_result(2,3)/100));
        
       %������Ϣ������
       %D_pn��¼�����������D_trainΪѵ������ent_DΪѵ���������棬k_maxΪѵ����ÿ�����Ե�ȡֵ��������
        gain = Gain(D_pn,D_train,ent_D,k_max);
        
        %����Ϣ������������ֵ��ѡ������������ӱ�ǩ��ȥ��
        [max_v , max_n] = max(gain);
        bestFeatureLabel = A{max_n};%��Ϣ������������ֵ��Ӧ�ı�ǩ
        featValues = D_train(:,max_n);
        bestFeatureValue = char(D_value{max_n}); %��Ϣ������������ֵ��Ӧ������
        uniqueVals = unique(featValues);%ȷ��Ҫ���ֵķ�֧��
        [r_A,c_A]=size(A);
        %����ǩ������ֵȥ��
        A = [A(1:max_n-1) A(max_n+1:c_A)];        
        D_value = [D_value(1:max_n-1) D_value(max_n+1:c_A)];
        %��������Ҷ�ڵ�
        tree = containers.Map;
        leaf = containers.Map;
        %�ݹ�ѭ��
        for i = 1:length(uniqueVals)
            A_sub = A;
            value = uniqueVals(i);
            D_sub =[];
            for j = 1:r
                data = D_train(j,:);
                if data(max_n) == value
                D_sub = [D_sub; [data(1:max_n-1) data(max_n+1:length(data))]];  %ȡ �������� ������ ��Ӧ������
                end
            end
            leaf(bestFeatureValue(value,:)) = ID3(D_sub,A_sub,D_value);
            tree(bestFeatureLabel) = leaf;
        end
        
end
%% ��������
function gain = Gain(D_pn,D_train,ent_D,k_max)
%������Ϣ������
% D_pn��¼�����������D_trainΪѵ����
% ent_DΪѵ���������棬k_maxΪѵ����ÿ�����Ե�ȡֵ��������
    [r , c] = size(D_train);
    gain =[];
    for j = 1:c-1
    ent = zeros(1,k_max(j));
    for i =1:k_max(j)
        % p_nΪ��������n_nΪ������
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

%% �Ѿ�����������
function print_tree(tree)
% �������������������nodeids���ڵ��ϵ����nodevalue���ڵ���Ϣ����branchvalue��֦����Ϣ��
nodeids(1) = 0;
nodeid = 0;
nodevalue={};
branchvalue={};

queue = {tree} ;      %�γɶ��У�һ��һ����ȥ
while ~isempty(queue)
    node = queue{1};
    queue(1) = [];                  %�ڶ����г�ȥ�ýڵ�
    if string(class(node))~="containers.Map" %Ҷ�ڵ�
        nodeid = nodeid+1;
        nodevalue = [nodevalue,{node}];
    elseif length(node.keys)==1 %�ڵ�Ļ�
        nodevalue = [nodevalue,node.keys];      %����ýڵ���
        node_info = node(char(node.keys));      %����ýڵ��µ����Զ�Ӧ��map
        nodeid = nodeid+1;
        branchvalue = [branchvalue,node_info.keys];   %ÿ���ڵ��µ�����
        for i=1:length(node_info.keys)
            nodeids = [nodeids,nodeid];
        end

        
    end
    
    if string(class(node))=="containers.Map" 
        keys = node.keys();
        for i = 1:length(keys)
            key = keys{i};
            queue=[queue,{node(key)}];                  %���б�ɸýڵ�����Ľڵ�
        end
    end
nodeids_=nodeids;
nodevalue_=nodevalue;
branchvalue_ = branchvalue;
end
[x,y,h] = treelayout(nodeids); %x:�����꣬y:�����ꣻh:�������
f = find(nodeids~=0); %��0�ڵ�
pp = nodeids(f); %��0ֵ
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
