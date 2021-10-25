function decision_tree
clear;
clc;
close all;
Sample = importdata('iris.txt');
n = length(Sample);
sample = zeros(n,5);
for i=1:n
   S = regexp(char(Sample(i)) ,',', 'split');
   for j = 1:4
       sample(i,j) = str2double(char(S(j)));
   end
   if strcmpi(S(5),"Iris-setosa")
       sample(i,5) = 0;
   elseif (strcmpi(S(5),"Iris-versicolor"))
      sample(i,5) = 1;
   else
       sample(i,5) = 2;
   end
end

maketree(sample);
end
%% ����һ����

function [] = maketree(sample)
    [sample1,sample2] = childtree(sample,0);
    [sample3,sample4] = childtree(sample1,1);
    [sample5,sample6] = childtree(sample2,2);    %���ĸ߶�Ϊ2
    
    
    %-1����û��
    tree_c3=struct('parent',1,'class',decideclass(sample3),'value',sample3,'child1',0,'child2',0);
    tree_c4=struct('parent',1,'class',decideclass(sample4),'value',sample4,'child1',0,'child2',0);
    tree_c5=struct('parent',2,'class',decideclass(sample5),'value',sample5,'child1',0,'child2',0);
    tree_c6=struct('parent',2,'class',decideclass(sample6),'value',sample6,'child1',0,'child2',0);
    tree_c1=struct('parent',0,'class',decideclass(sample1),'value',sample1,'child1',tree_c3,'child2',tree_c4);
    tree_c2=struct('parent',0,'class',decideclass(sample2),'value',sample2,'child1',tree_c5,'child2',tree_c6);
    tree_root=struct('parent',-1,'class',-1,'value',sample,'child1',tree_c1,'child2',tree_c2);
    
    %������ȷ��
    r = (testTree(sample3)+testTree(sample4)+testTree(sample5)+testTree(sample6))/4;
    
    fprintf('����Ϊ����\n���������ֱ�Ϊ��%i��%i��%i��%i\n',length(sample3(:,1)),length(sample4(:,1))-1,length(sample5(:,1)),length(sample6(:,1))-1);
    fprintf('�������ֱ�Ϊ��%i��%i��%i��%i\n',tree_c3.class,tree_c4.class,tree_c5.class,tree_c6.class);
    fprintf('�þ���������ȷ��Ϊ��%.2f\n',r);
    %%  ��֦
    num = 0;
    %���ڵ�
    if testTree(tree_root.value)>=(testTree(tree_root.child1.value)+testTree(tree_root.child2.value))/2
        tree_root.child1 = 0;
         tree_root.child2 = 0;
         num=num+1;
    end
    
    %�ڲ��ڵ�1
    if testTree(tree_c1.value)>=(testTree(tree_c1.child1.value)+testTree(tree_c1.child2.value))/2
        tree_c1.child1 = 0;
         tree_c1.child2 = 0;
         num=num+1;
    end
    %�ڲ��ڵ�2
    if testTree(tree_c2.value)>=(testTree(tree_c2.child1.value)+testTree(tree_c2.child2.value))/2
        tree_c2.child1 = 0;
        tree_c2.child2 = 0;
        num=num+1;
    end
   %fprintf('��ɼ�֦%i��\n',num);
   %r = (testTree(sample1)+testTree(sample2))/2;
   %fprintf('��ȷ�ʱ�Ϊ��%.2f\n',r);
    
    
end

%% ������Ϣ��
function E = entropy(sample,f)
    [n,~] = size(sample);
    x0 = 0;
    x1 = 0; 
    x2 = 0; 
    for i = 1:n
       switch sample(i,f)
           case 0
               x0 =x0+1;
           case 1
               x1 =x1+1;
           case 2
               x2 =x2+1;
       end
    end
    p0 = x0/n;
    p1 = x1/n;
    p2 = x2/n;
    
    if (p0 ==0)
        s(1) = 0;
    else
        s(1)=(p0*log2(p0));
    end
    
    if (p1 ==0)
        s(2) = 0;
    else
        s(2)=(p1*log2(p1));
    end
    
    if (p2 ==0)
        s(3) = 0;
    else
        s(3)=(p2*log2(p2));
    end
    
    E =-sum(s);
end

%% ��Ϣ���棨������
function [Gbest,Tbest] = Gain(sample,f,n)
    A = sort(unique(sample(:,f)));
    T = (A(1:end-1)+A(2:end))/2;
    num = length(T);
    G = zeros(1,num);     %ÿһ�ֿ��ܵ���Ϣ����
    for j = 1:num
        [s1,s2] = dividetree_hard(sample,f,T(j));
        G(j) = entropy(sample,5)-(length(s1(:,1))/n*entropy(s1,5)+length(s2(:,1))/n*entropy(s2,5));
    end
    [Gbest,index] = max(G);
    Tbest = T(index);
end

%% ��Ϣ���棨��ɢ��
function [Gbest,Tbest] = Gain_discrete(sample,f,n)
    T = sort(unique(sample(:,f)));
    num = length(T);
    G = zeros(1,num);     %ÿһ�ֿ��ܵ���Ϣ����
    for j = 1:num
        [s1,s2] = dividetree_discrete(sample,f,T(j));
        G(j) = entropy(sample,5)-(length(s1(:,1))/n*entropy(s1,5)+length(s2(:,1))/n*entropy(s2,5));
    end
    [Gbest,index] = max(G);
    Tbest = T(index);
end

%% �õ�����
function [sample1,sample2] = childtree(sample,time)

    [n,~] = size(sample);
    
    if (n ==1 || testTree(sample)==1)
        sample1 = sample;
        sample2 = 0;
    elseif n==0
        sample1 = 0;
        sample2 = 0;
    else
        
        G = zeros(1,4);
    T = zeros(1,4);
    for i=1:4
        [G(i),T(i)] = Gain(sample,i,n); 
    end
    [Gb,index] = max(G);      
    Tb = T(index);            %�����ѷ�����
   
    switch time
        case 0
            fprintf('���ڸ��ڵ㣬���õ� %i���������������з��࣬���ŷ����׼TΪ��%.2f����Ϣ����Ϊ%.2f\n',index, Tb,Gb);
        case 1
             fprintf('�����ڲ��ڵ�1�����õ� %i���������������з��࣬���ŷ����׼TΪ��%.2f����Ϣ����Ϊ%.2f\n',index, Tb,Gb);
        case 2
             fprintf('�����ڲ��ڵ�2�����õ� %i���������������з��࣬���ŷ����׼TΪ��%.2f����Ϣ����Ϊ%.2f\n',index, Tb,Gb);
    end
    
    [sample1,sample2] = dividetree_soft(sample,index,Tb);
    %{
    a=1;
    b=1;
    for i = 1:n              %ʵ�ַ���
        if (sample(i,index) < Tb)
            sample1(a,:) = sample(i,:);
            a = a+1;
        else
            sample2(b,:) = sample(i,:);
            b=b + 1;
        end
    end
  %}
    end
end

%%  ��������Ӳ�߽磩
function [sample1,sample2] = dividetree_hard(sample,f,T)
    a=1;
    b=1;
    [n,~] = size(sample);
    for i = 1:n              %ʵ�ַ���
        if (sample(i,f) < T)
            sample1(a,:) = sample(i,:);
            a = a+1;
        else
            sample2(b,:) = sample(i,:);
            b=b + 1;
        end
    end
end

%%  ����������߽磩
function [sample1,sample2] = dividetree_soft(sample,f,T)
    a=1;
    b=1;
    [n,~] = size(sample);
    x1 = min(sample(:,f));
    x2 = max(sample(:,f));
    range = (x2-x1)/10;
    
    for i = 1:n              %ʵ�ַ���
        if (sample(i,f) < T-range)
            sample1(a,:) = sample(i,:);
            a = a + 1;
        elseif(sample(i,f) > T+range)
            sample2(b,:) = sample(i,:);
            b = b + 1;
        else
            sample1(a,:) = sample(i,:);
            sample2(b,:) = sample(i,:);
            a = a + 1;
            b = b + 1;
        end
    end
    
end

function [sample1,sample2] = dividetree_discrete(sample,f,T)
    a=1;
    b=1;
    [n,~] = size(sample);
    for i = 1:n              %ʵ�ַ���
        if (sample(i,f) == T)
            sample1(a,:) = sample(i,:);
            a = a+1;
        else
            sample2(b,:) = sample(i,:);
            b=b + 1;
        end
    end
end


%% ����Ч��
function [rate] = testTree(sample)

    [n,m] = size(sample);
    if(n==1&&m==1)
        rate = 1;
    else
        x0 = 0;
        x1 = 0; 
        x2 = 0; 
    for i = 1:n
       switch sample(i,5)
           case 0
               x0 =x0+1;
           case 1
               x1 =x1+1;
           case 2
               x2 =x2+1;
       end
    end
        p(1) = x0/n;
        p(2) = x1/n;
        p(3) = x2/n;
        rate=max(p);
    end
    
end
%% �õ����
function [class] = decideclass(sample)
    [n,m]=size(sample);
    if (m~=1)
        
        x0 = 0;
        x1 = 0; 
        x2 = 0; 
        for i = 1:n
            switch sample(i,5)
                case 0
                    x0 =x0+1;
                case 1
                    x1 =x1+1;
                case 2
                    x2 =x2+1;
                end
        end
        p(1) = x0/n;
        p(2) = x1/n;
        p(3) = x2/n;
        if  (p(1) == max(p))
            class = 0;
        elseif(p(2) == max(p))
            class = 1;
        else 
            class = 2;
        end
    else
        class = -1;
    end
end