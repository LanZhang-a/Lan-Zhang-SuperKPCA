function [disc_set] =Eigenface_f(Train_SET,Eigen_NUM)
%------------------------------------------------------------------------
% Eigenface computing function
% mTrain_SET = mean(Train_SET,2);
% Dis = dist2(mTrain_SET',Train_SET');
% [val ind]=sort(Dis);
% Train_SET = Train_SET(:,ind(1:round(0.95*size(Train_SET,2))));

[NN,Train_NUM]=size(Train_SET);

% if NN<=Train_NUM % not small sample size case
    
   Mean_Image=mean(Train_SET,2);  %��ÿһ�еľ�ֵ
   Train_SET=Train_SET-Mean_Image*ones(1,Train_NUM);%ԭ���ݵ�ÿһ�����ص��ڸ�����ֵ��ȥ����ƽ��ֵ
   R=Train_SET*Train_SET'/(Train_NUM-1);%��ȥƽ��ֵ��ľ��󣬳�������ת�ã��ٳ������ظ���-1
   %����������������Э���������
   
%    subplot(2,1,1);plot(Train_SET(:,1:10:end));   xlim([1 103])
%    subplot(2,1,2);imagesc(corr(Train_SET'));
%    subplot(2,1,2);
%    imagesc(R);xlabel('Band Index');ylabel('Band Index');
   
   [V,S]=Find_K_Max_Eigen(R,Eigen_NUM);%������ֵ����������
%    ratio = S(1)/S(2);
%    plot(S)
%    subplot(1,2,1);semilogy(S);subplot(1,2,2);plot(Train_SET);
   disc_value=S;
   disc_set=V;

% else % for small sample size case
%      
%   Train_SET=Train_SET-(mean(Train_SET,2))*ones(1,Train_NUM);
%   R=Train_SET'*Train_SET/(Train_NUM-1);
%   [V,S]=Find_K_Max_Eigen(R,Eigen_NUM);
%   clear R
%   disc_set=zeros(NN,Eigen_NUM);
%   Train_SET=Train_SET/sqrt(Train_NUM-1);
%   
%   for k=1:Eigen_NUM
%     a = Train_SET*V(:,k);
%     b = (1/sqrt(S(k)));
%     disc_set(:,k)=b*a;
%   end

% end

function [Eigen_Vector,Eigen_Value]=Find_K_Max_Eigen(Matrix,Eigen_NUM)

[NN,NN]=size(Matrix);

%��ȡ����ֵ(200*200,�Խ���Ԫ��Ϊ����ֵ������������(200*200,ÿһ��Ϊ��Ӧ������������)
[V,S]=eig(Matrix);        %Note this is equivalent to; [V,S]=eig(St,SL); also equivalent to [V,S]=eig(Sn,St); %

S=diag(S); %ȡ����ֵ�ĶԽ���Ԫ�� ��200*1��
[S,index]=sort(S);

Eigen_Vector=zeros(NN,Eigen_NUM); %200*30
Eigen_Value=zeros(1,Eigen_NUM);  %1*30

p=NN;  %p=200
for t=1:Eigen_NUM
    Eigen_Vector(:,t)=V(:,index(p));%ȡ��200�е��������������������ĵ�1�У�ȡ��199�з����2�У���������
    Eigen_Value(t)=S(p);%ȡ��200�е�����ֵ�����������ĵ�1�У�ȡ��199�з����2�У���������
    p=p-1;
end



