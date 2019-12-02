%   X: data matrix, each row is one observation, each column is one feature
%   D: pair-wise distance matrix

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its 
%   Applications in Face Recognition and Active Shape Models. 
%   arXiv:1207.3538 [cs.CV], 2012. 

function para=distanceMatrix(X)

% ԭ����
% N=size(X,1);
% 
% XX=sum(X.*X,2);
% XX1=repmat(XX,1,N);
% XX2=repmat(XX',N,1);
% 
% D=XX1+XX2-2*(X*X');
% D(D<0)=0;
% D=sqrt(D);


% %���Լ��ĵĴ���
% N = size(X,1); %���ж��ٹ۲�����
% Dmin = zeros(N,1); %�洢ÿһ���۲�����������۲������ľ�����Сֵ
% 
% parfor N1 = 1:N
%     Ddistance = zeros(N,1); %�洢ĳһ�۲����������������ľ���
%     for N2 = 1:N
%         if(N1 ~= N2)
%             Ddistance(N2) = sqrt(sum(power((X(N1,:)-X(N2,:)),2)));
%         else
%             Ddistance(N2) = inf;
%         end
%     end
%     Dmin(N1) = min(Ddistance);
% end
% D = mean(Dmin);


%���Լ��ĵĴ���2
para = 1;%Ĭ��para����1
N = size(X,1); %���ж��ٹ۲�����
if(N <= 1000)
    DIST = pdist2(X,X);
    DIST(DIST == 0) = inf;
    DIST = min(DIST);
    para = mean(DIST);
elseif(N > 1000)
    numBlock = ceil(N/1000); %��Ҫ��Ϊ���ٿ�� 
    eachBlock = ones(1,numBlock)*1000; %����ÿһ�����������ÿһ�ж�Ϊ4000
    eachBlock(numBlock)=N-(numBlock-1)*1000; %��һ�������һ���޸�ΪXʣ�������
    XtoBlock = mat2cell(X,eachBlock); %�ֿ��
    minDIST = zeros(N,1);
    for iBlock = 1:numBlock
        DISTtemp = pdist2(XtoBlock{iBlock,1},X);
        DISTtemp(DISTtemp == 0) = inf;
        xtemp = min(DISTtemp,[],2);
        [tempM1,~]=size(xtemp);
        xtemp1 = find(minDIST==0);
        minDIST(xtemp1(1):(xtemp1(1)+tempM1-1))=xtemp;
    end
    para = mean(minDIST);
end
end