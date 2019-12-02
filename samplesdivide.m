function [DataTest, DataTrain, CTest, CTrain] = samplesdivide(indian_pines_corrected,indian_pines_gt,train,randpp)

[m, n, p] = size(indian_pines_corrected); %145*145*30
CTrain = [];CTest = [];
DataTest  = [];
DataTrain = [];
%indian_pines_map =uint8(zeros(m,n));%zeros(145,145)删除了indian_pines_map及返回值indian_pines_map
data_col = reshape(indian_pines_corrected,m*n,p);%dataDR→21025*30

for i = 1:max(indian_pines_gt(:))%对于地面真实地类分布中的每一个地类
    ci = length(find(indian_pines_gt==i)); %每个地类的个数   
    [v]=find(indian_pines_gt==i);    %每个地类在真实分布中的像素位置
    datai = data_col(indian_pines_gt==i,:);  %每类真实地物所对应的dataDR的像素值(datai) 46*30,1428*30,830*30,237*30,483*30...
    if train>1
        cTrain = round(train);%如果train（trainpercentage）＞1，则向train最近的方向取整。
    elseif train<1
        cTrain = round(ci*train);%如果train（trainpercentage）＜1，则用每个地类的总个数乘以train，再取整，这时，train代表的是每个地类取多少百分比
    end
    if train>ceil(ci/2)%每个地类像素数的一半，并向上取整
        cTrain = ceil(ci/2);%如果train超过该类的一半以上，则cTrain为该类别的一半
    end
    cTest  = ci-cTrain;
    CTrain = [CTrain cTrain];%[类1训练样本数量 类2训练样本数量 类3训练样本数量...]
    CTest = [CTest cTest];%[类1测试样本数量 类2测试样本数量 类3测试样本数量...]
    index = randpp{i};%randpp的构造为1*16的cell，每一个cell里，1*46,1*1428,1*830,1*237,1*483这样的每一个地类，从1到像素数的随机排序
    DataTest = [DataTest; datai(index(1:cTest),:)];%dataDR中，每一类真实地物所对应的像素值中，前cTest项组成测试集
    DataTrain = [DataTrain; datai(index(cTest+1:cTest+cTrain),:)];    %dataDR中，每一类真实地物所对应的像素值中，第cTest+1到该地类的像素数项组成训练集
   
    %indian_pines_map(v(index(1:cTest))) = i;    
end

%%%%%%%% Normalize
DataTest = fea_norm(DataTest);%9812*30
DataTrain = fea_norm(DataTrain);%437*30
 