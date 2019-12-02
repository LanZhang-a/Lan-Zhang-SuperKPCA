function [DataTest, DataTrain, CTest, CTrain] = samplesdivide(indian_pines_corrected,indian_pines_gt,train,randpp)

[m, n, p] = size(indian_pines_corrected); %145*145*30
CTrain = [];CTest = [];
DataTest  = [];
DataTrain = [];
%indian_pines_map =uint8(zeros(m,n));%zeros(145,145)ɾ����indian_pines_map������ֵindian_pines_map
data_col = reshape(indian_pines_corrected,m*n,p);%dataDR��21025*30

for i = 1:max(indian_pines_gt(:))%���ڵ�����ʵ����ֲ��е�ÿһ������
    ci = length(find(indian_pines_gt==i)); %ÿ������ĸ���   
    [v]=find(indian_pines_gt==i);    %ÿ����������ʵ�ֲ��е�����λ��
    datai = data_col(indian_pines_gt==i,:);  %ÿ����ʵ��������Ӧ��dataDR������ֵ(datai) 46*30,1428*30,830*30,237*30,483*30...
    if train>1
        cTrain = round(train);%���train��trainpercentage����1������train����ķ���ȡ����
    elseif train<1
        cTrain = round(ci*train);%���train��trainpercentage����1������ÿ��������ܸ�������train����ȡ������ʱ��train�������ÿ������ȡ���ٰٷֱ�
    end
    if train>ceil(ci/2)%ÿ��������������һ�룬������ȡ��
        cTrain = ceil(ci/2);%���train���������һ�����ϣ���cTrainΪ������һ��
    end
    cTest  = ci-cTrain;
    CTrain = [CTrain cTrain];%[��1ѵ���������� ��2ѵ���������� ��3ѵ����������...]
    CTest = [CTest cTest];%[��1������������ ��2������������ ��3������������...]
    index = randpp{i};%randpp�Ĺ���Ϊ1*16��cell��ÿһ��cell�1*46,1*1428,1*830,1*237,1*483������ÿһ�����࣬��1�����������������
    DataTest = [DataTest; datai(index(1:cTest),:)];%dataDR�У�ÿһ����ʵ��������Ӧ������ֵ�У�ǰcTest����ɲ��Լ�
    DataTrain = [DataTrain; datai(index(cTest+1:cTest+cTrain),:)];    %dataDR�У�ÿһ����ʵ��������Ӧ������ֵ�У���cTest+1���õ���������������ѵ����
   
    %indian_pines_map(v(index(1:cTest))) = i;    
end

%%%%%%%% Normalize
DataTest = fea_norm(DataTest);%9812*30
DataTrain = fea_norm(DataTrain);%437*30
 