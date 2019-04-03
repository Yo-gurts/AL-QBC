%  每一列表示一个样本

clear; clc;
load Batch.mat C_data C_label;  % 加载数据

Data = C_data{10};              % 数据
Label = C_label{10};            % 标签

indices = crossvalind('Kfold',3600,6);      % 随机
trainindices = (indices == 1);        testindices = ~trainindices;  
traindata = Data(:, trainindices);    testdata = Data(:, testindices);    % 训练数据600个
trainlabel = Label(:, trainindices);  testlabel = Label(:, testindices);  % 测试数据3000个

% 数据归一化（可选）
[traindata,PS]=mapminmax(traindata,0,1);              %要进行一次转置，这样的归一化是对列（也就是每个传感器响应曲线）的归一化，也有的是对每一次采样得到的数据进行归一化。 
testdata=mapminmax('apply',testdata,PS);

% setdemorandstream(pi);        % 固定随机值,

% 训练i个弱分类器
i = 20;
adaboost_model = AdaBoost_tr(@Threshold_tr, @Threshold_te, traindata, trainlabel, i); % 用adaboost法训练i个弱分类器，放在model中；

% 训练样本测试  
% [L_tr,hits_tr] = AdaBoost_te(adaboost_model, @Threshold_te, traindata, trainlabel); % L_tr为训练样本的分类结果，hits_tr为正确分类的样本个数。
% tr_n = size(traindata,2);
% tr_error = (tr_n-hits_tr)/tr_n;

% 测试样本测试 ，hits_te是正确分类的个数
[L_te,hits_te] = AdaBoost_te(adaboost_model, @Threshold_te, testdata, testlabel); 

te_n = size(testdata, 2);   % 得到测试数据的数量
te_error = (te_n - hits_te) / te_n * 100; % 错误率

Accuracy = 100 - te_error;  % 正确率

fprintf('the Accuracy is: %6.2f%%, the length of traindata is %d.\n', Accuracy, length(traindata));