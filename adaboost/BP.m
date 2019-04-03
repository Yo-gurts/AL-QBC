
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

setdemorandstream(pi);        % 固定随机值,
% BP神经网络的初始权值或阈值是随机的。这里固定随机值（正常使用应注释掉）

% build the back_propagation neural network （搭网络结构及设定相关参数）
BPnet=newff(traindata,trainlabel,8,{'tansig','softmax'},'trainlm');
BPnet.LW{2,1}=BPnet.LW{2,1}*0.01;               % （权值）weight value
BPnet.b{2}=BPnet.b{2}*0.01;                     % （阈值）threshold value
BPnet.trainParam.showWindow = true;            % （不显示训练的过程）close the training window
BPnet.trainParam.showCommandLine = false;  
BPnet.performFcn='crossentropy' ;               % （误差性能函数）error performance function
BPnet.performFcn='mse';
BPnet.trainparam.epochs=1000;               % （最大迭代次数）the maximum number of iterations
BPnet.trainparam.goal=0.01;                 % （训练目标）the performance goal
BPnet.trainparam.mc=0.1;                    % （学习率）momentum factor 
BPnet.trainParam.max_fail=10; 

[BPnet,tr] = train(BPnet,traindata,trainlabel);   % 训练网络

testPrediction = sim(BPnet,testdata);          % 用测试数据进行测试
% 测试分类精度
testClassPrediction=compet(testPrediction);
testClass1=vec2ind(testClassPrediction);
ttest=vec2ind(testlabel);

count = sum(testClass1 == ttest);

Accuracy1 = 100*count/length(testlabel);
disp('测试数据的正确率为：');disp(Accuracy1);