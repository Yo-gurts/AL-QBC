function model = Threshold_tr(traindata, samples_weight, trainlabel)

    % 此函数仅用于训练一个弱分类器；
    
% model = struct('min_error',[],'net',[],'C',[]);    
% min_error = sum(samples_weight); 

% 训练一个弱分类器；
BP=newff(traindata,trainlabel,8,{'tansig' 'tansig'},'trainlm');
BP.LW{2,1}=BP.LW{2,1}*0.01;             % 权值
BP.b{2}=BP.b{2}*0.01;                   % 阈值
BP.trainparam.epochs=100;             	%最大训练次数
BP.trainparam.goal=0.01;               	%训练误差
[BP,tr]=train(BP,traindata,trainlabel);

model = BP;

% trainPrediction = sim(BP,traindata);
% % 统计分类情况
% trainClassPrediction=compet(trainPrediction);
% trainClass=vec2ind(trainClassPrediction);
% ttrain=vec2ind(trainlabel);
% 
% error = 0;
% for i = 1:length(ttrain)
%     if(trainClass(i) ~= ttrain(i))
%         error = error+samples_weight(i);
%     end
% end
% 
% model.min_error = error;
% model.net = BP;
% model.C = ttrain;