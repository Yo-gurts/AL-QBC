
function [Accuracy,Record] = Chairman_Ensemble(traindata, trainlabel,testdata,testlabel,Record)
    % 训练i个弱分类器
    i = 20;
    adaboost_model = AdaBoost_tr(@Threshold_tr,@Threshold_te,traindata,trainlabel,i); % 用adaboost法训练i个弱分类器；

    % 测试样本测试   L_te: 分类的结果   hits_te: 正确分类的个数
    [L_te,hits_te] = AdaBoost_te(adaboost_model,@Threshold_te,testdata,testlabel);

    % 计算正确率
    te_n = size(testdata,2);
    te_error = (te_n-hits_te)/te_n*100;
    Accuracy = 100 - te_error;

    disp('the Accuracy is ');disp(Accuracy);
    disp('the length of traindata is ');disp(length(traindata));
    Record = [Record;Accuracy,length(traindata)];

end

% 训练集成分类器
function  adaboost_model  =  AdaBoost_tr(tr_func_handle,te_func_handle,train_set,labels,no_of_hypothesis)

    adaboost_model = struct('weights',zeros(1,no_of_hypothesis),'parameters',[]); %cell(1,no_of_hypothesis));

    sample_n = size(train_set,2);  
    samples_weight = ones(sample_n,1)/sample_n;

    for turn=1:no_of_hypothesis  
        model=tr_func_handle(train_set,samples_weight,labels);  % 训练一个弱分类器
        adaboost_model.parameters{turn} = model;  
        [L,hits,error_rate]=te_func_handle(adaboost_model.parameters{turn},train_set,samples_weight,labels);  
        if(error_rate==1)  
            error_rate=1-eps;  
        elseif(error_rate==0)  
            error_rate=eps;  
        end  

        % The weight of the turn-th weak classifier  
        adaboost_model.weights(turn) = log10((1-error_rate)/error_rate);
        C = vec2ind(L);
        t_labeled = (C == vec2ind(labels));  % true labeled samples  

        % Importance of the true classified samples is decreased for the next weak classifier  
        samples_weight(t_labeled) = samples_weight(t_labeled).*((error_rate)/(1-error_rate));  % 降低被正确分类的样本的权重； 

        % Normalization  
        samples_weight = samples_weight/sum(samples_weight);   % 训练样本权重标准化
    end

    % Normalization  
    adaboost_model.weights=adaboost_model.weights/sum(adaboost_model.weights); % 分类器权重标准化s
end

% 测试集成分类器
function [L,hits] = AdaBoost_te(adaboost_model,te_func_handle,test_set,true_labels)

    hypothesis_n = length(adaboost_model.weights);	% 集成的个数
    sample_n = size(test_set,2);  					% 测试样本数
    class_n = size(true_labels,1);  % class_n = length(unique(true_labels)); % 类别数
	% 记录每一个分类器的分类结果
    temp_L = zeros(class_n,sample_n,hypothesis_n);  % likelihoods for each weak classifier  

    % for each weak classifier, likelihoods of test samples are collected  
    for i=1:hypothesis_n  
        [temp_L(:,:,i),hits,error_rate] = te_func_handle(adaboost_model.parameters{i},test_set,ones(sample_n,1),true_labels);  
        temp_L(:,:,i) = temp_L(:,:,i)*adaboost_model.weights(i);  
    end  
    L = sum(temp_L,3);  	% 最终的分类结果为 L
    hits = sum(vec2ind(L) == vec2ind(true_labels));		% 集成分类器正确分类的个数。
end

% 训练单个分类器
function model = Threshold_tr(traindata, samples_weight, trainlabel)

    % 此函数仅用于训练一个弱分类器；
    % model = struct('min_error',[],'net',[],'C',[]);    
    % min_error = sum(samples_weight);    
    % 训练一个弱分类器；
    BP=newff(traindata,trainlabel,8,{'tansig' 'tansig'},'trainlm');
    BP.LW{2,1}=BP.LW{2,1}*0.01;             % 权值
    BP.b{2}=BP.b{2}*0.01;                   % 阈值
    BP.trainparam.epochs=100;             %最大训练次数
    BP.trainparam.goal=0.01;               %训练误差
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
end

% 测试单个分类器
function [L,hits,error_rate] = Threshold_te(model,test_set,samples_weight,true_labels) 

    testPrediction = sim(model,test_set);
    % 统计分类情况
    testClassPrediction=compet(testPrediction);
    testClass=vec2ind(testClassPrediction);
    ttest = vec2ind(true_labels);

    hits = sum(ttest == testClass);
    error = 0;
    for i = 1:length(ttest)
        if(testClass(i) ~= ttest(i))
            error = error+samples_weight(i);
        end
    end

    error_rate = error;
    L = testClassPrediction;
end