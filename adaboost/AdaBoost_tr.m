function  adaboost_model  =  AdaBoost_tr(tr_func_handle,te_func_handle,train_set,labels,no_of_hypothesis)
    % 说明：
        % no_of_hypothesis: 训练弱分类器的个数

    adaboost_model = struct('weights',zeros(1,no_of_hypothesis),'parameters',[]); % cell(1,no_of_hypothesis));
        % weights:      每一个分类器在集成中所占的权重
        % parameters:   分类器

    sample_n = size(train_set,2);  

    samples_weight = ones(sample_n,1) / sample_n; % 每个样本的初始权重为1/n，n为样本个数

    for turn=1:no_of_hypothesis  

        model=tr_func_handle(train_set,samples_weight,labels);  % 训练一个弱分类器

        adaboost_model.parameters{turn} = model;  

        [L,hits,error_rate]=te_func_handle(adaboost_model.parameters{turn},train_set,samples_weight,labels);  

        if(error_rate==1)  
            error_rate=1-eps;  
        elseif(error_rate==0)  
            error_rate=eps;  
        end  
          
        % The weight of the turn-th weak classifier     计算弱分类器的权重
        adaboost_model.weights(turn) = log10((1 - error_rate) / error_rate);

        C = vec2ind(L);
        t_labeled = (C == vec2ind(labels));  % true labeled samples  正确分类了的样本
          
        % Importance of the true classified samples is decreased for the next weak classifier  
        % 降低被正确分类的样本的权重;
        samples_weight(t_labeled) = samples_weight(t_labeled).*((error_rate)/(1-error_rate));   
          
        % Normalization  
        samples_weight = samples_weight/sum(samples_weight);   % 训练样本权重标准化
    end

    % Normalization  
    adaboost_model.weights=adaboost_model.weights/sum(adaboost_model.weights); % 分类器权重标准化s