function [querydata,querylabel,traindata,trainlabel]...
    = Select(querydata,querylabel,traindata,trainlabel,committees,method)
    
    global R_divergence k;
    %% 说明： 此函数用于选择“信息量最大”的样本，加入到训练集;
        % input: 
            % querydata:    询问集，在此样本集中选择“信息量最大”的样本加入到训练集
            % querylabel:   询问数据的标签
            % traindata:    训练数据集，将选出的样本加进训练数据集，并最后返回
            % trainlabel:   训练数据的标签
            % committees:   由各个分类器组成的委员会
            % R_divergence: 记录数据的变量，记录的是一次选择中所有样本的分歧度的计算值。
            % flag:         用于选取不同的分歧度测量方法,如下：
            %                     1: KL
            %                     2: VE
            %                     3: JS
            %                     4: VE+1JS
            %                     5: VE+JS
                        
        % output:
            % querydata:    从输入的querydata中减去被挑选出的样本后的数集
            % querylabel:   与输出的querydata对应的标签
            % traindata:    加入了从querydata中选出的样本后的数集
            % trainlabel:   与输出traindata对应的标签
            % R_divergence: 用于记录数据
        % 判断样本“信息量”：
            % 让委员会中的成员对询问数据集(querydata)进行分类，再用一些分歧度度量的方法(KL 、 VE)
            % 来判断对一个样本的预测中，几个委员会成员之间的分歧，分歧越大，认为信息量大。   

    dataclass = size(querylabel,1);     % 统计总的类别数
    numofcom = length(committees);      % 委员会的大小
    
    %% 用委员会对询问数据进行预测，并准备相关数据
    % predictresult 记录每个委员会成员对样本的分类结果,如5个分类器，它们对第一个样本的
    % 分类结果为 predictresult(1,:) = [1,1,1,3,1];即第4个分类器对这个样本的预测为第3类数据
    predictresult = zeros(length(querydata),length(committees));
       
    %% 让 numofcom 个委员会成员依次对querydata进行预测，预测结果保存在 Predict_cell
    Predict_cell = {numofcom,1};
    for Sel_i = 1:numofcom  
        Prediction = committees{Sel_i}(querydata);      % 第k个分类器对querydata进行预测
                                                        % 将第k个分类器的预测结果存放在Predict_cell中
        Predict_cell{Sel_i} = Prediction;
        count=vec2ind(compet(Prediction))';             % 将分类器的预测结果以一列数的形式表示
        predictresult(:,Sel_i) = count;                 % 将第k个分类器的结果放在Cvotes的第k列中
    end
    
    %% 投票熵：D(e) = -1/log[min(K,C)] * sum{V(c,e)/K  *  log(V(c,e)/K)};
    % VE_V 记录每个委员会成员分类结果统计之后的情况，VE_V(1,:)=[4,0,1,0,0,0];表示
    % 有四个分类器对这个样本的类别预测为第一类，有1个分类器认为它是第3类
    
    VE_V = zeros(length(querydata),size(querylabel,1));     % V 
    VE_K = numofcom;                                        % K
    for VE_m = 1:dataclass
        VE_V(:,VE_m) = sum((predictresult == VE_m),2);
    end
    temp_VE = sum((VE_V./VE_K) .* log2(VE_V / VE_K + eps ), 2);
    VE = (-1 / log2(min(VE_K,dataclass)) * temp_VE)';

    %% KL divergence
%     Pavg = 0;
%     for KL_i = 1:numofcom
%         Pavg = Pavg + 0.2 * Predict_cell{KL_i};    % 0.2 = 1/5(numofcom)
%     end
%     temp_KLK = 0;
%     for KL_j = 1:numofcom
%         temp_KLC = Predict_cell{KL_j};             % 第KL_j个分类器的预测结果               
%         temp_KLD = sum(temp_KLC .* log2(temp_KLC ./ Pavg + eps), 2);
%         temp_KLK = temp_KLK + 0.2 * temp_KLD;
%     end
%     KL = temp_KLK;
    KL_temp = 1;
    for KL_i = 1:4
        for KL_j = KL_i+1:5
            KL(KL_temp,:) = KLdivergence(Predict_cell{KL_i}, Predict_cell{KL_j});
            KL_temp = KL_temp + 1;
        end
    end
    KL = sum(KL);
    %% JS divergence
    sumdistri = 0;  % 每一个委员会成员预测结果之和
    sum_Shan_individual = 0; % 
    for JS_i = 1:numofcom
        sumdistri = sumdistri + 0.2 * Predict_cell{JS_i}; 
        sum_Shan_individual = sum_Shan_individual + 0.2 * Shannon(Predict_cell{JS_i}); % (Wi * H(Pi))i=1->numofcom
    end
    JS = Shannon(sumdistri) - sum_Shan_individual;
   
    %% 根据选定的度量方法，仅选择一个分歧度最大的样本，同行条件选择最前面那一个
    switch method
        case 'KL'
            [~, subscript] = sort(KL);   
            Msubscript = subscript(:,length(querydata)-99:length(querydata)); % 找出分歧度最大的100个样本的位置
        case 'VE'
            [~, subscript] = sort(VE);   
            Msubscript = subscript(:,length(querydata)-99:length(querydata)); % 找出分歧度最大的100个样本的位置
        case 'JS'
            [~, subscript] = sort(JS);   
            Msubscript = subscript(:,length(querydata)-99:length(querydata)); % 找出分歧度最大的100个样本的位置
    end
    
%     R_divergence{k} = [VE;JS;KL];
    
    % 获得分歧度最大的样本并将其加入到训练集中
    Mdata = querydata(:,Msubscript);
    Mlabel = querylabel(:,Msubscript);
    traindata = [traindata,Mdata];
    trainlabel = [trainlabel,Mlabel];
    
    % 将已经加入到训练集的样本从询问集中删除
    querydata(:,Msubscript) = [];
    querylabel(:,Msubscript) = [];
    
end

% 计算香农熵的函数
function entropy = Shannon(P)
    entropy = -sum(P .* log2(P+eps));
end

function divergence = KLdivergence(P,Q)
    divergence = sum(P.* log2(P./Q + eps));
end