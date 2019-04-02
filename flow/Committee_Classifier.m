function [committees] = Committee_Classifier(traindata,trainlabel,ratio,number)
    % 说明：
    % input:
        % traindata:    训练数据
        % trainlabel:   训练数据的标签
        % ratio:        比例值（0~1），从traindata中选择一定比例(ratio)的数据作为子分类器，组成委员会
        % number:       子分类器的个数（委员会成员的数据）
    % output:
        % committees:   训练得到的委员会，一个元胞数组，调用分类器时可用 committees{com_i}的形式调用第i个分类器
      
    committees = cell(1,number);
    for com_i = 1:number
        
        % 从traindata 中随机选择比例为ratio的数据作为子训练数据，用于训练委员会成员
        indices = crossvalind('Kfold',size(traindata,2),size(traindata,2));
        Train = (indices <= floor(size(traindata,2) * ratio)); 
        SLtrain = traindata(:,Train); SLlabel = trainlabel(:,Train);
        
        % /*    创建网络（可只修改此部分替换committees的网络类型）
        Classifier = trainSoftmaxLayer(SLtrain,SLlabel,'MaxEpoch',1000,'ShowProgressWindow',false); % 训练一个Classifier分类器
        % */    

        committees{com_i} = Classifier;
    end

end