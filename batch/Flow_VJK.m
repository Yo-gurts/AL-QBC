clc; clear;
load Batch.mat C_data C_label;

Methods = {'VE', 'JS', 'KL'};				% 三种分歧度的度量方法
global Record R_divergence k;				% 数据记录准备

batch = 1;                                  % 用作初始训练数据的板子

% #1 准备数据，加载A、B、C，准备训练、询问、测试数据
A = C_data{1};  Alabel = C_label{1};


for Main_i = [1:batch-1, batch+1:10]    % 使用不同的板子来进行测试
    
    QT = C_data{Main_i}; QTlabel = C_label{Main_i};
    % 保存原始训练、询问、测试数据
    O_traindata = A;            O_trainlabel = Alabel;
    indices = crossvalind('Kfold',size(QT,2),2);
    query = (indices == 1); 
    O_querydata = QT(:,query); 	O_querylabel = QTlabel(:,query);
    O_testdata = QT(:,~query);  O_testlabel = QTlabel(:,~query);

    m = 1;
    for item = Methods	% 不同的度量方法
        method = item{1};
        % 加载数据
        traindata = O_traindata; 	trainlabel = O_trainlabel;
        testdata = O_testdata; 		testlabel = O_testlabel;
        querydata = O_querydata; 	querylabel = O_querylabel;

        Record = [];			% 记录对testdata以及querydata的分类正确率以及此时的训练数据的长度
        R_divergence = {};  	% 记录每一次对询问数据评价的分歧度
        Chairman_cell = {};		% 记录每一次训练后的分类器

        [Accuracy, Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel);
        Chairman_cell{1} = Chairman;
        for k = 1:50

            % 训练大小为5的委员会
            [Committees] = Committee_Classifier(traindata,trainlabel,0.75,5);
            % 选择信息量最大的样本
            [querydata,querylabel,traindata,trainlabel,m_divergence] = ...
                Select(querydata,querylabel,traindata,trainlabel,Committees,method);
            % 用增加了样本后的训练集训练主席，并测试其正确率
            [Accuracy, Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel);
            Chairman_cell{k+1} = Chairman;

        end

        RECORD{Main_i, m} = Record;
        m = m+1;
        str = sprintf('save batch%d_%s_100record_%02d',batch, method, Main_i);
        eval(str);
    end

end

Flow_RAND;  % 随机的情况


%% 准备数据
% 		indices = crossvalind('Kfold',size(A,2),size(A,2));
% 		Train = (indices <= floor(size(A,2) * percent/100)); 
% 		O_traindata = A(:,Train); 	O_trainlabel = Alabel(:,Train);


%                 if (method == "VE" ) && (m_divergence < 0.3)
%                     break;
%                 end