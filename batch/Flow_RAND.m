clc; clear;
% batch 需要设置

global Record R_divergence k;				% 数据记录准备
batch = 1;

for Main_i = [1:batch-1, batch+1:10]    % 使用不同的板子来进行测试
    
    str = sprintf('load batch1_JS_100record_%02d.mat O_traindata O_trainlabel O_querydata O_querylabel O_testdata O_testlabel',Main_i);
    eval(str);
    
    m = 1;
    traindata = O_traindata; 	trainlabel = O_trainlabel;
    testdata = O_testdata; 		testlabel = O_testlabel;
    querydata = O_querydata; 	querylabel = O_querylabel;

    Record = [];			% 记录对testdata以及querydata的分类正确率以及此时的训练数据的长度
    R_divergence = {};  	% 记录每一次对询问数据评价的分歧度
    Chairman_cell = {};		% 记录每一次训练后的分类器

    [Accuracy, Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel);
    Chairman_cell{1} = Chairman;

    for k = 1:50
        % 从B中随机选择一个样本加入到训练集
        indices = crossvalind('Kfold',size(querydata,2),size(querydata,2));
        Selecting = (indices == 1); 
        Selected = querydata(:,Selecting); SelectedLabel = querylabel(:,Selecting);
       % 将选中的样本加入到训练集
        traindata = [traindata,Selected];
        trainlabel = [trainlabel,SelectedLabel];

        % 将已经加入到训练集的样本从询问集中删除
        querydata(:,Selecting) = [];
        querylabel(:,Selecting) = [];

        % 用增加了样本后的训练集训练主席，并测试其正确率
        [Accuracy, Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel);
        Chairman_cell{k+1} = Chairman;

    end

    RECORD{Main_i, m} = Record;
    m = m+1;
    str = sprintf('save batch1_RAND_100record_%02d', Main_i);
    eval(str);
end