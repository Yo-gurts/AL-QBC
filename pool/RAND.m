clc; clear;

divergence_name = 'RAND';  					% 选择分歧度的度量方法
global Record R_divergence k;				% 数据记录准备
		
Percentage = [3 5 10 25 50 75 100];			% 初始训练集占A的百分比
RECORD = cell(7,10);  						% 保存每一次的正确率

for Main_j = 1:10						% 每一种比例训练重复10次
    
    for Main_i = 1:7							% 用A中的3% 5% 10% 25% 50% 75% 100% 来训练
        
        str = sprintf('load E:\\Matlab\\Active_learning\\Query_by_committees\\T5_12\\Pool\\JS_%02drecord_%02d.mat  O_traindata O_trainlabel O_querydata O_querylabel O_testdata O_testlabel'...
            ,Percentage(Main_i), Main_j);
        eval(str);
        % 加载数据
        traindata = O_traindata; 	trainlabel = O_trainlabel;
        testdata = O_testdata; 		testlabel = O_testlabel;
        querydata = O_querydata; 	querylabel = O_querylabel;

		Record = [];    		% 记录对testdata以及querydata的分类正确率以及此时的训练数据的长度
		R_divergence = {};  	% 记录每一次对询问数据评价的分歧度
		Chairman_cell = {};		% 记录每一次训练后的分类器
		
		% 训练主席
		[Accuracy, Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel);
		Chairman_cell{1} = Chairman;

            indices = crossvalind('Kfold',size(querydata,2),size(querydata,2));
            Selecting = (indices <= 100); 
            Selected = querydata(:,Selecting); SelectedLabel = querylabel(:,Selecting);
            
            querydata(:,Selecting) = [];
            querylabel(:,Selecting) = [];
            
            % 将选中的样本加入到训练集
            traindata = [traindata,Selected];
            trainlabel = [trainlabel,SelectedLabel];
    

		   % 用增加了样本后的训练集训练主席，并测试其正确率
		   [Accuracy, Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel);
		   Chairman_cell{2} = Chairman;
				
		RECORD{Main_i,Main_j} = Record; % 保存正确率
				
		% 自动保存变量 
		s2 = sprintf('save %s_%02drecord_%02d.mat;',divergence_name,Percentage(Main_i),Main_j);
		eval(s2);
    end

end