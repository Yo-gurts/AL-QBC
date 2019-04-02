clc; clear;

Methods = {'RAND'};				% 三种分歧度的度量方法
Percentages = [3 5 10 25 50 75 100];			% 初始训练集占A的百分比
global Record R_divergence k;				% 数据记录准备

RECORD = cell(10,3);
for Main_i = 1:10

	for percent = Percentages
		%% 准备数据
		str = sprintf('load JS_%02drecord_%02d.mat O_traindata O_trainlabel O_querydata O_querylabel O_testdata O_testlabel',percent, Main_i);
        eval(str);
		m = 1;

		for item = Methods	% 不同的度量方法
           	%% 加载数据
           	traindata = O_traindata; 	trainlabel = O_trainlabel;
			testdata = O_testdata; 		testlabel = O_testlabel;
			querydata = O_querydata; 	querylabel = O_querylabel;
            
            method = item{1};
			Record = [];			% 记录对testdata以及querydata的分类正确率以及此时的训练数据的长度
			R_divergence = {};  	% 记录每一次对询问数据评价的分歧度
			Chairman_cell = {};		% 记录每一次训练后的分类器
		   	
		   	[Accuracy, Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel);
		   	Chairman_cell{1} = Chairman;
            
			for k = 1:100 
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
			str = sprintf('save %s_%02drecord_%02d', method, percent, Main_i);
			eval(str);

		end

	end

end