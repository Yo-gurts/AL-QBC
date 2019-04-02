% 这个就是主程序，只需要运行这个

clc; clear;
load Celldata.mat;							% 加载数据，预先随机将3600个数据分成了A,B,C三份

Methods = {'VE', 'JS', 'KL'};				% 三种分歧度的度量方法
Percentages = [3 5 10 25 50 75 100];		% 初始训练集占A的百分比
global Record R_divergence k;				% 数据记录准备

% #1 准备数据，加载A、B、C，准备训练、询问、测试数据
A = ABCcell{1};     Alabel = ABClabelcell{1}; 
B = ABCcell{2};     Blabel = ABClabelcell{2};
C = ABCcell{3};     Clabel = ABClabelcell{3};

RECORD = cell(10,3);
for Main_i = 1:10                       % 重复做10次

	for percent = Percentages		% 3 5 10 25 50 75 100
		%% 准备数据
		indices = crossvalind('Kfold',size(A,2),size(A,2));
		Train = (indices <= floor(size(A,2) * percent/100)); 	% 训练数据分别取A一定的的百分比

		% 保存原始数据
		O_traindata = A(:,Train); 	O_trainlabel = Alabel(:,Train);
		O_querydata = B; 			O_querylabel = Blabel;
		O_testdata = C; 			O_testlabel = Clabel;		
		m = 1;	

		for item = Methods	% 不同的度量方法 VE, JS, KL
          
            % 加载数据，同一百分比下，不同方法使用的数据相同。
           	traindata = O_traindata; 	trainlabel = O_trainlabel;
			testdata = O_testdata; 		testlabel = O_testlabel;
			querydata = O_querydata; 	querylabel = O_querylabel;
            
            method = item{1};		% method为字符串
			Record = [];			% 记录对testdata以及querydata的分类正确率以及此时的训练数据的长度
			R_divergence = {};  	% 记录每一次对询问数据评价的分歧度
			Chairman_cell = {};		% 记录每一次训练后的分类器
		   	
		   	% 输入训练、询问、测试数据，使用全部训练数据训练，再对测试数据和询问数据进行测试。
		   	% 返回测试数据的正确率，和训练好的分类器
		   	[Accuracy, Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel);
		   	Chairman_cell{1} = Chairman;
			
			for k = 1:100 
				% 训练大小为5的委员会
				[Committees] = Committee_Classifier(traindata,trainlabel,0.75,5);
				% 选择信息量最大的样本
                [querydata,querylabel,traindata,trainlabel] = ...
					Select(querydata,querylabel,traindata,trainlabel,Committees,method);
			   	% 用增加了样本后的训练集训练主席，并测试其正确率
			   	[Accuracy, Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel);
			   	Chairman_cell{k+1} = Chairman;
			end

			RECORD{Main_i, m} = Record;	
			m = m+1;
			str = sprintf('save %s_%02drecord_%02d', method, percent, Main_i);	% 保存数据
			eval(str);
		end

	end

end
