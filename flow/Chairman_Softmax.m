function [Accuracy,Chairman] = Chairman_Softmax(traindata,trainlabel,testdata,testlabel,querydata,querylabel)
	%% 说明：此函数用于训练一个分类器
		% input:
			% traindata:    训练数据输入
			% trainlabel:   训练数据的标签
			% testdata:		测试数据输入
			% testlabel:	测试数据的标签
			% querydata:	询问数据
			% querylabel:	询问数据的标签
			% Record：		用于保存相关数据
		% output:
			% Accuracy: 	用测试数据测试分类器时的正确率
			% Chairman: 	训练得到的分类器
			% Record: 		在输入的Record上加上此次相关数据的记录

	global Record;			% 全局变量

	Chairman = trainSoftmaxLayer(traindata,trainlabel,'MaxEpoch',1000,'ShowProgressWindow',false); % 训练一个Softmax分类器
	
	% 对testdata预测并统计分类正确率
	testPrediction = Chairman(testdata);
	testClassPrediction = compet(testPrediction);
	testClass = vec2ind(testClassPrediction);
	ttest = vec2ind(testlabel);
	count = sum(testClass == ttest);
	Accuracy = 100*count/length(testlabel);
	
	% 对querydata进行预测并统计分类正确率
	queryPrediction = Chairman(querydata);
	queryClassPrediction = compet(queryPrediction);
	queryClass = vec2ind(queryClassPrediction);
	tquery = vec2ind(querylabel);
	count2 = sum(queryClass == tquery);
	Accuracy2 = 100*count2/length(querylabel);

	fprintf('the Accuracy of testdata: %.2f\t querydata: %.2f\t',Accuracy,Accuracy2);    
	fprintf('the length of traindata is %d\n',size(traindata,2));
	Record = [Record;Accuracy,Accuracy2,size(traindata,2)];

end