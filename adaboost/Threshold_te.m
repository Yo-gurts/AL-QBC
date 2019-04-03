function [L,hits,error_rate] = Threshold_te(model,test_set,samples_weight,true_labels) 
	% 对单个分类器进行测试

	testPrediction = sim(model,test_set);
	% 统计分类情况
	testClassPrediction=compet(testPrediction);
	testClass=vec2ind(testClassPrediction);
	ttest = vec2ind(true_labels);

	hits = sum(ttest == testClass);	% 正确分类的个数

	error = 0;
	for i = 1:length(ttest)
	    if(testClass(i) ~= ttest(i))
	        error = error+samples_weight(i);
	    end
	end

	error_rate = error;			% 错误率与样本权重有关
	L = testClassPrediction; 	% 分类结果