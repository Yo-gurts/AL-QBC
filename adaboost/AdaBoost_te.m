function [L, hits] = AdaBoost_te(adaboost_model,te_func_handle,test_set,true_labels)

hypothesis_n = length(adaboost_model.weights);  	% 集成分类器中分类器的个数

sample_n = size(test_set,2);  						% 样本数

class_n = size(true_labels,1);  % class_n = length(unique(true_labels));
temp_L = zeros(class_n,sample_n,hypothesis_n);   % likelihoods for each weak classifier  
  
% for each weak classifier, likelihoods of test samples are collected  
for i=1:hypothesis_n  

	% temp_L : 分类结果， hits： 正确率， error_rate: 错误率（与样本权重有关，不过这里样本权重均为1，ones(sample_n,1))
    [temp_L(:,:,i), hits, error_rate] = te_func_handle(adaboost_model.parameters{i},test_set,ones(sample_n,1),true_labels);  
    
    % 把每一个分类器的 分类结果 * 分类器的权重 最后把各个分类器的结果相加，得到最终的分类结果
    temp_L(:,:,i) = temp_L(:,:,i)*adaboost_model.weights(i);  

end  

L = sum(temp_L,3);  	% 最终的分类结果为 L

hits = sum(vec2ind(L) == vec2ind(true_labels));	% 集成分类器正确分类的个数。