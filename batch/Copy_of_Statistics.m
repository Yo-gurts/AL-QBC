clc; clear;

% 处理数据，将数据的初始值、+50、最大值、最大值的位置、此时的样本个数

Methods = {'VE', 'JS', 'KL', 'RAND'};				% 三种分歧度的度量方法
DATA = [];
batch = 1;          % 训练数据的板子

raw = 1; 
    figure;
for item = 1:4
    method = Methods{item};
    col = 1;
    k=1;
    temp =  {};
    for i = [1:batch-1,batch+1:10]
       
        str = sprintf('load batch%d_%s_100record_%02d.mat Record',batch,method,i);
        eval(str);
        temp{i} = Record;
        [maxvalue, loc] = max(temp{i}(:,1));
        DATA(raw,col:col+4) = [temp{i}(1,1),temp{i}(51,1),maxvalue,loc,temp{i}(loc,3)];
        raw = raw + 1;
       
    end
        subplot(3,3,k); 
        plot(1:51,temp{1,1}(:,1)); hold on;
        plot(1:51,temp{2}(:,1)); hold on;
        plot(1:51,temp{3}(:,1)); hold on;
        plot(1:51,temp{4}(:,1)); hold on;
        title(['batch1--batch',num2str(i)]);
        k=k+1;
    raw = raw + 1;
    
end
    legend('VE', 'JS', 'KL','RAND');
    suptitle([method,'\_batch',num2str(batch)]);        % suptitle放最后，不然会与子图的title重合一点
    figname = ['batch',num2str(batch),'_',method];
    saveas(1, figname);