clc; clear;

% 处理数据，将数据的初始值、+50、最大值、最大值的位置、此时的样本个数

Methods = {'VE', 'JS', 'KL', 'RAND'};				% 三种分歧度的度量方法
DATA = [];
batch = 1;          % 训练数据的板子

raw = 1; 

for item = 1:4
    method = Methods{item};
    col = 1;
    k=1;
    figure;
    for i = [1:batch-1,batch+1:10]
       
        str = sprintf('load batch%d_%s_100record_%02d.mat Record',batch,method,i);
        eval(str);
        temp = Record;
        [maxvalue, loc] = max(temp(:,1));
        DATA(raw,col:col+4) = [temp(1,1),temp(51,1),maxvalue,loc,temp(loc,3)];
        raw = raw + 1;
        
        subplot(3,3,k);
        plot(1:51,temp(:,1));
        title(['batch1--batch',num2str(i)]);
        k=k+1;
    end
    suptitle([method,'\_batch',num2str(batch)]);        % suptitle放最后，不然会与子图的title重合一点
    figname = ['batch',num2str(batch),'_',method];
    saveas(item, figname);
    raw = raw + 1;
    
end
