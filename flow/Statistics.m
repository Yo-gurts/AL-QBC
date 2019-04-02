clc; clear;

Methods = {'VE', 'JS', 'KL', 'RAND'};				% 三种分歧度的度量方法
Percentages = [3 5 10 25 50 75 100];			% 初始训练集占A的百分比
DATA = [];

Raw = 1; 
for item = Methods
    method = item{1};
    col = 1;
    
    for percent = Percentages

        raw = Raw;
        for i = 1:10
            str = sprintf('load %s_%02drecord_%02d.mat Record',method,percent,i);
            eval(str);
            
            temp = Record;
            [maxvalue, loc] = max(temp(:,1));
            DATA(raw,col:col+4) = [temp(1,1),temp(101,1),maxvalue,loc,temp(loc,3)];
            
            raw = raw + 1;
        end
        col = col + 6;

    end
    
    Raw = Raw + 11;
    
end
