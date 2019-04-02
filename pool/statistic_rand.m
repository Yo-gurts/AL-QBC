clc;clear;
load RAND_100record_10.mat RECORD;

b = [];
for i = 1:7
    a = [];
    for j = 1:10
        temp = RECORD{i,j};
        a(j,1:2) = [temp(1,1),temp(2,1)];
        [mv,ml] = max(temp(:,1));
        samples = temp(ml,3);
        a(j,3:5) = [mv,ml,samples];
    end
    b = [b,zeros(10,1),a];
end