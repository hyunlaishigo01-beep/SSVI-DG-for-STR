clear
clc

load('trnX.mat')
load('trnY.mat')
load('tstX.mat')
load('tstY.mat')

output = cell(5,1);

for cv = 1:5
    
    [Solution] =  part1_final_STR( trnX{cv},trnY{cv},tstX{cv},tstY{cv} );
   
    output{cv} = Solution;

end

save('output','output');
