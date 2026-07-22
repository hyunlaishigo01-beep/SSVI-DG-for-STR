function [Solution] = part1_final_STR( Xtrain,Ytrain,Xtest,Ytest )

dims = size(Xtrain{1});
tensorSize = dims;      
w0 = arrayfun(@(d) randn(1,d), dims, 'UniformOutput', false);
STRmax_iter = 200;    
STRtol_error = 1e-5; 
dcdm_Max_ite = 5000;


cp_values = 1;      
eps_values = 0.9;
Solution=-ones(length(cp_values)*length(eps_values),3); 
num_eps=length(eps_values);

for i = 1:length(eps_values)
    eps = eps_values(i);
    
    fprintf('Running eps = %.1f ( %d/%d)...\n', ...
            eps, i, length(eps_values)); 

    [Sol]= part2_STR(Xtrain,Ytrain,Xtest,Ytest,tensorSize,w0,STRmax_iter,STRtol_error,dcdm_Max_ite,cp_values,eps);            
  
    Solution((i-1)*length(cp_values)+1:i*length(cp_values),:)=Sol;
            
            
end  
end
        
        
        


