function [Solution] = part1_final_HRSTR(  Xtrain,Ytrain,Xtest,Ytest  )
dims = size(Xtrain{1});
tensorSize = dims;      
STRmax_iter = 200;    
STRtol_error = 1e-5; 
dcdm_Max_ite = 5000;

%%
rank_values = [5];%%%%%%
cp_values =10.^[-1];        
eps_values = [0.9];
num_cp  = length(cp_values);
num_eps = length(eps_values);
num_each_r = num_cp * num_eps;

Solution=-ones(num_cp*num_eps*length(rank_values),4); 

for r = 1:length(rank_values)
          Rk=rank_values(r);
        
    fprintf('Running Rk = %.1f ( %d/%d )...\n', ...
            Rk, r, length(rank_values)); 
          Umatrix0 = cell(1, length(tensorSize));    
          for k=1:length(tensorSize)
          Umatrix0{k} = randn(tensorSize(k), Rk); 
          end

for i = 1:num_eps
    eps = eps_values(i);
    
    fprintf('Running eps = %.1f ( %d/%d )...\n', ...
            eps, i, num_eps); 
       
          
             
            [Sol]= part2_HRSTR(Xtrain,Ytrain,Xtest,Ytest,tensorSize,Umatrix0,STRmax_iter,STRtol_error,dcdm_Max_ite,cp_values,eps,Rk);
           


            row_start = (r-1)*num_each_r + (i-1)*num_cp + 1;
            row_end   = row_start + num_cp - 1;

            Solution(row_start:row_end,1:3) = Sol;
                  
            
end  

 Solution((r-1)*num_each_r+1:r*num_each_r,4) = Rk;

end
 
end




