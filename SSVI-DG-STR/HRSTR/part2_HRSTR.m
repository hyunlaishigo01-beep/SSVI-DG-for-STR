function [Sol]= part2_HRSTR(Xtrain,Ytrain,Xtest,Ytest,tensorSize,Umatrix0,STRmax_iter,STRtol_error,dcdm_Max_ite,cp_values,eps,Rk)

Sol = -ones(length(cp_values),3);
Sol(:,1)=eps;
Sol(:,2)=cp_values;
n_C=length(cp_values);
g=[eps*ones(length(Ytrain),1)-Ytrain;eps*ones(length(Ytrain),1)+Ytrain];

MAE=zeros(n_C,1);

for s = 1:n_C

    fprintf('Running C = %.4f (%d/%d)\n', ...
        cp_values(s), s, n_C);

    if s == 1



%% =================  C =================
 [~,FW,Falpha,Fb,vec_Ui_fake,T,Binvhalf] = part3_dcdm_HRSTR(Xtrain, Ytrain, tensorSize, Umatrix0, STRmax_iter, STRtol_error, dcdm_Max_ite, cp_values(1), g, Rk,eps);
 
 [MAE_temp] = part4_perdict(Xtest, Ytest, FW, Fb);
    MAE(1) = MAE_temp;

    %current_Umatrix0  = FUm;
    prev_X    = T;
    prev_vec = vec_Ui_fake;
    prev_B = Binvhalf;
    prev_d1 = sqrt(sum(prev_X.*prev_X,2));

 else 

     current_Umatrix0 = Umatrix0;
     C1 = cp_values(s-1);
     C2 = cp_values(s);
 
%% ==================================   
    if mod(s,2)==0

        
     [~,FW,~,Fb,vec_Ui_fake,T,Binvhalf] = part6_BACKdvi_HRSTR(Xtrain, Ytrain, tensorSize, current_Umatrix0, STRmax_iter, STRtol_error, dcdm_Max_ite, C1, C2, g, Rk,prev_X,prev_B,prev_vec,prev_d1,eps);
       
    %current_Umatrix0  = FUm;
    prev_X    = T;
    prev_vec = vec_Ui_fake;
    prev_B = Binvhalf;
    prev_d1 = sqrt(sum(prev_X.*prev_X,2)); 
      
      
    [MAE_temp1]= part4_perdict( Xtest,Ytest,FW,Fb )    ;
    MAE(s) = MAE_temp1;
      
    else

   
     [~,FW,~,Fb,vec_Ui_fake,T,Binvhalf] = part5_dvi_HRSTR(Xtrain, Ytrain, tensorSize, current_Umatrix0, STRmax_iter, STRtol_error, dcdm_Max_ite, C1, C2, g, Rk,prev_X,prev_B,prev_vec,prev_d1,eps);
       
    %current_Umatrix0  = FUm;
    prev_X    = T;
    prev_vec = vec_Ui_fake;
    prev_B = Binvhalf;
    prev_d1 = sqrt(sum(prev_X.*prev_X,2));  
      
      
    [MAE_temp1]= part4_perdict( Xtest,Ytest,FW,Fb )    ;
       MAE(s) = MAE_temp1;
       

    end
    end
end


      Sol(:,3)=MAE;



end
    


