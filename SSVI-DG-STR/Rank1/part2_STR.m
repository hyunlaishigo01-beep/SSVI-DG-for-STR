function [Sol]= part2_STR(Xtrain,Ytrain,Xtest,Ytest,tensorSize,w0,STRmax_iter,STRtol_error,dcdm_Max_ite,cp_values,eps)


%% ==================================
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
%% =================================

      [~,Fw_fake,FW,Falpha,Fb,T,NORM_FIX] = part3_dcdm_STR(Xtrain,Ytrain, tensorSize, w0, STRmax_iter, STRtol_error, dcdm_Max_ite, cp_values(1), g,eps); 

      [MAE_temp] = part4_perdict(Xtest,Ytest,FW,Fb);    
       MAE(1) = MAE_temp;

     prev_w_fake    = Fw_fake;
     prev_X    = T;
     prev_norm = NORM_FIX;
     prev_d1=sqrt(sum(prev_X.*prev_X,2));

    else 

     prev_w=w0; 
     C1 = cp_values(s-1);
     C2 = cp_values(s);
%% ==================================   
    
   
           if mod(s,2)==0
                   % ==========
                 
                     [~,Fw_fake,FW,~,Fb,T,NORM_FIX]= part6_BACKdvi_STR(Xtrain,Ytrain, tensorSize, prev_w, prev_w_fake,STRmax_iter, STRtol_error, dcdm_Max_ite, C1,C2, g,prev_X,prev_norm,prev_d1,eps);                  
                
                   % ==========
                    prev_w_fake    = Fw_fake;
                    prev_X    = T;
                    prev_norm = NORM_FIX;    
                    prev_d1=sqrt(sum(prev_X.*prev_X,2));
                      
                      
                    [MAE_temp1]= part4_perdict( Xtest,Ytest,FW,Fb )    ;
                       MAE(s) = MAE_temp1;
                      
   
           else
                    % ==========   
 
                     [~,Fw_fake,FW,~,Fb,T,NORM_FIX]= part5_dvi_STR(Xtrain,Ytrain, tensorSize, prev_w, prev_w_fake,STRmax_iter, STRtol_error, dcdm_Max_ite, C1, C2, g,prev_X,prev_norm,prev_d1,eps);                  

                   % ==========
                    prev_w_fake    = Fw_fake;
                    prev_X    = T;
                    prev_norm = NORM_FIX;    
                    prev_d1=sqrt(sum(prev_X.*prev_X,2)); 
                      
                      
                    [MAE_temp1]= part4_perdict( Xtest,Ytest,FW,Fb )    ;
                       MAE(s) = MAE_temp1;
                    
           end
    end
end

      Sol(:,3)=MAE;

end

