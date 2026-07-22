function [FUm,FW,Falpha,Fb,vec_Ui_fake,X_1,Binv_half] = part6_BACKdvi_HRSTR(Xtrain, Ytrain, tensorSize, initial_Umatrix0, STRmax_iter, STRtol_error, dcdm_Max_ite, C1, C2, g, Rk, prev_X, prev_B, prev_vec,prev_d1,eps)

    w_norm = sqrt(prev_vec{end}*prev_vec{end}');
    it1=prev_X*prev_vec{end}'/C1;
    item1=(0.5*(C2+C1)*it1);
    item2=(0.5*(C2-C1)*prev_d1*w_norm/C1);
    T1= (item1-item2)>-g-1e-8;
    T2= (item1+item2)<-g+1e-8;  

idx_d2 = find(T1 | T2);      
idx_r2 = find(~(T1 | T2));   

alpha_t2 = zeros(length(idx_d2),1);
alpha_t2(T2(idx_d2)) = 1;    

iter = 1;
error(iter) = 10;
FUm = initial_Umatrix0;

order = numel(FUm):-1:1;
Uminus1 = FUm(order(1:end-1));
KR = kr(Uminus1);
W1 = FUm{1} * KR';

    n_mode = size(tensorSize, 2);

   


%% %% ================= STR Outer Loop ======================
while error(iter) > STRtol_error && iter < STRmax_iter
    old_norm = sqrt(frob(W1)); 

    %% --------- Update each mode ---------
for i = n_mode :-1: 1
   if  i == n_mode && iter == 1 
          
          X_1     = prev_X;
          Binv_half = prev_B;
   
   else  

       j_iter = 1:size(FUm, 2);
       j_iter(i) = [];
       sort_j_iter=sort(j_iter,'descend');
       
     for s =1: (size(FUm,2)-1)  
     Uminusj{1,s}=FUm{1,sort_j_iter(s)}; 
     end   
        
     UMINUSJ=kr(Uminusj); 
     B = UMINUSJ' * UMINUSJ;
     eps_reg = 1e-8;
     B = B + eps_reg * eye(size(B));  
     B_half = sqrtm(B);
     Binv_half = B_half \ eye(size(B_half));
    
%%       
    tem_data = zeros(length(Ytrain), tensorSize(i)*Rk); 
     
%%     
            
            X_i_fold = cellfun(@(t)tens2mat(t,i), Xtrain, 'UniformOutput', false);
            
            for k = 1:length(Ytrain)
            need_data = X_i_fold{k}*UMINUSJ*Binv_half;
            need_datavec = tens2vec(need_data,1);
            tem_data(k,:) = need_datavec';
            end 
%%
        tem_data = [tem_data ones(size(tem_data,1),1)]; 
        X_1=[tem_data;-tem_data];
      

   end

C=C2;

if i== n_mode && iter == 1 && ~isempty(idx_r2)

      [alpha_int, w_int] = SSVIDG(X_1,Ytrain,g,C,dcdm_Max_ite,eps,idx_r2, idx_d2, alpha_t2);

  if all(alpha_int(idx_d2) == alpha_t2)
        
        alpha = alpha_int;
        w     = w_int;
  else


    [alpha_fix, idx, ~, ~, ~] = ...
    DVI_check_SVR7th( ...
        X_1,  g, ...
        T1, T2, w_int);

    R_new = find(idx);
    D_new = find(~idx);  



    if isempty(R_new)
     
        alpha = alpha_fix;
        w = C * X_1(D_new,:)' * alpha(D_new);

    else
        
        [alpha_R, ~, ~] = DCDM( ...
            X_1(R_new,:), g(R_new), C, dcdm_Max_ite);

        alpha = alpha_fix;
        alpha(R_new) = alpha_R;
        w = C * X_1' * alpha;
    end
  end
  
else
   
    [alpha,w,~] = SSDG(X_1,Ytrain,g,C,dcdm_Max_ite,eps);
end
        Falpha{iter,i} = alpha;


        vec_Ui_wave=w';
        vec_Ui_wave_real = vec_Ui_wave(1:tensorSize(i)*Rk);
        vec_Ui_wave_bias = vec_Ui_wave(tensorSize(i)*Rk+1);
        
        vec_Ui_wave=vec_Ui_wave_real';
        mat_Ui_wave=reshape(vec_Ui_wave,tensorSize(i),Rk);
        mat_Ui=mat_Ui_wave*Binv_half;
        FUm{i} = mat_Ui;
        vec_Ui_fake_tmp{i}=w';
end
 

     %% --------- Update W_(1) ---------
    Uminus1 = FUm(order(1:end-1));
    KR = kr(Uminus1);
    W1 = FUm{1} * KR';
    
    iter = iter + 1;
    error(iter) = abs(old_norm - sqrt(frob(W1)));
end
vec_Ui_fake = vec_Ui_fake_tmp;
FW=W1;


X_vec = cellfun(@tens2vec, Xtrain, 'UniformOutput', false); 
innerP = zeros(numel(X_vec),1);
for h = 1:numel(X_vec)
v = X_vec{h};  
innerP(h) = tens2vec(FW)' * v; 
end
    
Fb = sum(Ytrain-innerP)/length(Ytrain);
end
