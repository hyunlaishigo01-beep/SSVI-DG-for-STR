function [FUm,FW,Falpha,Fb,vec_Ui_fake,X_1,Binv_half] = part3_dcdm_HRSTR(Xtrain, Ytrain, tensorSize, initial_Umatrix0, STRmax_iter, STRtol_error, dcdm_Max_ite,  C, g, Rk,eps)
iter = 1;
error(iter) = 10;
FUm = initial_Umatrix0;
order = numel(FUm):-1:1;
Uminus1 = FUm(order(1:end-1));
KR = kr(Uminus1);
W1 = FUm{1} * KR';

n_mode = size(tensorSize, 2);

while error(iter) > STRtol_error && iter < STRmax_iter
    old_norm = sqrt(frob(W1)); 

    for i = 1 : n_mode
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
 
    tem_data = zeros(length(Ytrain), tensorSize(i)*Rk); 
 
            
            X_i_fold = cellfun(@(t)tens2mat(t,i), Xtrain, 'UniformOutput', false);
            
            for k = 1:length(Ytrain)
            need_data = X_i_fold{k}*UMINUSJ*Binv_half;
            need_datavec = tens2vec(need_data,1);
            tem_data(k,:) = need_datavec';
            end 
        tem_data = [tem_data ones(size(tem_data,1),1)]; 
        X_1=[tem_data;-tem_data];

        assert(length(g) == size(X_1,1), 'g dimension mismatch');


        [alpha,w,~] = SSDG(X_1,Ytrain,g,C,dcdm_Max_ite,eps);
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
