function [Fw,Fw_fake,FW,Falpha,Fb,X_1,norm_fix]= part3_dcdm_STR(Xtrain,Ytrain, tensorSize, initial_w0, STRmax_iter, STRtol_error, dcdm_Max_ite, C, g,eps)

n_mode = numel(tensorSize);

iter = 1;
error(iter) = 10;
Fw=initial_w0;
W = outprod(Fw);

while error(iter) > STRtol_error && iter < STRmax_iter
    old_norm = sqrt(frob(W)); 
    for i = 1 : n_mode
        tem_data = zeros(length(Ytrain), tensorSize(i));
        k_mode_coef = Fw;
        k_mode_coef(i) = [];
        j_iter = 1:n_mode;
        j_iter(i) = [];
        for k = 1:length(Ytrain)
            need_data = Xtrain{k};

            for j = j_iter
                if j < i
                    need_data = tmprod(need_data,k_mode_coef{j},j);
                else
                    need_data = tmprod(need_data,k_mode_coef{j-1},j);
                end
            end
            
            tem_data(k,:) = need_data;
        end
        kmode_coef_tensor = outprod(k_mode_coef);
        norm_fix = frob(kmode_coef_tensor);
        tem_data = [tem_data ones(size(tem_data,1),1)];
        
        X_1=[tem_data;-tem_data];
        C0=C/(norm_fix);
        
       [alpha,w,~] = SSDG(X_1,Ytrain,g,C0,dcdm_Max_ite,eps);
       Falpha{iter,i} = alpha;
       w_new=w';
       Fw_fake{i}= w_new;
       Fw{i} = w_new(1:tensorSize(i));

    end
   
    W = outprod(Fw);
    iter = iter + 1;
    error(iter) = abs(old_norm - sqrt(frob(W)));
end

FW=W;

X_vec = cellfun(@tens2vec, Xtrain, 'UniformOutput', false);
innerP = zeros(length(Ytrain),1);
for h = 1:length(Ytrain)
v = X_vec{h};  
innerP(h) = tens2vec(FW)' * v;  
end
Fb = (sum(Ytrain-innerP))/length(Ytrain);
end

