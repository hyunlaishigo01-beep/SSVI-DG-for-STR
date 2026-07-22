function [Fw,Fw_fake,FW,Falpha,Fb,X_1,norm_fix] = part6_BACKdvi_STR(Xtrain,Ytrain, tensorSize, prev_w, prev_w_fake,STRmax_iter, STRtol_error, dcdm_Max_ite, C1, C2, g,prev_X,prev_norm,prev_d1,eps)

w_norm = sqrt(prev_w_fake{end}*prev_w_fake{end}');
it1=prev_X*prev_w_fake{end}'/C1;
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
Fw = prev_w;
W = outprod(Fw);
n_mode = numel(tensorSize);

while error(iter) > STRtol_error && iter < STRmax_iter
  
    
    old_norm = sqrt(frob(W)); 

    for i = n_mode :-1: 1

       if i == n_mode && iter == 1 
        
         X_1     = prev_X;
         norm_fix = prev_norm;

       else  
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
       
       end
         C0=C2/(norm_fix);
if i== n_mode && iter == 1 && ~isempty(idx_r2)
   
     [alpha_int, w_int] =SSVIDG(X_1,Ytrain,g,C0,dcdm_Max_ite,eps,idx_r2, idx_d2, alpha_t2);

              if all(alpha_int(idx_d2) == alpha_t2)
                    alpha = alpha_int;
                    w     = w_int;
              else
                [alpha_fix, idx, ~, ~, ~] = ...
                DVI_check( ...
                    X_1,  g, ...
                    T1, T2, w_int);
            
                R_new = find(idx);
                D_new = find(~idx);
         
                            if isempty(R_new)
                                alpha = alpha_fix;
                                w = C0 * X_1(D_new,:)' * alpha(D_new);
                        
                            else 
                                [alpha_R, ~, ~] = DCDM( ...
                                    X_1(R_new,:), g(R_new), C0, dcdm_Max_ite);
                        
                                alpha = alpha_fix;
                                alpha(R_new) = alpha_R;
                                w = C0 * X_1' * alpha;
                            end
              end 
else
   [alpha,w,~] = SSDG(X_1,Ytrain,g,C0,dcdm_Max_ite,eps);
end   

       Falpha{iter,i} = alpha;
       w_new=w';
       Fw_fake{i}=w_new;
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

