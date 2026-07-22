function [alpha1,w1,object1] = SSDG(X,Y,g,C,tol_num,eps)



check_period = 50;      
full_check_period = 1000; 
dvi_period = 50;       

[m,n] = size(X);
s = 0;
order = 1:1:m;
DVI_0 = zeros(m,1)==1;
DVI_1 = zeros(m,1)==1;
alpha = zeros(m,1);
w = C * X' * alpha;
object = 0.5*(w'*w) + C*g'*alpha;
Stop = 1;
Q = sum(X.^2,2);
X_half=X(1:(m/2),:);  

for k = 1:tol_num
order = order(randperm(length(order)));
M_max = -Inf; 
M_min = Inf;   
    for i = 1:length(order)
        j = order(i);


        G = X(j,:)*w+g(j);
        if alpha(j)==0
            PG = min(G,0);
        elseif alpha(j)==1
            PG = max(G,0);
        else
            PG = G;
        end
        M_max = max(M_max,PG);
        M_min = min(M_min,PG);
        if PG~=0
            alpha_old = alpha(j);
            alpha(j) = min(max(alpha(j)-G/(C*Q(j)),0),1); 
            w = w + C*(alpha(j)-alpha_old)*X(j,:)'; 
        end
    end
    Stop = M_max - M_min;
    if  k == tol_num || Stop < 1e-6
            break;
    end

    if mod(k, check_period) == 0
           eps_func = max(0, abs(X_half*w - Y) - eps);
           object = 0.5*(w'*w) + C*g'*alpha;
           P_object = 0.5*(w'*w) + C*sum(eps_func);
           Gap = P_object + object;
           ABSGap=abs(Gap);
           Stop = M_max - M_min;
        if ABSGap<= 1e-6|| k==tol_num || Stop<1e-6
            break;
        end
    end
    
        if mod(k,full_check_period)==0
            M_max = -Inf;
            M_min = Inf;
            for j = 1:m
                G = X(j,:)*w + g(j);
                if alpha(j)==0
                    PG = min(G,0);
                elseif alpha(j)==1
                    PG = max(G,0);
                else
                    PG = G;
                end
                M_max = max(M_max, PG);
                M_min = min(M_min, PG);
            end
        end
    

    if mod(k, dvi_period) == 0 && Gap > 0
                s=s+1;
        s1=X*w+g;
        s2 = sqrt(Q*Gap);
        DVI_0 = s1-s2>0; 
        DVI_1 = s1+s2<0; 
        order = find(~DVI_1 &~DVI_0)'; 
        alpha(DVI_0) = 0;
        alpha(DVI_1) = 1;
        
        if isempty(order)
        break;
        end
    end
end

w1 = w;
alpha1 = alpha;
object1 = object;

end