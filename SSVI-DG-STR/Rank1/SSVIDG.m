function [alpha1,w1,object1] = SSVIDG(X,Y,g,C,tol_num,eps,idx_r, idx_d, alpha_t)

check_period = 50;    
full_check_period = 1000; 
dvi_period = 50;       

[m, n] = size(X);
m_r = length(idx_r);
m_d = length(idx_d);
d = 0;
DVI_0_r = zeros(m_r,1)==1;
DVI_1_r = zeros(m_r,1)==1;

alpha = zeros(m,1);
for f = 1:m_d
    alpha(idx_d(f)) = alpha_t(f);
end

w = C * (X' * alpha);
object = 0.5*(w'*w) +C*g'*alpha; 
order = 1:1:m_r;

Stop = 1;
k = 0;
Q = sum(X.^2,2);
X_half=X(1:(m/2),:);


for k=1:tol_num
    order = order(randperm(length(order)));
    M_max = -Inf;
    M_min = Inf;

    for l = 1:length(order)
        j = idx_r(order(l));
        G = X(j,:) * w + g(j);
       
        if alpha(j) == 0
            PG= min(G, 0);
        
        elseif alpha(j) == 1
            PG = max(G, 0);
        
        else
            PG = G;
        end
        M_max = max(M_max,PG);
        M_min = min(M_min,PG);
        if  PG ~= 0
            alpha_old = alpha(j);
            alpha(j) = min(max(alpha(j) - G / (C*Q(j)), 0), 1);
            w = w + C*(alpha(j)-alpha_old)*X(j,:)';
        end
    end

    for s = 1:m_d
        i = idx_d(s);
        G = X(i,:) * w + g(i);

        if alpha(i) <= 0
            PG = min(G, 0);
        elseif alpha(i) >= 1
            PG = max(G, 0);
        else
            PG = G;
        end

        M_max = max(M_max, PG);
        M_min = min(M_min, PG);
    end

    Stop = M_max - M_min;
    if  k == tol_num || Stop < 1e-6
            break;
    end    

if mod(k,1000)==0
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


%%
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
    if mod(k, dvi_period) == 0 && Gap > 0 
        d=d+1;
        s1=X*w+g;
        s2 = sqrt(Q*Gap);
        DVI_0_r = (s1(idx_r) - s2(idx_r) > 0);
        DVI_1_r = (s1(idx_r) + s2(idx_r) < 0);
        order = find(~DVI_0_r & ~DVI_1_r);
        alpha(idx_r(DVI_0_r)) = 0;
        alpha(idx_r(DVI_1_r)) = 1;
        
        if isempty(order)
        break;
        end
        
    end 
    k = k+1;
end
w1 = w;
alpha1 = alpha;
object1 = object;
end
