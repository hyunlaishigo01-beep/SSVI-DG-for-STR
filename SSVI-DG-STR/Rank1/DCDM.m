function [alpha1,w1,object1] = DCDM(X,g,C,dcdm_Max_ite)

[m,n] = size(X);
k = 0; 
alpha = zeros(m,1); 
w =C*X'*alpha;
object = 0.5*(w'*w) +C*g'*alpha; 
Stop = 1;
Q = sum(X.^2,2);
order = 1:1:m;

while k<dcdm_Max_ite && Stop>1e-6
    order = order(randperm(length(order)));
    M_max = -Inf;
    M_min = Inf;
    for i = 1:m
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
    object = 0.5*w'*w + C*g'*alpha;
    Stop = M_max - M_min;
    k = k+1;
end
w1 = w;
alpha1 = alpha;
object1 = object;


end
