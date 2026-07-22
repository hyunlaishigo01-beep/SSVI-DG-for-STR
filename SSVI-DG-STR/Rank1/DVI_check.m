function [alpha_fix, idx, DVI_0, DVI_1, n_violate] = ...
    DVI_check(X, g, DVI_0, DVI_1, w)
tol = 1e-3;

Fi = X * w + g;      

idx0 = DVI_0;
if any(idx0)
    ok0 = (Fi(idx0) >= -tol);
    n_violate_0 = sum(~ok0);
    DVI_0(idx0) = ok0;
else
    n_violate_0 = 0;
end

idx1 = DVI_1;
if any(idx1)
    ok1 = (Fi(idx1) <= tol);
    n_violate_1 = sum(~ok1);
    DVI_1(idx1) = ok1;
else
    n_violate_1 = 0;
end

n_violate = n_violate_0 + n_violate_1;

idx = ~(DVI_0 | DVI_1);

alpha_fix = zeros(size(Fi));
alpha_fix(DVI_1) = 1;

end

