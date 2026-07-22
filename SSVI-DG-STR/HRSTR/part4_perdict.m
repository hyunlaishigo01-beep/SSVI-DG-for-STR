function  [MAE]= part4_perdict( Xtest,Ytest,FW,Fb)


                        tstX_vec = cellfun(@tens2vec, Xtest, 'UniformOutput', false);
                        innerProds = zeros(numel(tstX_vec),1);
                        for k = 1:numel(tstX_vec)
                        v = tstX_vec{k}(:);  
                        innerProds(k) = tens2vec(FW)' * v;  
                        end
                        PY = innerProds + Fb;

                        MAE=mean(abs(PY-Ytest));
                           

end

