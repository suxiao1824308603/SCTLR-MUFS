function F = UpdateF(X,W,L,F,lambda1,nClass)
    [d,n] = size(X{1,1});
    num_view = size(X,2);
    [a,b] = size(L{1});
    LL = zeros(a,b);
    alpha = 1/lambda1;
    gamma = 1e8;
    iter = 0;
    bStop = 0;
    objValue = [];
    Q = cell(1,num_view);
    temp = cell(1,num_view);
    for v = 1:num_view
        Q{v} = zeros(n,n);
        LL = LL + L{v};
    end
    while ~bStop
        iter = iter + 1;
        for v = 1:num_view
            temp{v} = X{v}'*W{v}-F;
        end
        for v =1:num_view
            q = 0.5./(sqrt(sum(temp{v}.*temp{v},2)+eps));
            Q{v} = diag(q);
        end
        FF = F'*F;
        QQ = 0;
        for i = 1:num_view
            QQ = QQ + Q{i};
        end
        TEMPP = 0;
        for i = 1:num_view
            TEMPP = TEMPP + Q{i}*X{i}'*W{i};
        end
        F = F.*(alpha*QQ*F+2*gamma*F)./max(LL*F+2*gamma*F*FF+alpha*TEMPP, 1e-10);
        k = find(F == Inf);
        if ~isempty(k)
             F(k) = 0;
        end
        
%         FF = F'*F;
        F = F*diag(sqrt(1./(diag(F'*F)+eps))); %normalize  
        
        objValue(iter) = calMainObjFuncValue(X, L, F, W, alpha, gamma,num_view,nClass);
        if iter > 1
            %maxIterNum
            if iter > 30
                bStop = 1;
            else
                %minDiffValue
                if abs(objValue(iter-1) - objValue(iter))/objValue(iter) <= 1e-3
                        bStop = 1;
                end
            end
        end
        
    end
%     save F
end

function final_value = calMainObjFuncValue(X, L, F, W, alpha, gamma,num_view,nclass)
% Calculate the main subjective function value.
    final_value = 0;
    for v = 1:num_view
    [d,c] = size(W{v});
    objValue = trace(F'*L{v}*F);
    objValue = objValue + alpha*norm_2_1(X{v}'*W{v}-F);
    objValue = objValue + gamma*norm(F'*F-eye(c),'fro')^2;%默认就是2范数
    final_value = objValue + final_value;
    end
end

