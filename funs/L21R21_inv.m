function [X, obj]=L21R21_inv(A, Y, r)
%% 21-norm loss with 21-norm regularization

%% Problem
%
% A =  X^T;
% X = W;
% Y = F;
% r = lambda3/lambda1; 

%  min_X  || A X - Y||_21 + r * ||X||_21

% Ref: Feiping Nie, Heng Huang, Xiao Cai, Chris Ding. 
% Efficient and Robust Feature Selection via Joint L21-Norms Minimization.  
% Advances in Neural Information Processing Systems 23 (NIPS), 2010.



NIter = 30;
[m n] = size(A);
% if nargin < 4
    d = ones(n,1);
    d1 = ones(m,1);
% else
%     Xi = sqrt(sum(X0.*X0,2));
%     d = 0.5./*Xi;
%     AX = A*X0-Y;
%     Xi1 = sqrt(sum(AX.*AX,2)+eps);
%     d1 = 0.5./Xi1;
% end
bStop = 0;
iter = 0;
   while ~bStop
       iter = iter + 1;
        D = spdiags(d,0,n,n);
        D1 = spdiags(d1,0,m,m);
        AD = A'*D1;
        X = (AD*A+r*D)\(AD*Y);

        Xi = sqrt(sum(X.*X,2)+eps);
        d = 0.5./Xi;

        AX = A*X-Y;
        Xi1 = sqrt(sum(AX.*AX,2)+eps);
        d1 = 0.5./Xi1;

        obj(iter) = sum(Xi1) + r*sum(Xi);
        if iter>1
            if iter > 30
                    bStop = 1;
%                 else
%                     %minDiffValue
%                     if abs(obj(iter-1) - obj(iter))/obj(iter) <= 1e-3
%                             bStop = 1;
%                     end
            end
        end
   end
end

