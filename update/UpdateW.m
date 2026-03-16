function W = UpdateW(X,F,W,lambda2)

% X is the matrix d*n
% F is the matrix n*c
% W is the matrix d*c 

r = lambda2;
% maxiter = 30;
% XX = X*X';
% Wi = sqrt(sum(W.*W,2)+eps);
% d = 0.5./Wi;
% D = diag(d);
% iter = 0;
% n = size(X,1);
% % I = ones(1,n);
% while iter <=maxiter
%     iter = iter + 1;
%     G=(XX+lambda2*D);
% %     size(G)
%     W=G\(X*F);
%     Wi = sqrt(sum(W.*W,2)+eps);
%     d = 0.5./Wi;
%     D = diag(d);
%     clear Wi
% end
[W,~] = L21R21_inv(X',F,r);
end
