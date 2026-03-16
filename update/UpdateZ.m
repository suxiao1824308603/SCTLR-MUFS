function Z  = UpdateZ(Z,X,F,G,R,lambda1,rho)
    [d,n] = size(X);
    lapha = 1e-12;
%     iter_num = 20;
    P = zeros(n,n);

    X = [X',lapha*ones(n,n)]';
    
    %compute the P
    for i = 1:n        
        tmp = repmat(F(i,:),n,1)-F;
        P(:,i) = sum(tmp.*tmp,2);
    end
    
%     for iter = 1:iter_num
        %update the j-th row of Z
        for j = 1:n
            z = Z(j,:);%化成每一行的形式
            p = P(:,j);
            x = X(:,j);
            g = G(:,j);
            r = R(:,j);
            
            X_1 = X-(X*Z-x*z);
            v = (X_1'*x)/(x'*x);
            
            size(abs(v));
            size(G);
            size(R);
            size(p);
            rho;
            tt = v+rho/2*g-0.5*r;
            temp = abs(v+rho/2*g-0.5*r)-lambda1/4*p;
            Z(j,:) = (sign(tt).*temp.*(temp>0))/(1+rho/2);
            Z(j,j)=0;
        end    
%     end
end
