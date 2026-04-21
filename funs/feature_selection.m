function [scoree,Obj_all,F,W,history] = feature_selection(X,mmm,nClass,para)
    X = X';
    num_view = mmm;
    num_X = size(X{1,1},2); 
    size(X);
    n = size(X{1,1},2);
    %parameter setting
    lambda1 = para.lambda1;
    lambda2 = para.lambda2;
    lambda3 = para.lambda3;
    iter_num = para.iter;
    %initialization
    W = cell(1,num_view);
    R=cell(1,num_view);
    G=cell(1,num_view);
    F = rand(n,1);
    F = F*diag(sqrt(1./(diag(F'*F)+eps)));
    for v = 1:num_view
        R{v} = zeros(num_X,num_X);
        G{v} = zeros(num_X,num_X);
    end
    Z = cell(1,num_view);
    size(Z);
    L = cell(1,num_view);
    ww = cell(1,num_view);
    scoree = cell(1,num_view);
%     size(F)
    for v = 1 : num_view
        W{v} = pinv(X{v}')*F;
        Z{v} = ones(n,n)/n;
    end
    size(W);
    size(Z);
    %F,Z and W 
%     Loss  = [];
    iter  = 0 ;
    Isconverg = zeros(1,num_view);
    epson = 1e-5;
    rho=0.3*num_X;
    max_rho = 10e12; 
    pho_rho =1.3;
    Obj_all=[];

while ~sum(Isconverg)
        iter = iter + 1;
%         fprintf('----processing iter %d--------\n', iter);
        fprintf('Update Z\n');
        for v = 1 : num_view
            Z{v} = UpdateZ(Z{v},X{v},F,G{v},R{v},lambda1,rho);
            L{v} = Laplacian(Z{v});
        end
         fprintf('Update F\n'); 
%         for v = 1 : num_view
            F = UpdateF(X,W,L,F,lambda1,nClass);
            size(F);
%         end
        fprintf('Update W\n');
        for v = 1 : num_view
            W{v} = UpdateW(X{v},F,W{v},lambda2);
        end
        fprintf('Updata G\n');
        Z_tensor = cat(3, Z{:,:});
        R_tensor = cat(3, R{:,:});
        temp_Z = Z_tensor(:);
        size(temp_Z);
        temp_R = R_tensor(:);
        size(temp_R);

        sX = [num_X, num_X, num_view];
        %twist-version
        [g, objV] = Gshrink(temp_Z + 1/rho*temp_R,(num_X*lambda3)/rho,sX,0,3)   ; %%%%%%%%    
        G_tensor = reshape(g, sX);

        %5 update R
        fprintf('Updata R\n');
        temp_R = temp_R + rho*(temp_Z - g);
        size(temp_R);
%         R = temp_R;
        %record the iteration information
        history.objval(iter) = objV;
        temp_objall=0;
        WX = 0;
        alpha_g = 0;
        for v=1:num_view
            temp_objall= temp_objall + norm_2_1(W{v});
            WX = WX + norm_2_1(X{v}'*W{v} - F)+lambda1*trace(F'*L{v}*F);
            norm(X{v} - X{v}*Z{v},'fro')^2;
            alpha_g = alpha_g + norm(X{v} - X{v}*Z{v},'fro')^2;
        end
        Obj_all(iter)=lambda3*objV+temp_objall+WX+alpha_g;
%         fprintf('----Obj_all  %5.5f  \n', Obj_all(iter));
        %% converge condition
        Isconverg = ones(1,num_view);
        history.Z_G(iter+1)=0;
        for v=1:num_view        
            G{v} = G_tensor(:,:,v);
            R_tensor = reshape(temp_R , sX);
            R{v} = R_tensor(:,:,v);
            size(R{v});
            history.norm_Z_G = norm(Z{v}-G{v},inf);

            history.Z_G(iter)=history.Z_G(iter)+history.norm_Z_G;
            if (abs(history.norm_Z_G)>epson)           
                Isconverg(v) = 0;    
%                 fprintf('----norm_Z_G  %5.5f  \n', history.norm_Z_G);
            end
        end
        history.Z_G(iter)=history.Z_G(iter)/num_view;

        if (iter > iter_num)
            Isconverg  = ones(1,num_view);
        end
        rho = min(rho*pho_rho, max_rho);
end
% save F
for v = 1 : num_view
    d = size(X{1,v},1);
    for i=1:d
        ww{v}(i)=norm(W{v}(i,:),2);
    end 
    size(ww{v});
    scoree{v} = ww{v};
%     [~, indx] = sort(w, 'descend');
end
end
% size(scoree)
