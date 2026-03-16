function G= UpdateG(X,Z,R,lambda3,nClass)
    num_X = size(X{1,1},2); 
    num_view = size(X,2);
    rho = 0.3*num_X;
    Z_tensor = cat(3, Z{:,:});
    R_tensor = cat(3, R{:,:});
    temp_Z = Z_tensor(:);
    temp_R = R_tensor(:);
    
    sX = [num_X, num_X, num_view];
    %twist-version
    [g, objV] = Gshrink(temp_Z + 1/rho*temp_R,(num_X*lambda3)/rho,sX,0,3)   ; %%%%%%%%
    
    G_tensor = reshape(g, sX);
    
    %5 update R
    temp_R = temp_R + rho*(temp_Z - g);
    
    %record the iteration information
    history.objval(iter+1) = objV;
    
    %%
    
    temp_objall=0;
    alpha_g = zeros(num_view,1);
    for v=1:num_view
        temp_objall= temp_objall+sum(dd{1,v});
        WX{v} = X{v}'*W{v};
        alpha_g(v) = trace(WX{v}'*Ls{v}*WX{v});
    end
    Obj_all(iter+1)=sum(alpha_g)+lambda2*objV+lambda*temp_objall;
end