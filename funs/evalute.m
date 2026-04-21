function [indx,OBJ,F,W,match_error]=evalute(X,L,mmm,lam,lambda,tau)
nClass  = max(L);
para.lambda1 = lam;
para.lambda2 = lambda;
para.lambda3 = tau;
para.iter    = 100;
[indx,OBJ,F,W,match_error]  = feature_selection(X,mmm,nClass,para); 
   


