function result = evalute_num(X,L,feature_num,idx)
lab_val=unique(L);
mm=length(lab_val);
dat=[];
Lab=[];
for i=1:mm
      dat=[dat;X(L==lab_val(i),:)];
      Lab=[Lab;i*ones(sum(L==lab_val(i)),1)];
end
X=dat;
L=Lab;

indx=idx(1:feature_num);
newfea = NormalizeFea(X(:,indx));
data=[newfea,L];
% rec_acc_raw=crossvalidate([X,L],10,'KNN',1);
% disp(['raw recognition accuracy: ',num2str(rec_acc_raw)]);

rec_acc_fs=crossvalidate(data,10,'KNN',1);
fprintf('num of feature: %5i, accuracy: %5.3f\n', feature_num, rec_acc_fs);

MAXiter = 500;
REPlic = 20;

for i=1:20
    label = kmeans(newfea,mm,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
%     [accu(i),  MIhat(i)] = xinBestMap(L,label);
    result = ClusteringMeasure(L,label);
end
% rec_clu=mean(MIhat);
% rec_acc_clu=mean(accu);
% fprintf('num of feature: %5i, NMI: %5.3f\n', feature_num, rec_clu);
% fprintf('num of feature: %5i, cluster: %5.3f\n', feature_num, rec_acc_clu);
% 
% rebunduncy = evalFSRedncy(newfea,indx,feature_num);
% fprintf('num of feature: %5i, redundancy: %5.3f\n', feature_num, rebunduncy);
