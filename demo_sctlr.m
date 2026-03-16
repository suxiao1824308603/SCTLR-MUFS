    clear;
    clc;
    warning("off")
    %the path of data
    load('3sources.mat')
    X = data;
    num_view = size(X,2);%数据集的视图数
%     for v = 1:num_view
%       X{v} = X{v}';
%     end   
    Y = truelabel{1,1};
%     gt = Y;
    % Y =y;
  
    nFea=zeros(1,num_view);%两个视图的类别数
    sam_num=size(X{1,1},2);%每个视图的样本数
    temp_fea_num=0.02:0.02:0.12;
%     temp_fea_num = 0.12;
    lam_ind=-3:3;
%     lam_ind = -3;
    lam=10.^lam_ind;%lamda参数
    class_num = max(Y);
    MAXiter = 500; % Maximum of iterations for KMeans
    REPlic = 20; % Number of replications for KMeans
    
temp_std=cell(1,num_view);
for v=1:num_view
    temp_std{v}=std(X{1,v},0,2);

    X{1,v}(temp_std{v}==0,:)=[];
    
end
X_Multi=[];
% X_Multi2=[];
% temp_std=cell(1,num_view);
% %  对源数据集进行0均值标准化
%  for v=1:num_view 
%    X_Multi=[X_Multi;X{1,v}];
%    temp_std{v}=std(X{1,v},0,2);%按行求标准差
%    for i=1:size(X{1,v},1)
%        meanvalue_fea=mean(X{1,v}(i,:));%按行求数据集的均值
%        X{1,v}(i,:)=(X{1,v}(i,:)-meanvalue_fea)/temp_std{v}(i,:);%数据集标准化   （均值为0的标准化) 
%    end
%        X_Multi2=[X_Multi2;X{1,v}];  
%        nFea(v)=size(X{v},1);       
%  end
for v = 1:num_view
    X_Multi=[X_Multi;X{1,v}];
end

X_Multi = X_Multi;
% original_ACC_fs = zeros(1,50);
% original_NMI_fs = zeros(1,50);
% original_Purity_fs = zeros(1,50);
% original_Precision_fs = zeros(1,50);
% original_Recall_fs = zeros(1,50);
% original_F_score_fs = zeros(1,50);
% original_ARI_fs = zeros(1,50);

MeanACC_fs_Multi = zeros(7,6);
stdACC_fs_Multi = zeros(7,6);
MeanNMI_fs_Multi = zeros(7,6);
stdNMI_fs_Multi = zeros(7,6);
MeanPurity_fs_Multi = zeros(7,6);
stdPurity_fs_Multi = zeros(7,6);
MeanPrecision_fs_Multi = zeros(7,6);  
stdPrecision_fs_Multi = zeros(7,6);
MeanRecall_fs_Multi = zeros(7,6);
stdRecall_fs_Multi = zeros(7,6);
MeanF_score_fs_Multi = zeros(7,6);
stdF_score_fs_Multi = zeros(7,6);
MeanARI_fs_Multi = zeros(7,6);
stdARI_fs_Multi = zeros(7,6);

temp_ACC = zeros(7,6);
std_ACC = zeros(7,6);
temp_NMI = zeros(7,6);
std_NMI = zeros(7,6);
temp_Purity = zeros(7,6);
std_Purity = zeros(7,6);
temp_Precision = zeros(7,6);
std_Precision = zeros(7,6);
temp_Recall = zeros(7,6);
std_Recall = zeros(7,6);
temp_F_score = zeros(7,6);
std_F_score = zeros(7,6);
temp_ARI = zeros(7,6);
std_ARI = zeros(7,6);

temp2_ACC = zeros(7,6);
std2_ACC = zeros(7,6);
temp2_NMI = zeros(7,6);
std2_NMI = zeros(7,6);
temp2_Purity = zeros(7,6);
std2_Purity = zeros(7,6);
temp2_Precision = zeros(7,6);
std2_Precision = zeros(7,6);
temp2_Recall = zeros(7,6);
std2_Recall = zeros(7,6);
temp2_F_score = zeros(7,6);
std2_F_score = zeros(7,6);
temp2_ARI = zeros(7,6);
std2_ARI = zeros(7,6);

final_ACC = zeros(2,6);
final_NMI = zeros(2,6);
final_Purity = zeros(2,6);
final_Precision = zeros(2,6);
final_Recall = zeros(2,6);
final_F_score = zeros(2,6);
final_ARI = zeros(2,6);

pvalue_ACC = zeros(2,1);
pvalue_NMI = zeros(2,1);
pvalue_Purity = zeros(2,1);
pvalue_Precision = zeros(2,1);
pvalue_Recall = zeros(2,1);
pvalue_F_score = zeros(2,1);
pvalue_ARI = zeros(2,1);
% for kk = 1:50
%     class_num=max(gt);
%     original_idx = kmeans(X_Multi',class_num,'maxiter',500,'replicates',20,'EmptyAction','singleton');
%     original_result = EvaluationMetrics(gt, original_idx);       
%     original_ACC_fs(kk)=original_result(1,1);
%     original_NMI_fs(kk)=original_result(1,2);
%     original_Purity_fs(kk) = original_result(1,3);
%     original_Precision_fs(kk) = original_result(1,4);
%     original_Recall_fs(kk) = original_result(1,5);
%     original_F_score_fs(kk) = original_result(1,6);
%     original_ARI_fs(kk) = original_result(1,7);
% end
% %% Main
% a = [mean(original_ACC_fs),std(original_ACC_fs)]
% b = [mean(original_NMI_fs),std(original_NMI_fs)]
% c = [mean(original_Purity_fs),std(original_Purity_fs)]
% d = [mean(original_Precision_fs),std(original_Precision_fs)]
% e = [mean(original_Recall_fs),std(original_Recall_fs)]
% f = [mean(original_F_score_fs),std(original_F_score_fs)]
% g = [mean(original_ARI_fs),std(original_ARI_fs)]

d_fea = size(X_Multi,1);
num = ceil(temp_fea_num * d_fea);%最终聚类选择的样本数
YYY = 0;
% %% 记录代码
% % 初始化存储变量（根据参数数量定义，7×7×7=343组参数）
% param_records = struct();  % 结构体数组，每个元素存储一组参数的信息
% record_idx = 0;  % 记录索引
% target_folder = './results/feature_coefficients/'; 
%%
for p_count = 1:1
    for pn = 1:length(lam(5))
        for pk = 1:length(lam(5))
           for pm=1:length(lam(5))
               YYY = YYY + 1;
               lambda1=lam(pn);
               lambda2=lam(pk);
               lambda3=lam(pm);
               nnn = size(X);
               tic;
               [score,OBJ,F,W,match_error] =evalute(X',Y,num_view,lambda1,lambda2,lambda3);
               toc;
               sscore = [];
               for v=1:num_view
                   temp_score = score{1,v};
                   sscore=[sscore,temp_score];  
               end
               [~, index] = sort(sscore,'descend');
               Fea_fs = X_Multi(index,:);
               ACC_fs = zeros(1,50);
               NMI_fs = zeros(1,50);
               Purity_fs = zeros(1,50);
               Precision_fs = zeros(1,50);
               Recall_fs = zeros(1,50);
               F_score_fs = zeros(1,50);
               ARI_fs = zeros(1,50);
               for j=1:length(num)
                   feature_num=num(j);   
                   newfea = Fea_fs(1:feature_num,:);
                   result = [];
                   label = [];
                   for op = 1:50
                       label = kmeans(newfea',class_num,'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton');
                       result = EvaluationMetrics(Y,label);
                       ACC_fs(1,op) = result(1,1);
                       NMI_fs(1,op) = result(1,2);
                       Purity_fs(1,op) = result(1,3);
                       Precision_fs(1,op) = result(1,4);
                       Recall_fs(1,op) = result(1,5);
                       F_score_fs(1,op) = result(1,6);
                       ARI_fs(1,op) = result(1,7);
                  end
                       MeanACC_fs_Multi(pm,j) = mean(ACC_fs);%按列求均值
                       stdACC_fs_Multi(pm,j) = std(ACC_fs);%按列求均值
                       MeanNMI_fs_Multi(pm,j) = mean(NMI_fs);
                       stdNMI_fs_Multi(pm,j) = std(NMI_fs);%按列求均值
                       MeanPurity_fs_Multi(pm,j) = mean(Purity_fs);%按列求均值
                       stdPurity_fs_Multi(pm,j) = std(Purity_fs);%按列求均值
                       MeanPrecision_fs_Multi(pm,j) = mean(Precision_fs);
                       stdPrecision_fs_Multi(pm,j) = std(Precision_fs);%按列求均值
                       MeanRecall_fs_Multi(pm,j) = mean(Recall_fs);%按列求均值
                       stdRecall_fs_Multi(pm,j) = std(Recall_fs);%按列求均值
                       MeanF_score_fs_Multi(pm,j) = mean(F_score_fs);
                       stdF_score_fs_Multi(pm,j) = std(F_score_fs);%按列求均值
                       MeanARI_fs_Multi(pm,j) = mean(ARI_fs);%按列求均值
                       stdARI_fs_Multi(pm,j) = std(ARI_fs);%按列求均值
               end 
%                record_idx = record_idx + 1;
%                param_records(record_idx).lambda1 = lambda1;  % 当前lambda1
%                param_records(record_idx).lambda2 = lambda2;  % 当前lambda2
%                param_records(record_idx).lambda3 = lambda3;  % 当前lambda3
%                param_records(record_idx).mean_acc = mean(ACC_fs);  % 当前参数的ACC均值（或其他核心指标）
%                param_records(record_idx).score = score;  % 当前参数的分视图系数
%                param_records(record_idx).sscore = sscore;  % 当前参数的合并系数
           end
           [temp1,temp_max1] = max(MeanACC_fs_Multi);
           temp1 = temp1';
           temp_ACC(pk,:) = temp1;
           for i = 1:size(stdACC_fs_Multi,2)
                std_ACC(pk,i) = stdACC_fs_Multi(temp_max1(i),i);
           end
           [temp2,temp_max2] = max(MeanNMI_fs_Multi); 
           temp2 = temp2';
           temp_NMI(pk,:) = temp2;
           for i = 1:size(stdNMI_fs_Multi,2)
                std_NMI(pk,i) = stdNMI_fs_Multi(temp_max2(i),i);
           end
           [temp3,temp_max3] = max(MeanPurity_fs_Multi); 
           temp3 = temp3';
           temp_Purity(pk,:) = temp3;
           for i = 1:size(stdPurity_fs_Multi,2)
                std_Purity(pk,i) = stdPurity_fs_Multi(temp_max3(i),i);
           end
           [temp4,temp_max4] = max(MeanPrecision_fs_Multi); 
           temp4 = temp4';
           temp_Precision(pk,:) = temp4;
           for i = 1:size(stdPrecision_fs_Multi,2)
                std_Precision(pk,i) = stdPrecision_fs_Multi(temp_max4(i),i);
           end
           [temp5,temp_max5] = max(MeanRecall_fs_Multi); 
           temp5 = temp5';
           temp_Recall(pk,:) = temp5;
           for i = 1:size(stdRecall_fs_Multi,2)
                std_Recall(pk,i) = stdRecall_fs_Multi(temp_max5(i),i);
           end
           [temp6,temp_max6] = max(MeanF_score_fs_Multi); 
           temp6 = temp6';
           temp_F_score(pk,:) = temp6;
           for i = 1:size(stdF_score_fs_Multi,2)
                std_F_score(pk,i) = stdF_score_fs_Multi(temp_max6(i),i);
           end
           [temp7,temp_max7] = max(MeanARI_fs_Multi); 
           temp7 = temp7';
           temp_ARI(pk,:) = temp7;
           for i = 1:size(stdARI_fs_Multi,2)
                std_ARI(pk,i) = stdARI_fs_Multi(temp_max7(i),i);
           end
        end
           [temp1,temp_max1] = max(temp_ACC); 
           temp1 = temp1';
           temp2_ACC(pn,:) = temp1;
           for i = 1:size(std_ACC,2)
                std2_ACC(pn,i) = std_ACC(temp_max1(i),i);
           end
           [temp2,temp_max2] = max(temp_NMI); 
           temp2 = temp2';
           temp2_NMI(pn,:) = temp2;
           for i = 1:size(std_NMI,2)
                std2_NMI(pn,i) = std_NMI(temp_max2(i),i);
           end
           [temp3,temp_max3] = max(temp_Purity); 
           temp3 = temp3';
           temp2_Purity(pn,:) = temp3;
           for i = 1:size(std_Purity,2)
                std2_Purity(pn,i) = std_Purity(temp_max3(i),i);
           end
           [temp4,temp_max4] = max(temp_Precision); 
           temp4 = temp4';
           temp2_Precision(pn,:) = temp4;
           for i = 1:size(std_Precision,2)
                std2_Precision(pn,i) = std_Precision(temp_max4(i),i);
           end
           [temp5,temp_max5] = max(temp_Recall); 
           temp5 = temp5';
           temp2_Recall(pn,:) = temp5;
           for i = 1:size(std_Recall,2)
                std2_Recall(pn,i) = std_Recall(temp_max5(i),i);
           end
           [temp6,temp_max6] = max(temp_F_score); 
           temp6 = temp6';
           temp2_F_score(pn,:) = temp6;
           for i = 1:size(std_F_score,2)
                std2_F_score(pn,i) = std_F_score(temp_max6(i),i);
           end
           [temp7,temp_max7] = max(temp_ARI); 
           temp7 = temp7';
           temp2_ARI(pn,:) = temp7;
           for i = 1:size(std_ARI,2)
                std2_ARI(pn,i) = std_ARI(temp_max7(i),i);
           end
    end
    [temp1,temp_max1] = max(temp2_ACC); 
    final_ACC(1,:) = temp1;
    for i = 1:size(std2_ACC,2)
        final_ACC(2,i) = std2_ACC(temp_max1(i),i);
    end
    [temp2,temp_max2] = max(temp2_NMI); 
    final_NMI(1,:) = temp2;
    for i = 1:size(std2_NMI,2)
        final_NMI(2,i) = std2_NMI(temp_max2(i),i);
    end
    [temp3,temp_max3] = max(temp2_Purity); 
    final_Purity(1,:) = temp3;
    for i = 1:size(std2_Purity,2)
        final_Purity(2,i) = std2_Purity(temp_max3(i),i);
    end
    [temp4,temp_max4] = max(temp2_Precision); 
    final_Precision(1,:) = temp4;
    for i = 1:size(std2_Precision,2)
        final_Precision(2,i) = std2_Precision(temp_max4(i),i);
    end
    [temp5,temp_max5] = max(temp2_Recall); 
    final_Recall(1,:) = temp5;
    for i = 1:size(std2_Recall,2)
        final_Recall(2,i) = std2_Recall(temp_max5(i),i);
    end
    [temp6,temp_max6] = max(temp2_F_score); 
    final_F_score(1,:) = temp6;
    for i = 1:size(std2_F_score,2)
        final_F_score(2,i) = std2_F_score(temp_max6(i),i);
    end
    [temp7,temp_max7] = max(temp2_ARI); 
    final_ARI(1,:) = temp7;
    for i = 1:size(std2_ARI,2)
        final_ARI(2,i) = std2_ARI(temp_max7(i),i);
    end
    [pvalue_temp1,pvalue_index1]=max(final_ACC(1,2:end));
    pvalue_ACC(1,p_count)=pvalue_temp1;
    pvalue_ACC(2,p_count)=final_ACC(2,pvalue_index1);
    
    [pvalue_temp2,pvalue_index2]=max(final_NMI(1,2:end));
    pvalue_NMI(1,p_count)=pvalue_temp2;
    pvalue_NMI(2,p_count)=final_NMI(2,pvalue_index2);
    
    [pvalue_temp3,pvalue_index3]=max(final_Purity(1,2:end));
    pvalue_Purity(1,p_count)=pvalue_temp3;
    pvalue_Purity(2,p_count)=final_Purity(2,pvalue_index3);
    
    [pvalue_temp4,pvalue_index4]=max(final_Precision(1,2:end));
    pvalue_Precision(1,p_count)=pvalue_temp4;
    pvalue_Precision(2,p_count)=final_Precision(2,pvalue_index4);
    
    [pvalue_temp5,pvalue_index5]=max(final_Recall(1,2:end));
    pvalue_Recall(1,p_count)=pvalue_temp5;
    pvalue_Recall(2,p_count)=final_Recall(2,pvalue_index5);
    
    [pvalue_temp6,pvalue_index6]=max(final_F_score(1,2:end));
    pvalue_F_score(1,p_count)=pvalue_temp6;
    pvalue_F_score(2,p_count)=final_F_score(2,pvalue_index6);
    
    [pvalue_temp7,pvalue_index7]=max(final_ARI(1,2:end));
    pvalue_ARI(1,p_count)=pvalue_temp7;
    pvalue_ARI(2,p_count)=final_ARI(2,pvalue_index7);
end

% [max_acc, max_acc_idx] = max([param_records.mean_acc]);  % 找到ACC最高的记录索引
% best_params = param_records(max_acc_idx);  % 最优参数组合的所有信息
% 
% % 2. 提取并保存最优参数对应的系数
% best_score = best_params.score;  % 最优分视图系数矩阵
% best_sscore = best_params.sscore;  % 最优合并系数向量
% 
% % 3. 保存到mat文件（可自定义文件名）
% save([target_folder,'best_feature_coefficients.mat'], 'best_score', 'best_sscore', 'best_params');
% 
% save('best_feature_coefficients.mat', 'best_score', 'best_sscore', 'best_params');
% MMML = tsne(newfea');
% % gscatter(MMML(:,1),MMML(:,2),label);
% h = gscatter(MMML(:,1), MMML(:,2), label, 'cmkbrg', 'xo*^sdpvH+<>',5);
% legend('Location', 'northeast', ...  % 固定位置为右上角
%        'FontName', 'SimHei', ...      % 支持中文
%        'FontSize', 10, ...            % 字体大小
%        'Box', 'on');   

