clear;
clc;
warning('off');

%% ========== 用户可配置参数 ==========
% 超参数（你可以随时修改这些值）
lambda1 = 0.1;      % 正则化参数1
lambda2 = 1;      % 正则化参数2
lambda3 = 0.1;      % 正则化参数3

% 特征选择比例（按总特征数的比例）
temp_fea_num = 0.02;  % 可改为 [0.1] 或其他固定值
% 0.02:0.02:0.12
% KMeans 设置
MAXiter = 500;    % 最大迭代次数
REPlic  = 20;     % 重复次数（每次随机初始化）

% 数据路径
load('NGs.mat');  % 确保 X 和 gt 已定义

%% ========== 数据预处理 ==========
% Y = gt;   % 真实标签
X = data;
Y = truelabel{1}';
% for i = 1:length(X)
%     X{i} = X{i}';
% end 
num_view = size(X, 2);           % 视图数量
sam_num = size(X{1}, 2);         % 样本数（假设所有视图样本数一致）
class_num = max(Y);              % 类别数

% 去除每个视图中零方差的特征（行）
for v = 1:num_view
    std_v = std(X{v}, 0, 2);     % 按行计算标准差
    valid_idx = std_v ~= 0;      % 找出非零方差的行
    X{v} = X{v}(valid_idx, :);   % 保留有效特征
end

% 合并所有视图为一个大特征矩阵（用于后续聚类）
X_Multi = [];
for v = 1:num_view
    X_Multi = [X_Multi; X{v}];
end

d_fea = size(X_Multi, 1);        % 总特征数
num_features_to_select = ceil(temp_fea_num * d_fea);  % 每个比例对应的特征数

%% ========== 调用特征评分函数 ==========
% 注意：你的函数名为 evalute（可能拼写错误），请确保它存在
fprintf('Running feature evaluation with lambda1=%.2g, lambda2=%.2g, lambda3=%.2g...\n', ...
        lambda1, lambda2, lambda3);

tic;
[score, OBJ,F,W,match_error] = evalute(X', Y, num_view, lambda1, lambda2, lambda3);
toc;

% 合并所有视图的特征得分
sscore = [];
for v = 1:num_view
    sscore = [sscore, score{v}];  % 假设 score 是 cell，每个元素是列向量
end

% 按得分降序排序，获取特征索引
[~, index] = sort(sscore, 'descend');
Fea_fs = X_Multi(index, :);  % 按重要性排序后的特征矩阵

%% ========== 聚类评估 ==========
% 初始化结果存储
ACC_all = zeros(length(num_features_to_select), 50);
NMI_all = zeros(length(num_features_to_select), 50);
Purity_all = zeros(length(num_features_to_select), 50);
Precision_all = zeros(length(num_features_to_select), 50);
Recall_all = zeros(length(num_features_to_select), 50);
F_score_all = zeros(length(num_features_to_select), 50);
ARI_all = zeros(length(num_features_to_select), 50);

% 对每个特征子集进行聚类评估
for j = 1:length(num_features_to_select)
    feature_num = num_features_to_select(j);
    newfea = Fea_fs(1:feature_num, :)';  % 注意转置：KMeans 需要 样本×特征
    
    fprintf('Evaluating with %d features (%.1f%%)...\n', feature_num, temp_fea_num(j)*100);
    
    for op = 1:50
        % 运行 KMeans
        label = kmeans(newfea, class_num, ...
            'MaxIter', MAXiter, ...
            'Replicates', REPlic, ...
            'EmptyAction', 'singleton');
        
        % 计算评价指标
        result = EvaluationMetrics(Y, label);
        ACC_all(j, op)       = result(1);
        NMI_all(j, op)       = result(2);
        Purity_all(j, op)    = result(3);
        Precision_all(j, op) = result(4);
        Recall_all(j, op)    = result(5);
        F_score_all(j, op)   = result(6);
        ARI_all(j, op)       = result(7);
    end
end

%% ========== 计算均值和标准差 ==========
MeanACC = mean(ACC_all, 2);
StdACC  = std(ACC_all, 0, 2);

MeanNMI = mean(NMI_all, 2);
StdNMI  = std(NMI_all, 0, 2);

MeanPurity = mean(Purity_all, 2);
StdPurity  = std(Purity_all, 0, 2);

MeanPrecision = mean(Precision_all, 2);
StdPrecision  = std(Precision_all, 0, 2);

MeanRecall = mean(Recall_all, 2);
StdRecall  = std(Recall_all, 0, 2);

MeanF_score = mean(F_score_all, 2);
StdF_score  = std(F_score_all, 0, 2);

MeanARI = mean(ARI_all, 2);
StdARI  = std(ARI_all, 0, 2);

%% ========== 显示结果 ==========
fprintf('\n=== Results (Mean ± Std) ===\n');
for j = 1:length(num_features_to_select)
    fprintf('Top %.1f%% features (%d):\n', temp_fea_num(j)*100, num_features_to_select(j));
    fprintf('  ACC:      %.4f ± %.4f\n', MeanACC(j), StdACC(j));
    fprintf('  NMI:      %.4f ± %.4f\n', MeanNMI(j), StdNMI(j));
    fprintf('  Purity:   %.4f ± %.4f\n', MeanPurity(j), StdPurity(j));
    fprintf('  F-score:  %.4f ± %.4f\n', MeanF_score(j), StdF_score(j));
    fprintf('  ARI:      %.4f ± %.4f\n\n', MeanARI(j), StdARI(j));
end

%% ========== 可选：保存结果 ==========
% save('single_run_results.mat', 'MeanACC', 'StdACC', 'MeanNMI', 'StdNMI', ...
%      'lambda1', 'lambda2', 'lambda3', 'temp_fea_num');


MMML = tsne(newfea);
% gscatter(MMML(:,1),MMML(:,2),label);
h = gscatter(MMML(:,1), MMML(:,2), label, 'cmkbrg', 'xo*^sdpvH+<>',5);
legend('Location', 'northeast', ...  % 固定位置为右上角
       'FontName', 'SimHei', ...      % 支持中文
       'FontSize', 10, ...            % 字体大小
       'Box', 'on');   