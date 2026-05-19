%Transpose file from Kuo scripts, label and merge for CNN analysis
clc; clear; close all;
% File names
fileA = 'Spectra_500_YAA10_ALD_HIF7S_Mono_pH7_train.xlsx';%Training data no PTM
fileB = 'Specclnnorm_400_YAA4_HIF7SPTM_PTMtrain.xlsx';%Training data with PTM
fileAP = 'Spectra_500_YAA10_ALD_HIF7S_Mono_pH7_post.xlsx';% Post evaluation no PTM
fileBP = 'Specclnnorm_400_YAA4_HIF7SPTM_PTMpost.xlsx';% Post evaluation with PTM
outputFile = 'mergedlabeledspectra.mat';
outputFilePost = 'mergedlabeledspectraPost.mat';
%% Merge training data
% Read data
A = readmatrix(fileA);
B = readmatrix(fileB);

% ----- Preprocessing -----
% Remove first column (e.g., Raman shift)
A = A(:, 2:end);
B = B(:, 2:end);

% Transpose so each row is one spectrum
A = A.';
B = B.';

% ----- Labeling -----
labelsA = ones(size(A,1), 1);      % label = 1
labelsB = 2 * ones(size(B,1), 1);  % label = 2

% Append labels
A_labeled = [A labelsA];
B_labeled = [B labelsB];

% ----- Merge -----
mergedData = [A_labeled; B_labeled];

% Save result
save(outputFile,  'mergedData', '-v7.3');


disp('Training data processing complete: first column removed, transposed, labeled, and merged.');
%% Merge Postevaluation data
% Read data
AP= readmatrix(fileAP);
BP= readmatrix(fileBP);

% ----- Preprocessing -----
% Remove first column (e.g., Raman shift)
AP = AP(:, 2:end);
BP = BP(:, 2:end);

% Transpose so each row is one spectrum
AP = AP.';
BP = BP.';

% ----- Labeling -----
labelsAP = ones(size(AP,1), 1);      % label = 1
labelsBP = 2 * ones(size(BP,1), 1);  % label = 2

% Append labels
AP_labeled = [AP labelsAP];
BP_labeled = [BP labelsBP];

% ----- Merge -----
mergedData = [AP_labeled; BP_labeled];

% Save result
save(outputFilePost,  'mergedData', '-v7.3');


disp('Postevaluation data processing complete: first column removed, transposed, labeled, and merged.');
%%  导入数据
S=load('mergedlabeledspectra.mat');
res=S.mergedData;
SizeA=size(A,1);
SizeB=size(B,1);
TotalNum=SizeA+SizeB;
avg_1=mean(res(1:SizeA,:),1);
avg_2=mean(res((SizeA+1):end,:),1);
results=[avg_1;avg_2];
writematrix(results,'average.xlsx');
disp('average');


%%  CNN ANALYSIS
temp = randperm(TotalNum);%total spectra number
Devide = floor(0.8 * TotalNum);
%Read data and label and Define number for test
P_train = res(temp(1: Devide), 1: 981)';
T_train = res(temp(1: Devide), 982)';
M = size(P_train, 2);
% Define number for test
P_test = res(temp((Devide+1): end), 1: 981)';
T_test = res(temp((Devide+1): end), 982)';
N = size(P_test, 2);

[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

p_train =  double(reshape(P_train, 981, 1, 1, M));
p_test  =  double(reshape(P_test , 981, 1, 1, N));
save('preprocessParams.mat',"ps_input");
layers = [
 imageInputLayer([981, 1, 1])                                % 输入层
 convolution2dLayer([3, 1], 64, 'Padding', 'same','Name','ConvolutionnalNN_1')          % 卷积核大小为 2*1 生成16个卷积
 batchNormalizationLayer('Name', 'bn1')                       % 批归一化层
 reluLayer('Name', 'relu1')                                                  % relu 激活层
 maxPooling2dLayer([2, 1], 'Stride', [2, 1])                % 最大池化层 大小为 2*1 步长为 [2, 1]
 convolution2dLayer([3, 1], 128, 'Padding', 'same','Name','ConvolutionnalNN_2')          % 卷积核大小为 2*1 生成32个卷积
 batchNormalizationLayer('Name', 'bn2')                     % 批归一化层
 reluLayer('Name', 'relu2')                                 % relu 激活层
 maxPooling2dLayer([2, 1], 'Stride', [2, 1]) 
 convolution2dLayer([3, 1], 64, 'Padding', 'same','Name','ConvolutionnalNN_3') 
 batchNormalizationLayer                                    % 批归一化层
 reluLayer
 maxPooling2dLayer([2, 1], 'Stride', [2, 1])                % 最大池化层 大小为 2*1 步长为 [2, 1]
 dropoutLayer(0.5, 'Name', 'dropout1')
 convolution2dLayer([3, 1], 32, 'Padding', 'same','Name','ConvolutionnalNN_4') 
 batchNormalizationLayer                                    % 批归一化层
 reluLayer
 fullyConnectedLayer(32)
 dropoutLayer(0.5, 'Name', 'dropout2')
 fullyConnectedLayer(2)
 softmaxLayer                                               % 损失函数层
 classificationLayer];                                      % 分类层

options = trainingOptions('adam', ...      % Adam 梯度下降
    'MaxEpochs', 100, ...                  % 最大训练次数 500
    'InitialLearnRate', 1e-4, ...          % 初始学习率为 0.0001
    'L2Regularization', 1e-4, ...          % L2正则化参数
    'MiniBatchSize', 64, ...
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 60, ...        % 经过450次训练后 学习率为 0.001 * 0.1
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

net = trainNetwork(p_train, t_train, layers, options);

t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

analyzeNetwork(layers)

%[T_train, index_1] = sort(T_train);
%[T_test , index_2] = sort(T_test );

%T_sim1 = T_sim1(index_1);
%T_sim2 = T_sim2(index_2);
ground_truth = double(T_test);
scores_class1 = t_sim2(:, 1);
scores_class2 = t_sim2(:, 2);
[FP_class1, TP_class1, ~, AUC_class1] = perfcurve(ground_truth, scores_class1, 1);
[FP_class2, TP_class2, ~, AUC_class2] = perfcurve(ground_truth, scores_class2, 2);
disp(['AUC for HIF-S: ', num2str(AUC_class1)]);
disp(['AUC for HIF-S-PTM: ', num2str(AUC_class2)]);

roc_data_class1 = [FP_class1, TP_class1];
roc_data_class2 = [FP_class2, TP_class2];
%thresholds_custom = 0:0.01:1;  % 自定义的阈值范围，从0到1，每步0.01
%FP_class1 = zeros(length(thresholds_custom), 1);
%TP_class1 = zeros(length(thresholds_custom), 1);
%FP_class2 = zeros(length(thresholds_custom), 1);
%TP_class2 = zeros(length(thresholds_custom), 1);
%for i = 1:length(thresholds_custom)
    %threshold = thresholds_custom(i);
    %predicted_labels_class1 = scores_class1 > threshold;
    %predicted_labels_class2 = scores_class2 > threshold;  
    %TP_class1(i) = sum(T_test' == 1 & predicted_labels_class1)/sum(T_test' == 1);
    %FP_class1(i) = sum(T_test' == 2 & predicted_labels_class1)/sum(T_test' == 2);  
    %TP_class2(i) = sum(T_test' == 2 & predicted_labels_class2)/sum(T_test' == 2);
    %FP_class2(i) = sum(T_test' == 1 & predicted_labels_class2)/sum(T_test' == 1);
%end
%AUC_class1 = trapz(FP_class1, TP_class1);
%AUC_class2 = trapz(FP_class2, TP_class2);
%disp(['AUC for Hyp: ', num2str(AUC_class1)]);
%disp(['AUC for Pro: ', num2str(AUC_class2)]);

writematrix(roc_data_class1, 'ROC_HIFalldata.xlsx', 'Sheet', 1, 'Range', 'A1');
writematrix(roc_data_class2, 'ROC_HIFPalldata.xlsx', 'Sheet', 1, 'Range', 'A1');

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
xlim([1, M])
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
xlim([1, N])
grid

figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

% Plot ROC curves
figure;
plot(FP_class1, TP_class1, 'r', 'LineWidth', 2); % 绘制类别1的ROC曲线
hold on;
plot(FP_class2, TP_class2, 'b', 'LineWidth', 2); % 绘制类别2的ROC曲线
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Class 1 and Class 2');
legend('Class 1', 'Class 2');
grid on;
% 保存训练好的模型到当前工作目录
save('trainedCNNModelforYingQi.mat', 'net');
%% Post evaluation
load('trainedCNNModelforYingQi.mat', 'net');
load('preprocessParams.mat', 'ps_input');
S=load('mergedlabeledspectraPost.mat');
newData=S.mergedData;


% 提取特征数据
newFeatures = newData(:, 1:981)';  % 大小为 [1189, 样本数]

% 如果有真实标签，可以提取（可选）
if size(newData, 2) >= 981
    trueLabels = newData(:, 982)';
    ground_truth = double(trueLabels);% 大小为 [1, 样本数]
end

% 使用训练数据的归一化参数对新数据进行归一化
newFeaturesNorm = mapminmax('apply', newFeatures, ps_input);

% 获取新样本的数量
numNewSamples = size(newFeaturesNorm, 2);

% 重塑数据
newFeaturesReshaped = reshape(newFeaturesNorm, [981, 1, 1, numNewSamples]);

% 使用模型进行分类，获取预测标签和预测概率
[predictedLabels, scores] = classify(net, newFeaturesReshaped);

% 将预测标签转换为数组
predictedLabelsArray = cellstr(predictedLabels);

sampleLabels = {'1', '2'};
% 获取类别列表
classNames = net.Layers(end).Classes;
scores_class1 = scores(:, 1);
scores_class2 = scores(:, 2);
[FP_class1, TP_class1, ~, AUCP_class1] = perfcurve(ground_truth, scores_class1, 1);
[FP_class2, TP_class2, ~, AUCP_class2] = perfcurve(ground_truth, scores_class2, 2);
disp(['AUCP for HIF7S: ', num2str(AUCP_class1)]);
disp(['AUCP for HIF7SPTM: ', num2str(AUCP_class2)]);
roc_data_class1 = [FP_class1, TP_class1];
roc_data_class2 = [FP_class2, TP_class2];
writematrix(roc_data_class1, 'ROC_HIF-Sallpost1.xlsx', 'Sheet', 1, 'Range', 'A1');
writematrix(roc_data_class2, 'ROC_HIF-S-PTMallpost1.xlsx', 'Sheet', 1, 'Range', 'A1');
% 显示预测结果
for i = 1:numNewSamples
    % 获取当前样本的预测分数（概率）
    sampleScores = scores(i, :);

    % 获取最高的预测概率和对应的类别
    [maxScore, idx] = max(sampleScores);
    predictedClass = classNames(idx);

    fprintf('样本 %d 的预测类别是: %s，预测概率: %.2f%%\n', i, char(predictedClass), maxScore * 100);
end

% 创建表格
T = table((1:numNewSamples)', predictedLabelsArray, max(scores, [], 2) * 100, ...
    'VariableNames', {'SampleIndex', 'PredictedLabel', 'PredictionProbability'});

% 保存到 Excel 文件
writetable(T, '1HIF7S2HIF7SPTM.xlsx');

fprintf('预测结果已保存到 PredictionResults.xlsx\n');

if exist('trueLabels', 'var')
    % 将真实标签转换为分类变量
    trueLabelsCategorical = categorical(trueLabels, [1,2], {'1','2'})';

    % 计算准确率
    accuracy = sum(predictedLabels == trueLabelsCategorical) / numNewSamples * 100;

    fprintf('预测准确率为: %.2f%%\n', accuracy);

    % 生成并显示混淆矩阵
    figure;
    cm = confusionchart(trueLabelsCategorical, predictedLabels);
    cm.Title = '新数据集的混淆矩阵';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end