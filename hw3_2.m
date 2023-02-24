emails = readtable("emails.csv", ReadRowNames=true, ReadVariableNames=true, VariableNamingRule="preserve");
%%
fold1_test = emails(1:1000,:);
fold1_train = emails(1001:end,:);
fold2_test = emails(1000:2000,:);
fold2_train =emails([1:999, 2001:end], :);
fold3_test = emails(2000:3000,:);
fold3_train =emails([1:1999, 3001:end], :);
fold4_test = emails(3000:4000,:);
fold4_train =emails([1:2999, 4001:end], :);
fold5_test = emails(4000:end,:);
fold5_train =emails(1:3999, :);

[accuracy1, precision1, recall1] = test_knn(fold1_train, fold1_test)
[accuracy2, precision2, recall2] = test_knn(fold2_train, fold2_test)
[accuracy3, precision3, recall3] = test_knn(fold3_train, fold3_test)
[accuracy4, precision4, recall4] = test_knn(fold4_train, fold4_test)
[accuracy5, precision5, recall5] = test_knn(fold5_train, fold5_test)
%%
function [accuracy, precision, recall] = test_knn(train_data, test_data) 
    mdl = fitcknn(train_data, 'Prediction', Distance='euclidean', NumNeighbors=1);
    pred = predict(mdl, test_data(:, 1:end-1));
    ind_pred = pred == 1;
    ind_y = table2array(test_data(:,end)) == 1;
    
    TP = sum(ind_pred & ind_y);
    FP = sum(ind_pred & ~ind_y);
    TN = sum(~ind_pred & ~ind_y);
    FN = sum(~ind_pred & ind_y);

    accuracy = (TP + TN) / (TP + FP + TN + FN);
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
end