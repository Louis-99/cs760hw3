emails = table2array(readtable("emails.csv", ReadRowNames=true, ReadVariableNames=true, VariableNamingRule="preserve"));
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

%%
[accuracy1, precision1, recall1]=test_logistic_5fold(fold1_train, fold1_test, 0.05)
[accuracy2, precision2, recall2]=test_logistic_5fold(fold2_train, fold2_test, 0.05)
[accuracy3, precision3, recall3]=test_logistic_5fold(fold3_train, fold3_test, 0.05)
[accuracy4, precision4, recall4]=test_logistic_5fold(fold4_train, fold4_test, 0.05)
[accuracy5, precision5, recall5]=test_logistic_5fold(fold5_train, fold5_test, 0.05)
%%
function [accuracy, precision, recall] = test_logistic_5fold(train_data, test_data, eta)
    fn = logistic_fit_fn(train_data(:,1:end-1), train_data(:,end), eta);
    pred = fn(test_data(:,1:end-1)) ~= 0;
    y = test_data(:,end) ~= 0;
    TP = sum(pred & y);
    FP = sum(pred & ~y);
    TN = sum(~pred & ~y);
    FN = sum(~pred & y);

    accuracy = (TP + TN) / (TP + FP + TN + FN);
    precision = TP / (TP + FP);
    recall = TP / (TP + FN);
end

function theta = logistic_fit(X, y, eta)
    theta = zeros(1,size(X,2));
    for k = 1:3000
        d = (logistic(theta, X) - y)' * X ./ length(y);
        theta = theta - eta .* d;
    end
end

function fn = logistic_fit_fn(X, y, eta)
    theta = logistic_fit(X, y, eta);
    fn = @(x) logistic(theta, x) > 0.5;
end

function ret = logistic(theta, x)
    ret = 1 ./ (exp(-theta*x')' + 1);
end