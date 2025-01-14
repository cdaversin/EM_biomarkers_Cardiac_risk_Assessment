
%% Load validation data
clear
load('val_BIOMT.mat') 
pop_size=190;
val_BIOMT=val_BIOMT;
for i=1:length(val_BIOMT)/pop_size %val_BIOMT
    data =val_BIOMT((i-1)*pop_size+1:i*pop_size,:);
     
    load('SVM_classifier.mat')
    load('LogisticRegressionClassifiers.mat')
    rows_with_eads = any(isnan(data), 2);
    data_no_eads = data(~rows_with_eads, :);
    output_ind = SVM_classifier.predict(data_no_eads); 
    percentages(i,:)=[(sum(rows_with_eads==1)+sum(output_ind==1))/pop_size sum(output_ind==2)/pop_size sum(output_ind==3)/pop_size]; %CAMBIO: numel(rows_with_eads)

    save('Test_percentages','percentages')
%% 

    ypredictH=Hregres.predictFcn(percentages(i,:)); %High vs. non-high prediction
    ypredictL=Lregres.predictFcn(percentages(i,:)); %Low vs. non-low prediction
    % [1 0] = High risk; [0 0] = Intermedite risk; [0 1] = Low risk

%% 
    [TdP_score(i)] = calculate_TdP_score (percentages(i,:));

%% 

    load('TdP-score_thresholds.mat')
    if TdP_score(i) < threshold_LvsnL 
        label = 'Low TdP-risk';
        disp('Prediction: Low TdP-risk')
        TdP_risk{i}='Low TdP-risk';
        X=['TdP-score: ',num2str(TdP_score(i))];
        disp(X)
    elseif TdP_score(i) >= threshold_HvsnH
        label = 'High TdP-risk';
        disp('Prediction: High TdP-risk')
        TdP_risk{i} ='High TdP-risk';
        X=['TdP-score: ',num2str(TdP_score(i))];
        disp(X)
    elseif TdP_score(i) >= threshold_LvsnL && TdP_score(i)< threshold_HvsnH
        label = 'Int. TdP-risk';
        disp('Prediction: Int. TdP-risk')
        TdP_risk{i} ='Intermediate TdP-risk';
        X=['TdP-score: ',num2str(TdP_score(i))];
        disp(X)
    end
end
TdP_score=TdP_score';
TdP_risk=TdP_risk';
save("TdP_results_val",'TdP_score','TdP_risk')
