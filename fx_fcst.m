function fx_fcst()

    set(0,'DefaultFigureWindowStyle','docked')
    clear;
    clc;
    fclose all;
    close all;
    
    data_tbl=readtable('EURUSD_1d.csv','delimiter',',');    
    data=[0.5*(data_tbl.High+data_tbl.Low)]';

    X=data(1:end-1);
    Y=data(2:end);
    
    numTimeStepsTrain = floor(0.85*numel(X));
    
    XTrain=X(1:numTimeStepsTrain);
    YTrain=Y(1:numTimeStepsTrain);
    
    XTest=X(numTimeStepsTrain+1:end);
    YTest=Y(numTimeStepsTrain+1:end);
    
    mu_x = mean(XTrain);
    sig_x = std(XTrain);
    
    mu_y = mean(YTrain);
    sig_y = std(YTrain);
    
    XTrain_std=(XTrain - mu_x) / sig_x;
    XTest_std=(XTest - mu_x) / sig_x;
    
    YTrain_std=(YTrain - mu_y) / sig_y;
    YTest_std=(YTest - mu_y) / sig_y;
    
    numFeatures = 1;
    numResponses = 1;
    numHiddenUnits = 25;

    layers = [ ...
        sequenceInputLayer(numFeatures)
        lstmLayer(numHiddenUnits)    
        dropoutLayer(0.2)    
        fullyConnectedLayer(numResponses)
        regressionLayer];

    options = trainingOptions('adam', ...
        'MaxEpochs',250, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',0.005, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',125, ...
        'LearnRateDropFactor',0.2, ...
        'Verbose',true, ...
        'Plots','none',...
        'ValidationData',{XTest_std,YTest_std}, ...
        'ValidationFrequency',50);

    net = trainNetwork(XTrain_std,YTrain_std,layers,options);

    net = resetState(net);
    net = predictAndUpdateState(net,XTrain_std);

    YPred_std = [];
    numTimeStepsTest = numel(XTest);
    for i = 1:numTimeStepsTest
        [net,YPred_std(:,i)] = predictAndUpdateState(net,XTest_std(:,i),'ExecutionEnvironment','cpu');
    end

    YPred = sig_y*YPred_std + mu_y;
    rmse = sqrt(mean((YPred-YTest).^2))

    figure();
    plot(YTest,'.-');
    hold on;
    plot(YPred,'.-');
    hold off;
    legend(["Observed" "Predicted"]);
    grid on;
    disp('');

end


