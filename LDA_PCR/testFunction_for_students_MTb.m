% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 
% clear ; close all; clc

%% function RMSE = testFunction_for_students_MTb(tempname)

function [RMSE, Results] = testFunction_for_students_MTb(teamName)

%load monkeydata0.mat
load monkeydata_training
clear positionEstimator

% Set random number generator
rng(2013);
ix = randperm(length(trial));

addpath(teamName);

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start
% Configuration
plotResults = true;   % set true to enable plotting
showProgress = true;

% Train Model
trainTic = tic;
trainTime = toc(trainTic);
fprintf('Training completed in %.2f s\n', trainTime);

% Containers for aggregated evaluation
decodedAllX = []; decodedAllY = [];
actualAllX  = []; actualAllY  = [];
sampleMap = [];   % rows: [row, direction, trialId, original_T]
decodedTrajectories = {}; actualTrajectories = {};
perSampleCounts = [];   % number of timepoints per sample (for mapping)

% For per-direction aggregation
nDirections = size(testData,2);
directionErrors = cell(1, nDirections);
for d = 1:nDirections, directionErrors{d} = []; end

% Timing
evalTic = tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end

meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start
        % store per-sample decoded and actual for this (tr,direc)
        decoded_this = zeros(2, length(times));
        actual_this  = zeros(2, length(times));
        tIdx = 0;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end
        
        for t=times
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start
            tIdx = tIdx + 1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            % uncomment -- meanSqError = meanSqError +
            % norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2; 

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start

            % accumulate error
            actualPos = testData(tr,direc).handPos(1:2,t);
            err = norm(actualPos - decodedPos)^2;
            meanSqError = meanSqError + err;

            % store for aggregated metrics
            decoded_this(:, tIdx) = decodedPos;
            actual_this(:,  tIdx) = actualPos;

            n_predictions = n_predictions + 1;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end
            
        end
        % uncomment -- n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start
    % Save per-sample results
    decodedTrajectories{end+1} = decoded_this; %#ok<AGROW>
    actualTrajectories{end+1}  = actual_this;  %#ok<AGROW>
    sampleMap(end+1, :) = [tr, direc, testData(tr,direc).trialId, size(testData(tr,direc).spikes,2)]; %#ok<AGROW>
    perSampleCounts(end+1) = size(decoded_this,2); %#ok<AGROW>

    % accumulate arrays for global metrics
    decodedAllX = [decodedAllX; decoded_this(1,:)']; %#ok<AGROW>
    decodedAllY = [decodedAllY; decoded_this(2,:)']; %#ok<AGROW>
    actualAllX  = [actualAllX;  actual_this(1,:)'];  %#ok<AGROW>
    actualAllY  = [actualAllY;  actual_this(2,:)'];  %#ok<AGROW>

    % per-direction errors (squared)
    sqErrs = sum((decoded_this - actual_this).^2, 1); % 1 x nTimes
    directionErrors{direc} = [directionErrors{direc}, sqErrs]; %#ok<AGROW>
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end
end

legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% start
evalTime = toc(evalTic);

% Compute primary metrics
Y = [actualAllX, actualAllY];      % N x 2
Yhat = [decodedAllX, decodedAllY]; % N x 2
SSE = sum((Y - Yhat).^2, 1);       % 1 x 2
SST = sum((Y - mean(Y,1)).^2, 1);  % 1 x 2
MSE = mean((Y - Yhat).^2, 1);      % 1 x 2
RMSE_xy = sqrt(mean(MSE));         % scalar across both dims (same unit as original RMSE)
R2 = 1 - SSE ./ (SST + eps);       % 1 x 2

% Baseline (mean predictor) MSE
Ybar = mean(Y,1);
baselineMSE = mean((Y - Ybar).^2, 1);

% Per-sample MSE (averaged across timepoints and dims)
nSamples = numel(decodedTrajectories);
perSampleMSE = zeros(nSamples,1);
for s = 1:nSamples
    dec = decodedTrajectories{s}; act = actualTrajectories{s};
    diffsq = sum((dec - act).^2, 1);        % 1 x T
    perSampleMSE(s) = mean(diffsq);        % mean across time (already sums dims)
end

% Per-direction RMSE
perDirectionRMSE = zeros(1, nDirections);
for d = 1:nDirections
    if isempty(directionErrors{d})
        perDirectionRMSE(d) = NaN;
    else
        perDirectionRMSE(d) = sqrt(mean(directionErrors{d}));
    end
end

% Pack Results struct
Results = struct();
Results.n_predictions = n_predictions;
Results.MSE = MSE;                 % [MSE_x, MSE_y]
Results.RMSE_xy = RMSE_xy;            % scalar across X and Y (cm)
Results.R2 = R2;                   % [R2_x, R2_y]
Results.baselineMSE = baselineMSE; % [MSE_x_baseline, MSE_y_baseline]
Results.perSample.MSE = perSampleMSE;
Results.perDirection.RMSE = perDirectionRMSE;
Results.decodedTrajectories = decodedTrajectories;
Results.actualTrajectories = actualTrajectories;
Results.sampleMap = sampleMap;     % [row, direction, trialId, original_T]
Results.runtimeSeconds = evalTime;
Results.trainingTimeSeconds = trainTime;
Results.plotHandles = [];          % filled if plotting enabled

% Optionally create summary plots
if plotResults
    ph = {};
    % Scatter actual vs predicted for X and Y
    for d = 1:2
        fh = figure('Name', sprintf('Actual vs Predicted - %s', char('X'+(d-1))));
        scatter(Y(:,d), Yhat(:,d), 8, 'filled'); hold on;
        mn = min([Y(:,d); Yhat(:,d)]); mx = max([Y(:,d); Yhat(:,d)]);
        plot([mn mx],[mn mx],'r--','LineWidth',1.2);
        xlabel('Actual'); ylabel('Predicted');
        title(sprintf('Actual vs Predicted (%s)  MSE=%.4f  R^2=%.3f', char('X'+(d-1)), MSE(d), R2(d)));
        grid on; hold off;
        ph{end+1} = fh; %#ok<AGROW>
    end

    % Residual histograms
    fh = figure('Name','Residuals (X and Y)');
    subplot(2,1,1); histogram(Y(:,1)-Yhat(:,1)); title('Residuals X');
    subplot(2,1,2); histogram(Y(:,2)-Yhat(:,2)); title('Residuals Y');
    ph{end+1} = fh;

    % Per-direction RMSE bar
    fh = figure('Name','Per-direction RMSE');
    bar(perDirectionRMSE);
    xlabel('Direction index'); ylabel('RMSE'); title('Per-direction RMSE (averaged across trials)');
    ph{end+1} = fh;

    Results.plotHandles = ph;
end



% Print concise summary
fprintf('Evaluation finished: %d predictions, runtime %.2fs\n', Results.n_predictions, Results.runtimeSeconds);
fprintf('RMSE (X,Y combined): %.4f\n', Results.RMSE_xy);
fprintf('MSE X=%.4f, Y=%.4f; R2 X=%.4f, Y=%.4f\n', Results.MSE(1), Results.MSE(2), Results.R2(1), Results.R2(2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end
rmpath(genpath(teamName))

end


% [RMSE, Results] = testFunction_for_students_MTb(tempname)