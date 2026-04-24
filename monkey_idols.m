%%% Team Members: Monkey Idols———Feifan Huang; Monika Kujur; Gum
%%% Chong;Junkai Wang
%%% BMI Spring 2015 (Update 17th March 2015)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         PLEASE READ BELOW            %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function positionEstimator has to return the x and y coordinates of the
% monkey's hand position for each trial using only data up to that moment
% in time.
% You are free to use the whole trials for training the classifier.

% To evaluate performance we require from you two functions:

% A training function named "positionEstimatorTraining" which takes as
% input the entire (not subsampled) training data set and which returns a
% structure containing the parameters for the positionEstimator function:
% function modelParameters = positionEstimatorTraining(training_data)
% A predictor named "positionEstimator" which takes as input the data
% starting at 1ms and UP TO the timepoint at which you are asked to
% decode the hand position and the model parameters given by your training
% function:

% function [x y] = postitionEstimator(test_data, modelParameters)
% This function will be called iteratively starting with the neuronal data 
% going from 1 to 320 ms, then up to 340ms, 360ms, etc. until 100ms before 
% the end of trial.


% Place the positionEstimator.m and positionEstimatorTraining.m into a
% folder that is named with your official team name.

% Make sure that the output contains only the x and y coordinates of the
% monkey's hand.


function [modelParameters] = positionEstimatorTraining(training_data)
  % Arguments:
  % ---------- hyperparams ----------
    modelParameters.binSize = 20;
    modelParameters.tStart  = 320;
    modelParameters.Khist   = 5;
    modelParameters.lambda  = 1e3;

    [nTrials, nDir] = size(training_data);
    nNeurons = size(training_data(1,1).spikes, 1);

    % max trial length
    Tmax = 0;
    for n = 1:nTrials
        for k = 1:nDir
            Tmax = max(Tmax, size(training_data(n,k).spikes,2));
        end
    end

    times = modelParameters.tStart:modelParameters.binSize:Tmax;
    modelParameters.times = times;
    nTime = numel(times);

    % ---------- (A) direction templates ----------
    dirMean = zeros(nNeurons, nDir);
    for k = 1:nDir
        acc = zeros(nNeurons,1);
        cnt = 0;
        for n = 1:nTrials
            sp = training_data(n,k).spikes;
            tUse = min(modelParameters.tStart, size(sp,2));
            acc = acc + sum(sp(:,1:tUse), 2);
            cnt = cnt + 1;
        end
        dirMean(:,k) = acc / max(cnt,1);
    end
    modelParameters.dirMeanCounts = dirMean;

    % ---------- (B) ridge regression weights ----------
    W = cell(nDir, nTime);

    for k = 1:nDir
        for ti = 1:nTime
            tNow = times(ti);
            X = []; Y = [];

            for n = 1:nTrials
                sp = training_data(n,k).spikes;
                hp = training_data(n,k).handPos;

                if tNow > size(sp,2), continue; end

                phi = local_makeFeature(sp, tNow, modelParameters.binSize, modelParameters.Khist);
                X = [X, [phi; 1]];
                Y = [Y, hp(1:2, tNow)];
            end

            F = nNeurons*modelParameters.Khist + 1;

            if isempty(X)
                W{k,ti} = zeros(2, F);
            else
                lam = modelParameters.lambda;
                W{k,ti} = (Y * X.') / (X*X.' + lam * eye(size(X,1)));
            end
        end
    end

    modelParameters.W = W;
end
  % - training_data:
  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)
  
  % ... train your model
  
  % Return Value:
  
  % - modelParameters:
  %     single structure containing all the learned parameters of your
  %     model and which can be used by the "positionEstimator" function.
  


function [x, y] = positionEstimator(test_data, modelParameters)
        sp = test_data.spikes;
    tNow = size(sp,2);

    % ---------- (1) direction classification ----------
    tCls = min(modelParameters.tStart, tNow);
    counts = sum(sp(:,1:tCls), 2);
    dirMean = modelParameters.dirMeanCounts;

    dists = sum((dirMean - counts).^2, 1);
    [~, kHat] = min(dists);

    % ---------- (2) pick latest trained time <= tNow ----------
    times = modelParameters.times;
    ti = find(times <= tNow, 1, 'last');

    if isempty(ti)
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        return;
    end

    % ---------- (3) ridge regression ----------
    phi = local_makeFeature(sp, tNow, modelParameters.binSize, modelParameters.Khist);
    W_now = modelParameters.W{kHat, ti};
    pos = W_now * [phi; 1];

    x = pos(1);
    y = pos(2);
end

% ===== helper =====
function phi = local_makeFeature(spikes, tNow, binSize, Khist)
    [nNeurons, ~] = size(spikes);
    bNow = floor(tNow / binSize);
    feat = zeros(nNeurons, Khist);

    for idx = 1:Khist
        bj = bNow - Khist + idx;
        if bj < 1, continue; end
        t1 = (bj-1)*binSize + 1;
        t2 = min(bj*binSize, tNow);
        feat(:,idx) = sum(spikes(:,t1:t2), 2);
    end
    phi = feat(:);
end


  % **********************************************************
  %
  % You can also use the following function header to keep your state
  % from the last iteration
  %
  % function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
  %                 ^^^^^^^^^^^^^^^^^^
  % Please note that this is optional. You can still use the old function
  % declaration without returning new model parameters. 
  %
  % *********************************************************

  % - test_data:
  %     test_data(m).trialID
  %         unique trial ID
  %     test_data(m).startHandPos
  %         2x1 vector giving the [x y] position of the hand at the start
  %         of the trial
  %     test_data(m).decodedHandPos
  %         [2xN] vector giving the hand position estimated by your
  %         algorithm during the previous iterations. In this case, N is 
  %         the number of times your function has been called previously on
  %         the same data sequence.
  %     test_data(m).spikes(i,t) (m = trial id, i = neuron id, t = time)
  %     in this case, t goes from 1 to the current time in steps of 20
  %     Example:
  %         Iteration 1 (t = 320):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = []
  %             test_data.spikes = 98x320 matrix of spiking activity
  %         Iteration 2 (t = 340):
  %             test_data.trialID = 1;
  %             test_data.startHandPos = [0; 0]
  %             test_data.decodedHandPos = [2.3; 1.5]
  %             test_data.spikes = 98x340 matrix of spiking activity
  
  
  
  % ... compute position at the given timestep.
  
  % Return Value:
  
  % - [x, y]:
  %     current position of the hand
   
