%% Monika Kujur, Gum Chong, Feifan Huang, Junkai Wang

function [x,y] = test_estimator(model, testTrials)
% function results = test_estimator(model, testTrials)

% TEST_ESTIMATOR  Apply trained decoder to test trials with identical preprocessing.
%
%   results = test_estimator(model, testTrials)
%
% Inputs
%   - model: struct returned by train_decoder, must contain fields:
%       .W       -> (numUnits x 3) decoder weights
%       .bin     -> bin size in ms (scalar)
%       .numBins -> number of bins used during training
%       .Ttrim   -> trimmed time length used during training (numBins*bin)
%       .maxT    -> maximum trial length used when padding during training
%       .numUnits-> number of units used
%   - testTrials: cell array of trial structs (each element has .spikes and .handPos)
%
% Output (struct)
%   - results.Y        -> actual targets matrix (N x 3)
%   - results.Yhat     -> predicted targets matrix (N x 3)
%   - results.sampleMap-> mapping from rows in Y/Yhat to original trial index and bin
%   - results.mse      -> 1x3 mean squared error per dimension
%   - results.R2       -> 1x3 coefficient of determination per dimension
%   - results.model    -> copy of model used
%   - results.bin      -> bin size
%   - results.plots    -> handles to generated figures (if plotting enabled)
%
% Notes
%   - This function pads each test trial to model.maxT with zeros, trims to model.Ttrim,
%     bins spikes (sum) and downsamples handPos (mean per bin), flattens, then predicts.
%   - It assumes the same preprocessing used in train_decoder.

plotResults = false; % true;   % set false to skip plotting

%% Validate model fields
required = {'W','bin','numBins','Ttrim','maxT','numUnits'};
for f = required
    if ~isfield(model, f{1})
        error('Model missing required field: %s', f{1});
    end
end

W = model.W;
bin = model.bin;
numBins = model.numBins;
Ttrim = model.Ttrim;
maxT = model.maxT;
numUnits = model.numUnits;

%% Prepare containers
nSamples = numel(testTrials);
% Preallocate spikeCounts and handPos arrays: samples x units x numBins and samples x 3 x numBins
spikeCountsAll = zeros(nSamples, numUnits, numBins);
handPosAll     = zeros(nSamples, 3, numBins);
sampleMap = zeros(nSamples, 2); % [trialIndex, original_T]

for i = 1:nSamples
    trStruct = testTrials{i};
    s = trStruct.spikes;    % expected size: numUnits x T_i
    h = trStruct.handPos;   % expected size: 3 x T_i

    % Basic sanity checks
    if size(s,1) ~= numUnits
        error('Test trial %d has %d units but model expects %d units.', i, size(s,1), numUnits);
    end

    T_i = size(s,2);
    sampleMap(i,:) = [i, T_i];

    % 1) Pad to model.maxT
    sPad = zeros(numUnits, maxT);
    hPad = zeros(3, maxT);
    sPad(:,1:T_i) = s;
    hPad(:,1:T_i) = h;

    % 2) Trim to Ttrim (same as training)
    sTrim = sPad(:, 1:Ttrim);
    hTrim = hPad(:, 1:Ttrim);

    % 3) Reshape into bins: (numUnits x bin x numBins) and sum over bin
    sReshaped = reshape(sTrim, numUnits, bin, numBins);   % numUnits x bin x numBins
    sBinned = squeeze(sum(sReshaped, 2));                 % numUnits x numBins

    % 4) Hand position: reshape and take mean per bin
    hReshaped = reshape(hTrim, 3, bin, numBins);          % 3 x bin x numBins
    hBinned = squeeze(mean(hReshaped, 2));                % 3 x numBins

    % Store
    spikeCountsAll(i,:,:) = sBinned;   % 1 x numUnits x numBins
    handPosAll(i,:,:)     = hBinned;   % 1 x 3 x numBins
end

%% Flatten to feature matrix X and target matrix Y
% X: (nSamples * numBins) x numUnits
% Y: (nSamples * numBins) x 3
X = reshape(permute(spikeCountsAll, [1 3 2]), [], numUnits);  % (samples*numBins) x numUnits
Y = reshape(permute(handPosAll, [1 3 2]), [], 3);             % (samples*numBins) x 3

%% Predict
Yhat = X * W;   % (N x 3)

%% Evaluation metrics
N = size(Y,1);
mse = mean((Y - Yhat).^2, 1);                % 1 x 3
SSE = sum((Y - Yhat).^2, 1);
SST = sum((Y - mean(Y,1)).^2, 1);
R2 = 1 - SSE ./ (SST + eps);

%% Pack results
% results.Y = Y;
% results.Yhat = Yhat;
% results.sampleMap = sampleMap;
% results.mse = mse;
% results.R2 = R2;
% results.model = model;
% results.bin = bin;
% results.numBins = numBins;
% results.nRows = N;
% results.plots = [];

x = Yhat(:,1);
y = Yhat(:,2);

%% Optional plotting: scatter and example trial traces
if plotResults
    figs = gobjects(3,1);

    % 1) Scatter actual vs predicted for each dimension
    dims = {'X','Y','Z'};
    for d = 1:3
        figs(d) = figure('Name', ['Actual vs Predicted - ' dims{d}]);
        scatter(Y(:,d), Yhat(:,d), 6, 'filled');
        hold on;
        mn = min([Y(:,d); Yhat(:,d)]);
        mx = max([Y(:,d); Yhat(:,d)]);
        plot([mn mx], [mn mx], 'r--', 'LineWidth', 1.2);
        xlabel(['Actual ' dims{d}]);
        ylabel(['Predicted ' dims{d}]);
        title(sprintf('Actual vs Predicted (%s)  MSE=%.4f  R^2=%.3f', dims{d}, mse(d), R2(d)));
        axis tight;
        grid on;
    end

    % 2) Time-series for first few test trials (actual vs predicted)
    nShow = min(6, nSamples);
    figTS = figure('Name','Example trial predictions (first few trials)');
    for i = 1:nShow
        subplot(nShow,1,i);
        % extract rows corresponding to this sample
        rowStart = (i-1)*numBins + 1;
        rowEnd   = i * numBins;
        t = (1:numBins) * bin;  % bin centers in ms (approx)
        plot(t, Y(rowStart:rowEnd,1), 'b-', 'LineWidth', 1); hold on;
        plot(t, Yhat(rowStart:rowEnd,1), 'r--', 'LineWidth', 1);
        ylabel(sprintf('Trial %d X', i));
        if i==1, legend('Actual','Predicted'); end
        if i==nShow, xlabel('Time (ms)'); end
        xlim([t(1) t(end)]);
    end

    results.plots = [figs; figTS];
end

end

% % Function call example

% s = trial(1, 4).spikes;
% h = trial(1, 4).handPos;
% 
% singleCell = cell(1,1);
% singleCell{1}.spikes = s;
% singleCell{1}.handPos = h;
%
% test_2(model, singleCell)