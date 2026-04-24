%% Monika Kujur, Gum Chong, Feifan Huang, Junkai Wang

function model = train_decoder(trial)
[numTrials, numAngles] = size(trial);
numUnits = size(trial(1,1).spikes,1);

%% 1. Compute maximum trial duration
maxT = 0;
for ang = 1:numAngles
    for tr = 1:numTrials
        maxT = max(maxT, size(trial(tr,ang).spikes,2));
    end
end

%% 2. Pad spikes and hand positions to maxT
paddedSpikes = zeros(numTrials, numAngles, numUnits, maxT);
paddedHand   = zeros(numTrials, numAngles, 3, maxT);

for ang = 1:numAngles
    for tr = 1:numTrials
        s = trial(tr,ang).spikes;   % numUnits x T
        h = trial(tr,ang).handPos;  % 3 x T
        T = size(s,2);
        paddedSpikes(tr,ang,:,1:T) = s;
        paddedHand(tr,ang,:,1:T)   = h;
    end
end

%% 3. Binning parameters and trimming to an integer number of bins
bin = 20;                          % ms per bin
numBins = floor(maxT / bin);       % integer number of bins
Ttrim = numBins * bin;             % trimmed time length (<= maxT)

% Trim the padded arrays to Ttrim to make reshape exact
paddedSpikesTrim = paddedSpikes(:,:,:,1:Ttrim);   % size: trials x angles x units x Ttrim
paddedHandTrim   = paddedHand(:,:,:,1:Ttrim);     % size: trials x angles x 3 x Ttrim

%% 4. Reshape into bins and sum to get spike counts per bin
% reshape time dimension into (bin x numBins) then sum over bin
% target shape after reshape: trials x angles x units x bin x numBins
tmp = reshape(paddedSpikesTrim, numTrials, numAngles, numUnits, bin, numBins);
spikeCounts = squeeze(sum(tmp, 4));   % trials x angles x units x numBins

%% 5. Downsample hand positions to same bins (take mean position per bin)
tmpH = reshape(paddedHandTrim, numTrials, numAngles, 3, bin, numBins);
handPos = squeeze(mean(tmpH, 4));    % trials x angles x 3 x numBins

%% 6. Flatten into feature matrix X and target matrix Y
X = reshape(spikeCounts, [], numUnits);   % (trials*angles*numBins) x numUnits
Y = reshape(handPos, [], 3);              % (trials*angles*numBins) x 3

%% 7. Train decoder (ridge regression)
lambda = 1e-3;
W = (X' * X + lambda * eye(numUnits)) \ (X' * Y);

%% 8. Store model metadata (so test_estimator can apply same trimming/binning)
model.W = W;
model.bin = bin;
model.numBins = numBins;
model.Ttrim = Ttrim;
model.maxT = maxT;
model.numUnits = numUnits;
end