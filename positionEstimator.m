function [x, y] = positionEstimator(test_data, modelParameters)
    persistent decoderState
    
    W = modelParameters.W;
    dirMeanCounts = modelParameters.dirMeanCounts;
    times = modelParameters.times;
    binSize = modelParameters.binSize;
    Khist = modelParameters.Khist;
    tStart = modelParameters.tStart;
    PCA = modelParameters.PCA;
    
    sp = test_data.spikes;
    
    if isempty(sp)
        x = 0;
        y = 0;
        return;
    end
    
    if isempty(decoderState)
        decoderState.last_t = 0;
        decoderState.spike_buffer = [];
        decoderState.prev_pred = [0; 0];
        decoderState.trialId = test_data.trialId;
    end
    
    if isfield(test_data, 'trialId') && test_data.trialId ~= decoderState.trialId
        decoderState.last_t = 0;
        decoderState.spike_buffer = [];
        decoderState.prev_pred = [0; 0];
        decoderState.trialId = test_data.trialId;
    end
    
    current_t = size(sp, 2);
    
    if current_t <= decoderState.last_t
        x = decoderState.prev_pred(1);
        y = decoderState.prev_pred(2);
        return;
    end
    
    if decoderState.last_t == 0
        decoderState.spike_buffer = sp;
    else
        decoderState.spike_buffer = [decoderState.spike_buffer, sp(:, decoderState.last_t+1:end)];
    end
    
    decoderState.last_t = current_t;
    
    % Classify direction
    tUse = min(tStart, size(decoderState.spike_buffer, 2));
    cumSpikes = sum(decoderState.spike_buffer(:, 1:tUse), 2);
    
    nDir = size(dirMeanCounts, 2);
    correlations = zeros(1, nDir);
    for d = 1:nDir
        template = dirMeanCounts(:, d);
        correlations(d) = corr_helper(cumSpikes, template);
    end
    
    [~, estDir] = max(correlations);
    [~, timeIdx] = min(abs(times - current_t));
    
    phi = makeFeature_PosEst(decoderState.spike_buffer, current_t, binSize, Khist);
    
    % Apply PCA transformation
    Scores = PCA.U' * (phi - PCA.mu);
    Xreg = [Scores; 1];
    
    W_dt = W{estDir, timeIdx};
    
    if all(W_dt(:) == 0)
        pred = decoderState.prev_pred;
    else
        pred = W_dt * Xreg;
    end
    
    alpha = 0.7;
    smoothed = alpha * pred + (1 - alpha) * decoderState.prev_pred;
    decoderState.prev_pred = smoothed;
    
    x = smoothed(1);
    y = smoothed(2);
    
end

% ===== HELPER: Make feature =====
function phi = makeFeature_PosEst(spikes, tNow, binSize, Khist)
    [nNeurons, Ttrial] = size(spikes);
    bNow = floor(tNow / binSize);
    feat = zeros(nNeurons, Khist);
    
    for idx = 1:Khist
        bj = bNow - Khist + idx;
        if bj < 1
            feat(:, idx) = 0;
            continue;
        end
        t1 = (bj - 1) * binSize + 1;
        t2 = bj * binSize;
        t1 = max(1, t1);
        t2 = min([t2, tNow, Ttrial]);
        if t1 > t2
            feat(:, idx) = 0;
        else
            feat(:, idx) = sum(spikes(:, t1:t2), 2);
        end
    end
    
    phi = feat(:);
end

% ===== HELPER: Simple correlation =====
function c = corr_helper(x, y)
    x = x - mean(x);
    y = y - mean(y);
    denom = sqrt(sum(x.^2)) * sqrt(sum(y.^2));
    if denom < eps
        c = 0;
    else
        c = sum(x .* y) / denom;
    end
end