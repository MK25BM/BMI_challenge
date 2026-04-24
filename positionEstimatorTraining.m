function modelParameters = positionEstimatorTraining(training_data)
% Train per-direction, per-time linear decoders using global PCA on causal features.
% Minimal, robust implementation that ensures feature-target alignment.
%
% Output (modelParameters):
%   W            - cell(nDir, nTime) of 2 x (nComp+1) weight matrices
%   dirMeanCounts- nNeurons x nDir direction templates (counts up to tStart)
%   times        - vector of decoding times
%   binSize, Khist, tStart, lambda
%   PCA.mu, PCA.U, PCA.nComp
%   sampleCounts - nDir x nTime counts (diagnostic)

    % -------------------- hyperparameters --------------------
    binSize = 20;        % ms per bin
    tStart  = 320;       % ms: first decode time and direction-template cutoff
    Khist   = 7;         % number of past bins concatenated into phi
    lambda  = 10;        % ridge penalty (intercept unregularized)
    pcaVarFrac = 0.10;   % keep components explaining this fraction of variance

    % -------------------- dataset sizes --------------------
    [nTrials, nDir] = size(training_data);

    % discover number of neurons and maximum spike length across dataset
    nNeurons = size(training_data(1,1).spikes,1);
    Tmax = 0;
    for tr = 1:nTrials
      for d = 1:nDir
        Tmax = max(Tmax, size(training_data(tr,d).spikes,2));
      end
    end

    % decoding times (same grid used at decode time)
    times = tStart : binSize : Tmax;
    nTime = numel(times);

    % -------------------- (A) direction templates --------------------
    % Use cumulative spike counts up to tStart as simple direction templates.
    % These are used at decode time for coarse direction classification.
    dirMeanCounts = zeros(nNeurons, nDir);
    for d = 1:nDir
      acc = zeros(nNeurons,1);
      for tr = 1:nTrials
        sp = training_data(tr,d).spikes;
        tUse = min(tStart, size(sp,2));   % clamp to available spike length
        acc = acc + sum(sp(:,1:tUse), 2);
      end
      dirMeanCounts(:,d) = acc / nTrials;
    end

    % -------------------- (B) collect all phi for global PCA --------------------
    % Build PhiAll (features x samples). Only include a sample if both spikes
    % and handPos have a column at tNow so features and targets align.
    Fraw = nNeurons * Khist;
    PhiAll = []; % Fraw x M
    for d = 1:nDir
      for ti = 1:nTime
        tNow = times(ti);
        for tr = 1:nTrials
          sp = training_data(tr,d).spikes;
          hp = training_data(tr,d).handPos;
          % include only when both spikes and handPos reach tNow
          if tNow <= size(sp,2) && tNow <= size(hp,2)
            phi = makeFeature_Helper(sp, tNow, binSize, Khist); % Fraw x 1
            PhiAll = [PhiAll, phi];
          end
        end
      end
    end

    % require at least one sample for PCA
    if isempty(PhiAll)
      error('No aligned feature-target samples found. Check training_data and times.');
    end

    % -------------------- (C) PCA application --------------------
    % pca() expects observations in rows -> transpose PhiAll to M x Fraw.
    % coeff: Fraw x Ffull, mu: 1 x Fraw, explained: percent variance per comp.
    [coeff, ~, ~, ~, explained, mu] = pca(PhiAll.', 'Centered', true);

    % choose nComp by cumulative explained variance, but do not exceed numeric rank
    cumExpl = cumsum(explained) / sum(explained);
    nComp = find(cumExpl >= pcaVarFrac, 1, 'first');
    if isempty(nComp), nComp = size(coeff,2); end
    rankPhi = rank(PhiAll);            % numeric rank of feature matrix
    nComp = min(nComp, rankPhi);       % ensure we don't request more components than rank

    Ured = coeff(:,1:nComp);           % Fraw x nComp
    muPhi = mu(:);                     % Fraw x 1 (column vector)

    % -------------------- (D) per-(direction,time) ridge training -----------
    % For each (direction, time) collect aligned samples, project to PCA scores,
    % add intercept (last row), and solve closed-form ridge: W = Y X' (X X' + lambda I)^{-1}
    W = cell(nDir, nTime);
    sampleCounts = zeros(nDir, nTime);
    F = nComp + 1;                      % feature length after adding intercept
    Ireg = eye(F); Ireg(end,end) = 0;   % do not penalize intercept

    for d = 1:nDir
      for ti = 1:nTime
        tNow = times(ti);
        Xphi = []; Y = []; % Xphi: Fraw x N, Y: 2 x N
        for tr = 1:nTrials
          sp = training_data(tr,d).spikes;
          hp = training_data(tr,d).handPos;
          % only include samples where both spikes and handPos reach tNow
          if tNow <= size(sp,2) && tNow <= size(hp,2)
            phi = makeFeature_Helper(sp, tNow, binSize, Khist);
            Xphi = [Xphi, phi];
            Y = [Y, hp(1:2, tNow)];
          end
        end

        N = size(Xphi,2);
        sampleCounts(d,ti) = N;

        % If no samples for this cell, store zeros and continue
        if N == 0
          W{d,ti} = zeros(2, nComp + 1);   % intercept last
          continue;
        end

        % project to PCA scores (nComp x N)
        Scores = Ured' * (bsxfun(@minus, Xphi, muPhi));

        % design matrix with intercept last
        Xreg = [Scores; ones(1,N)]; % (nComp+1) x N

        % closed-form ridge solution (intercept unregularized)
        Wkt = (Y * Xreg.') / (Xreg * Xreg.' + lambda * Ireg); % 2 x F
        W{d,ti} = Wkt;
      end
    end

    % -------------------- pack modelParameters --------------------
    modelParameters.W = W;
    modelParameters.dirMeanCounts = dirMeanCounts;
    modelParameters.times = times;
    modelParameters.binSize = binSize;
    modelParameters.Khist = Khist;
    modelParameters.tStart = tStart;
    modelParameters.lambda = lambda;
    modelParameters.PCA = struct('mu', muPhi, 'U', Ured, 'nComp', nComp);
    modelParameters.sampleCounts = sampleCounts;

end

% ===== NESTED HELPER FUNCTION =====
function phi = makeFeature_Helper(spikes, tNow, binSize, Khist)
    % Build causal feature vector by summing spikes in the last Khist bins.
    % Output phi is (nNeurons*Khist) x 1, bins ordered oldest -> newest.
    [nNeurons, Ttrial] = size(spikes);
    bNow = floor(tNow / binSize);
    feat = zeros(nNeurons, Khist);
    for idx = 1:Khist
      bj = bNow - Khist + idx;            % bin index for this slot
      if bj < 1
        feat(:,idx) = 0;
        continue;
      end
      % raw time bounds for this bin
      t1 = (bj-1)*binSize + 1;
      t2 = bj*binSize;
      % clamp to available data and to tNow
      t1 = max(1, t1);
      t2 = min([t2, tNow, Ttrial]);
      if t1 > t2
        feat(:,idx) = 0;
      else
        feat(:,idx) = sum(spikes(:, t1:t2), 2);
      end
    end
    phi = feat(:);
end