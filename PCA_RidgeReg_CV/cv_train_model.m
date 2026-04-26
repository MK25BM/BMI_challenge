function modelParameters = cv_train_model(training_data, lambda, pcaVarFrac, Khist)
% cv_train_model  Train PCA + Ridge Regression decoder with given hyperparameters.
%
%   modelParameters = cv_train_model(training_data, lambda, pcaVarFrac, Khist)
%
%   Refactored from PCA_RidgeReg/positionEstimatorTraining.m.
%   Logic is identical; hyperparameters are passed as arguments instead of
%   being hardcoded.
%
%   Inputs
%   ------
%   training_data : (nTrials x nDir) struct with fields .spikes and .handPos
%   lambda        : ridge regularisation strength
%   pcaVarFrac    : fraction of variance retained by PCA  (0 < pcaVarFrac <= 1)
%   Khist         : number of 20 ms spike-history bins
%
%   Output
%   ------
%   modelParameters : struct consumed by positionEstimator.m

    % Fixed hyperparameters (not tuned in this CV pipeline)
    binSize = 20;
    tStart  = 320;

    [nTrials, nDir] = size(training_data);
    nNeurons = size(training_data(1,1).spikes, 1);

    %% Find max time across all trials
    Tmax = 0;
    for tr = 1:nTrials
        for d = 1:nDir
            Tmax = max(Tmax, size(training_data(tr,d).spikes, 2));
        end
    end
    times = tStart : binSize : Tmax;
    nTime = numel(times);

    %% (A) Direction templates
    dirMeanCounts = zeros(nNeurons, nDir);
    for d = 1:nDir
        for tr = 1:nTrials
            sp   = training_data(tr,d).spikes;
            tUse = min(tStart, size(sp,2));
            dirMeanCounts(:,d) = dirMeanCounts(:,d) + sum(sp(:,1:tUse), 2);
        end
        dirMeanCounts(:,d) = dirMeanCounts(:,d) / nTrials;
    end

    %% (B) Collect all features for PCA
    Fraw   = nNeurons * Khist;
    PhiAll = [];
    for d = 1:nDir
        for ti = 1:nTime
            tNow = times(ti);
            for tr = 1:nTrials
                sp = training_data(tr,d).spikes;
                hp = training_data(tr,d).handPos;
                if tNow <= size(sp,2) && tNow <= size(hp,2)
                    phi    = makeFeature(sp, tNow, binSize, Khist);
                    PhiAll = [PhiAll, phi]; %#ok<AGROW>
                end
            end
        end
    end

    if isempty(PhiAll)
        error('cv_train_model: no aligned samples found.');
    end

    %% (C) PCA via SVD
    muPhi       = mean(PhiAll, 2);
    PhiCentered = bsxfun(@minus, PhiAll, muPhi);

    [U, S, ~]   = svd(PhiCentered, 'econ');
    varExplained = (diag(S).^2) / sum(diag(S).^2);
    cumVar       = cumsum(varExplained);
    nComp        = find(cumVar >= pcaVarFrac, 1, 'first');
    if isempty(nComp) || nComp > size(U,2)
        nComp = min(size(U,2), Fraw);
    end
    Ured = U(:, 1:nComp);

    %% (D) Per-(direction, time) ridge regression
    W    = cell(nDir, nTime);
    F    = nComp + 1;
    Ireg = eye(F);
    Ireg(end,end) = 0;  % Do not regularise intercept

    for d = 1:nDir
        for ti = 1:nTime
            tNow = times(ti);
            Xphi = [];
            Y    = [];

            for tr = 1:nTrials
                sp = training_data(tr,d).spikes;
                hp = training_data(tr,d).handPos;
                if tNow <= size(sp,2) && tNow <= size(hp,2)
                    phi  = makeFeature(sp, tNow, binSize, Khist);
                    Xphi = [Xphi, phi]; %#ok<AGROW>
                    Y    = [Y, hp(1:2, tNow)]; %#ok<AGROW>
                end
            end

            N = size(Xphi, 2);
            if N == 0
                W{d,ti} = zeros(2, F);
                continue;
            end

            % Project to PCA space and add intercept
            Scores = Ured' * (bsxfun(@minus, Xphi, muPhi));
            Xreg   = [Scores; ones(1,N)];

            % Ridge regression: W = Y * X' * inv(X*X' + lambda*I)
            W{d,ti} = (Y * Xreg.') / (Xreg * Xreg.' + lambda * Ireg);
        end
    end

    %% Pack output
    modelParameters.W             = W;
    modelParameters.dirMeanCounts = dirMeanCounts;
    modelParameters.times         = times;
    modelParameters.binSize       = binSize;
    modelParameters.Khist         = Khist;
    modelParameters.tStart        = tStart;
    modelParameters.lambda        = lambda;
    modelParameters.PCA           = struct('mu', muPhi, 'U', Ured, 'nComp', nComp);

end

% -------------------------------------------------------------------------
function phi = makeFeature(spikes, tNow, binSize, Khist)
% makeFeature  Build spike-history feature vector for a single timepoint.
    [nNeurons, Ttrial] = size(spikes);
    bNow = floor(tNow / binSize);
    feat = zeros(nNeurons, Khist);

    for idx = 1:Khist
        bj = bNow - Khist + idx;
        if bj >= 1
            t1 = max(1, (bj-1)*binSize + 1);
            t2 = min(tNow, min(Ttrial, bj*binSize));
            if t1 <= t2
                feat(:,idx) = sum(spikes(:, t1:t2), 2);
            end
        end
    end

    phi = feat(:);
end
