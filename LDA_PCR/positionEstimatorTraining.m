function modelParameters = positionEstimatorTraining(training_data)
% Train per-direction, per-time linear decoders with PCA 

    % Hyperparameters
    binSize = 20;
    tStart = 320;
    Khist = 11;
    lambda = 4;
    pcaVarFrac = 0.1;

    [nTrials, nDir] = size(training_data);
    nNeurons = size(training_data(1,1).spikes, 1);
    
    % Find max time
    Tmax = 0;
    for tr = 1:nTrials
        for d = 1:nDir
            Tmax = max(Tmax, size(training_data(tr,d).spikes, 2));
        end
    end
    times = tStart : binSize : Tmax;
    nTime = numel(times);

    % (A) Direction templates
    dirMeanCounts = zeros(nNeurons, nDir);
    for d = 1:nDir
        for tr = 1:nTrials
            sp = training_data(tr,d).spikes;
            tUse = min(tStart, size(sp,2));
            dirMeanCounts(:,d) = dirMeanCounts(:,d) + sum(sp(:,1:tUse), 2);
        end
        dirMeanCounts(:,d) = dirMeanCounts(:,d) / nTrials;
    end

    % (B) Collect all features for PCA
    Fraw = nNeurons * Khist;
    PhiAll = [];
    for d = 1:nDir
        for ti = 1:nTime
            tNow = times(ti);
            for tr = 1:nTrials
                sp = training_data(tr,d).spikes;
                hp = training_data(tr,d).handPos;
                if tNow <= size(sp,2) && tNow <= size(hp,2)
                    phi = makeFeature(sp, tNow, binSize, Khist);
                    PhiAll = [PhiAll, phi];
                end
            end
        end
    end

    if isempty(PhiAll)
        error('No aligned samples found');
    end

    % (C) Manual PCA using SVD 
    muPhi = mean(PhiAll, 2);  % Mean of each feature
    PhiCentered = bsxfun(@minus, PhiAll, muPhi);  % Center data
    
    % SVD for PCA
    [U, S, ~] = svd(PhiCentered, 'econ');
    varExplained = (diag(S).^2) / sum(diag(S).^2);  % Variance explained
    cumVar = cumsum(varExplained);
    nComp = find(cumVar >= pcaVarFrac, 1, 'first');
    if isempty(nComp) || nComp > size(U,2)
        nComp = min(size(U,2), Fraw);
    end
    
    Ured = U(:, 1:nComp);  % Keep top nComp components

    % (D) Per-(direction,time) ridge training
    W = cell(nDir, nTime);
    F = nComp + 1;
    Ireg = eye(F);
    Ireg(end,end) = 0;  % Don't regularize intercept

    for d = 1:nDir
        for ti = 1:nTime
            tNow = times(ti);
            Xphi = [];
            Y = [];
            
            for tr = 1:nTrials
                sp = training_data(tr,d).spikes;
                hp = training_data(tr,d).handPos;
                if tNow <= size(sp,2) && tNow <= size(hp,2)
                    phi = makeFeature(sp, tNow, binSize, Khist);
                    Xphi = [Xphi, phi];
                    Y = [Y, hp(1:2, tNow)];
                end
            end

            N = size(Xphi, 2);
            if N == 0
                W{d,ti} = zeros(2, F);
                continue;
            end

            % Project to PCA space
            Scores = Ured' * (bsxfun(@minus, Xphi, muPhi));
            Xreg = [Scores; ones(1,N)];

            % Ridge regression
            W{d,ti} = (Y * Xreg.') / (Xreg * Xreg.' + lambda * Ireg);
        end
    end

    % Pack output
    modelParameters.W = W;
    modelParameters.dirMeanCounts = dirMeanCounts;
    modelParameters.times = times;
    modelParameters.binSize = binSize;
    modelParameters.Khist = Khist;
    modelParameters.tStart = tStart;
    modelParameters.lambda = lambda;
    modelParameters.PCA = struct('mu', muPhi, 'U', Ured, 'nComp', nComp);

end

function phi = makeFeature(spikes, tNow, binSize, Khist)
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
