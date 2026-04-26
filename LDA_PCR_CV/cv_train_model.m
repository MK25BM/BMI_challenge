function modelParameters = cv_train_model(trainingData, r, M_pca, M_lda)
% cv_train_model  Train LDA direction classifier + PCR position decoder.
%
%   modelParameters = cv_train_model(trainingData, r, M_pca, M_lda)
%
%   Refactored from LDA_PCR/positionEstimatorTraining.m.
%   Logic is identical; hyperparameters are passed as arguments instead of
%   being hardcoded.
%
% Pipeline
% --------
%  1. Direction classification  : PCA -> LDA, nearest-centroid in LDA space.
%     Trained at t = 320, 400, 480, 560 ms (updated as more spikes arrive).
%  2. Position decoding         : PCR (Principal Component Regression)
%     One regressor per (direction, time) pair, dt = 20 ms.
%
% Inputs
% ------
%   trainingData : (nTrials x nDir) struct with fields .spikes and .handPos
%   r            : number of PCR components (tunable; replaces nTrials-1)
%   M_pca        : PCA dimensions fed into LDA (tunable)
%   M_lda        : LDA dimensions kept (tunable; max = nClasses-1 = 7)
%
% Output
% ------
%   modelParameters : struct consumed by positionEstimator.m

% -- Fixed Hyperparameters ------------------------------------------------
dt_class  = 80;                    % classification bin width (ms)
t_class   = 320:dt_class:560;     % times at which classifier is retrained
dt_reg    = 20;                    % regression bin width (ms)
T_end     = 560;                   % last time point used
t_reg     = 320:dt_reg:T_end;     % regression time grid

% -- Clamp r to avoid exceeding matrix rank --------------------------------
r = min(r, size(trainingData, 1) - 1);

% -- 1. Direction Classifier ----------------------------------------------
% Organise spike-count features: rows = feature×time bins, cols = trials
[F_cls, labels, t_tag] = organise_features(trainingData, dt_class, t_class(end));

classParams = struct();
for ti = 1:numel(t_class)
    T     = t_class(ti);
    X_sub = F_cls(t_tag <= T, :);   % row-subset: only bins up to time T

    % PCA, scatter matrices, and LDA all computed on X_sub so dimensions match.
    nComp = min(M_pca, size(X_sub, 2) - 1);   % can't exceed rank
    [~, mu_sub, Wpca_sub, ~] = pca_model(X_sub, nComp);
    [Sb_sub, Sw_sub]         = scatter_matrices(X_sub, labels);
    Wopt = lda_from_pca(Wpca_sub, nComp, M_lda, Sb_sub, Sw_sub);

    % Project all samples and store per-class centroids
    Z = Wopt' * (X_sub - mu_sub);
    centroids = zeros(M_lda, 8);
    for k = 1:8
        centroids(:,k) = mean(Z(:, labels == k), 2);
    end

    classParams(ti).Wopt      = Wopt;
    classParams(ti).mu        = mu_sub;
    classParams(ti).centroids = centroids;
end

% -- 2. PCR Position Decoder ----------------------------------------------
[F_reg, reg_labels, reg_t] = organise_features(trainingData, dt_reg, T_end);
[~, ~, x_all, y_all, ~, ~] = extract_hand_pos(trainingData);

% Align hand positions to the regression time grid
x_grid = x_all(:, t_reg, :);   % nTrials x nTimes x 8
y_grid = y_all(:, t_reg, :);

regressors = struct();
for ai = 1:8
    for ti = 1:numel(t_reg)
        T   = t_reg(ti);
        Fti = F_reg(reg_t <= T, reg_labels == ai);   % features up to T, this direction

        mu_x = mean(x_grid(:, ti, ai));
        mu_y = mean(y_grid(:, ti, ai));
        dx   = x_grid(:, ti, ai) - mu_x;
        dy   = y_grid(:, ti, ai) - mu_y;

        % PCR: project features to r PC directions, then OLS
        [~, mu_f, U, ~] = pca_model(Fti, r);
        W = U' * (Fti - mu_f);          % r x nTrials scores
        WWT_inv = (W * W')^(-1);

        regressors(ti, ai).mu_f  = mu_f;
        regressors(ti, ai).U     = U;
        regressors(ti, ai).bx    = U * WWT_inv * W * dx;
        regressors(ti, ai).by    = U * WWT_inv * W * dy;
        regressors(ti, ai).mu_x  = mu_x;
        regressors(ti, ai).mu_y  = mu_y;
    end
end

% -- Pack model -----------------------------------------------------------
modelParameters.classParams  = classParams;
modelParameters.t_class      = t_class;
modelParameters.regressors   = regressors;
modelParameters.t_reg        = t_reg;
modelParameters.dt_reg       = dt_reg;
modelParameters.dt_class     = dt_class;
modelParameters.r             = r;
modelParameters.avg_traj     = average_trajectory(trainingData);
end


%% -----------------------------------------------------------------------
%  LOCAL HELPERS
%% -----------------------------------------------------------------------

function [X, labels, t_tag] = organise_features(data, dt, T_end)
% Build feature matrix X (nFeatures x nSamples).
% Each column is one (trial, direction) sample.
% t_tag marks which time bin each feature row belongs to.
% labels gives the direction class (1-8) for each column.
    nNeurons = size(data(1,1).spikes, 1);
    T        = dt:dt:T_end;
    nBins    = numel(T);
    nTrials  = size(data, 1);
    nDirs    = size(data, 2);

    % Raw 4-D array: neuron x trial x direction x bin
    X0 = zeros(nNeurons, nTrials, nDirs, nBins);
    for bi = 1:nBins
        t1 = dt*(bi-1)+1;
        t2 = dt*bi;
        for k = 1:nDirs
            for n = 1:nTrials
                X0(:,n,k,bi) = sum(data(n,k).spikes(:, t1:t2), 2) / dt;
            end
        end
    end

    % Stack bins vertically: (nNeurons*nBins) x nTrials x nDirs
    X1    = zeros(nNeurons*nBins, nTrials, nDirs);
    t_tag = zeros(1, nNeurons*nBins);
    for bi = 1:nBins
        rows = (bi-1)*nNeurons + (1:nNeurons);
        X1(rows,:,:) = X0(:,:,:,bi);
        t_tag(rows)  = T(bi);
    end

    % Flatten directions into columns
    X      = zeros(nNeurons*nBins, nTrials*nDirs);
    labels = zeros(1, nTrials*nDirs);
    for k = 1:nDirs
        cols         = (k-1)*nTrials + (1:nTrials);
        X(:, cols)   = X1(:,:,k);
        labels(cols) = k;
    end
end


function [N, mu, U, L] = pca_model(X, p)
% PCA via covariance eigen-decomposition.
% Returns N=nSamples, mu=mean, U=top-p eigenvectors (data-space), L=eigenvalues.
    N  = size(X, 2);
    mu = mean(X, 2);
    A  = X - mu;
    S  = (A' * A) / N;          % sample-space covariance (faster when nFeatures >> N)
    [V, L]   = eig(S);
    p        = min(p, size(V,2));
    [~, idx] = maxk(diag(L), p);
    U        = A * V(:,idx);    % map back to feature space
    U        = U ./ sqrt(sum(U.^2));   % unit-normalise columns
    L        = L(idx,idx);
end


function [Sb, Sw] = scatter_matrices(X, labels)
% Between-class (Sb) and within-class (Sw) scatter matrices.
    mu_all = mean(X, 2);
    classes = unique(labels);
    mu_cls  = zeros(size(X,1), numel(classes));
    for ci = 1:numel(classes)
        mu_cls(:,ci) = mean(X(:, labels == classes(ci)), 2);
    end
    St = (X - mu_all) * (X - mu_all)';
    Sb = (mu_cls - mu_all) * (mu_cls - mu_all)';
    Sw = St - Sb;
end


function Wopt = lda_from_pca(Wpca, M_pca, M_lda, Sb, Sw)
% Build PCA-then-LDA projection matrix (feature-space -> M_lda dims).
    Wp = Wpca(:, 1:M_pca);
    Sw_pca = Wp' * Sw * Wp;
    Sb_pca = Wp' * Sb * Wp;
    [Wlda, Llda] = eig(Sw_pca \ Sb_pca);
    [~, idx] = maxk(diag(Llda), M_lda);
    Wopt = Wp * Wlda(:,idx);
end


function avg = average_trajectory(data)
% Per-direction average hand trajectory (x,y), padded to equal length.
    nDirs   = size(data, 2);
    nTrials = size(data, 1);

    % Find global max length
    Tmax = 0;
    for k = 1:nDirs
        for n = 1:nTrials
            Tmax = max(Tmax, size(data(n,k).spikes, 2));
        end
    end

    avg(nDirs).handPos = [];
    for k = 1:nDirs
        traj = zeros(2, Tmax);
        for n = 1:nTrials
            hp  = data(n,k).handPos(1:2,:);
            len = size(hp,2);
            % Pad last position
            if len < Tmax
                hp = [hp, repmat(hp(:,end), 1, Tmax-len)];
            end
            traj = traj + hp(:,1:Tmax);
        end
        avg(k).handPos = traj / nTrials;
    end
end


function [mx, my, x, y, lengths, in_data] = extract_hand_pos(data)
% Return per-trial hand positions as nTrials x nTime x 8 arrays (x and y).
    nDirs   = 8;
    nTrials = size(data, 1);

    % Pass 1: find global max length
    Tmax = 0;
    lengths = zeros(nTrials, nDirs);
    for k = 1:nDirs
        for n = 1:nTrials
            l = size(data(n,k).handPos, 2);
            lengths(n,k) = l;
            Tmax = max(Tmax, l);
        end
    end

    x = zeros(nTrials, Tmax, nDirs);
    y = zeros(nTrials, Tmax, nDirs);
    in_data = zeros(nTrials, Tmax, nDirs);

    for k = 1:nDirs
        for n = 1:nTrials
            hp  = data(n,k).handPos;
            len = size(hp,2);
            x(n, 1:len, k) = hp(1,:);
            y(n, 1:len, k) = hp(2,:);
            % Pad with last value
            if len < Tmax
                x(n, len+1:end, k) = hp(1,end);
                y(n, len+1:end, k) = hp(2,end);
            end
            in_data(n, 1:len, k) = 1;
        end
    end

    mx = squeeze(mean(x, 1))';   % 8 x Tmax
    my = squeeze(mean(y, 1))';
end
