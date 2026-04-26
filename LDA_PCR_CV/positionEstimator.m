function [x, y] = positionEstimator(test_data, modelParameters)
% positionEstimator  Decode hand position from incoming spikes.
%
% Inputs
%   test_data.spikes         - neuron x time spike matrix (1 : t ms)
%   test_data.startHandPos   - [x0; y0] hand position at trial start
%   modelParameters          - struct from positionEstimatorTraining
%
% Outputs
%   x, y  - decoded hand position at time t (mm)

    mp = modelParameters;
    sp = test_data.spikes;
    t  = size(sp, 2);   % current time (ms)

    % -- Fallback: before decoding window opens ----------------------------
    if t < mp.t_reg(1)
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        return
    end

    % -- 1. Direction classification ---------------------------------------
    % Pick the latest classification checkpoint <= t
    t_class = mp.t_class;
    ci      = find(t_class <= t, 1, 'last');
    if isempty(ci), ci = 1; end
    cp = mp.classParams(ci);

    % Compute average firing rate in dt_class-wide bins up to t_class(ci)
    dt  = mp.dt_class;
    T_c = t_class(ci);
    nBins    = floor(T_c / dt);
    nNeurons = size(sp, 1);
    feat_cls = zeros(nNeurons * nBins, 1);
    for bi = 1:nBins
        t1 = (bi-1)*dt + 1;
        t2 = min(bi*dt, size(sp,2));
        if t1 <= t2
            feat_cls((bi-1)*nNeurons + (1:nNeurons)) = ...
                sum(sp(:, t1:t2), 2) / dt;
        end
    end

    % Project to LDA space, find nearest centroid
    z     = cp.Wopt' * (feat_cls - cp.mu);
    dists = sum((cp.centroids - z).^2, 1);
    [~, dir_est] = min(dists);

    % -- 2. PCR position decoding ------------------------------------------
    % Find nearest regression time index
    t_reg = mp.t_reg;
    [~, ri] = min(abs(t_reg - t));
    rp = mp.regressors(ri, dir_est);

    % Compute spike-count features for regression (dt_reg bins up to t)
    dt2   = mp.dt_reg;
    nBins2 = floor(t / dt2);
    feat_reg = zeros(nNeurons * nBins2, 1);
    for bi = 1:nBins2
        t1 = (bi-1)*dt2 + 1;
        t2 = min(bi*dt2, size(sp,2));
        if t1 <= t2
            feat_reg((bi-1)*nNeurons + (1:nNeurons)) = ...
                sum(sp(:, t1:t2), 2) / dt2;
        end
    end

    % Trim / pad feat_reg to match the size expected by U
    nExpected = size(rp.U, 1);
    if numel(feat_reg) >= nExpected
        feat_reg = feat_reg(1:nExpected);
    else
        feat_reg = [feat_reg; zeros(nExpected - numel(feat_reg), 1)];
    end

    % PCR prediction: project to PC space, apply learned coefficients
    w = rp.U' * (feat_reg - rp.mu_f);   % r x 1 score
    x = rp.mu_x + rp.bx' * (feat_reg - rp.mu_f);
    y = rp.mu_y + rp.by' * (feat_reg - rp.mu_f);
end
