function plot_rmse_vs_time(Results_baseline, Results_improved)
% plot_rmse_vs_time  Plot RMSE (X, Y, and combined) vs decoding time bin
%                   with SEM shaded error bars, for two pipelines.
%
% Usage:
%   [RMSE_b, Results_b] = testFunction_for_students_MTb('PCA_Ridge');
%   [RMSE_i, Results_i] = testFunction_for_students_MTb('LDA_PCR');
%   plot_rmse_vs_time(Results_b, Results_i);

    dt      = 20;       % ms per bin (matches harness)
    t_start = 320;      % ms (first decode time)

    % --- 1. Find max number of time bins across all trials ---
    maxBins_b = max(cellfun(@(x) size(x,2), Results_baseline.decodedTrajectories));
    maxBins_i = max(cellfun(@(x) size(x,2), Results_improved.decodedTrajectories));
    maxBins   = max(maxBins_b, maxBins_i);

    % Time axis in ms
    timeAxis = t_start + (0:maxBins-1) * dt;   % 1 x maxBins

    % --- 2. Compute per-trial squared error at each time bin ---
    [errX_b, errY_b, err2D_b] = compute_bin_errors( ...
        Results_baseline.decodedTrajectories, ...
        Results_baseline.actualTrajectories, maxBins);

    [errX_i, errY_i, err2D_i] = compute_bin_errors( ...
        Results_improved.decodedTrajectories, ...
        Results_improved.actualTrajectories, maxBins);

    % --- 3. Compute RMSE and SEM at each bin (ignoring NaN = missing trials) ---
    [rmse_b,  sem_b]  = rmse_and_sem(err2D_b);
    [rmseX_b, semX_b] = rmse_and_sem(errX_b);
    [rmseY_b, semY_b] = rmse_and_sem(errY_b);

    [rmse_i,  sem_i]  = rmse_and_sem(err2D_i);
    [rmseX_i, semX_i] = rmse_and_sem(errX_i);
    [rmseY_i, semY_i] = rmse_and_sem(errY_i);

    % --- 4. Plot ---
    colors = struct( ...
        'base2D', [0.2  0.4  0.8], ...
        'impr2D', [0.8  0.2  0.2], ...
        'baseX',  [0.4  0.6  1.0], ...
        'imprX',  [1.0  0.4  0.4], ...
        'baseY',  [0.0  0.6  0.4], ...
        'imprY',  [0.9  0.6  0.0]);

    figure('Name', 'RMSE vs Decoding Time', 'Position', [100 100 1200 400]);

    % -- Subplot 1: Combined 2D RMSE --
    subplot(1,3,1); hold on;
    shaded_plot(timeAxis, rmse_b, sem_b, colors.base2D, 'Baseline');
    shaded_plot(timeAxis, rmse_i, sem_i, colors.impr2D, 'Improved');
    xline(320, 'k--', 'LineWidth', 1, 'Label', 't_{start}');
    xlabel('Decoding time (ms)');
    ylabel('RMSE (mm)');
    title('2D Euclidean RMSE vs Time');
    legend('Location','northeast');
    grid on; hold off;

    % -- Subplot 2: X RMSE --
    subplot(1,3,2); hold on;
    shaded_plot(timeAxis, rmseX_b, semX_b, colors.baseX, 'Baseline X');
    shaded_plot(timeAxis, rmseX_i, semX_i, colors.imprX, 'Improved X');
    xline(320, 'k--', 'LineWidth', 1);
    xlabel('Decoding time (ms)');
    ylabel('RMSE_X (mm)');
    title('X RMSE vs Time');
    legend('Location','northeast');
    grid on; hold off;

    % -- Subplot 3: Y RMSE --
    subplot(1,3,3); hold on;
    shaded_plot(timeAxis, rmseY_b, semY_b, colors.baseY, 'Baseline Y');
    shaded_plot(timeAxis, rmseY_i, semY_i, colors.imprY, 'Improved Y');
    xline(320, 'k--', 'LineWidth', 1);
    xlabel('Decoding time (ms)');
    ylabel('RMSE_Y (mm)');
    title('Y RMSE vs Time');
    legend('Location','northeast');
    grid on; hold off;

    sgtitle('RMSE vs Decoding Time (shaded = ±1 SEM across trials)');
end


%% -----------------------------------------------------------------------
%  HELPERS
%% -----------------------------------------------------------------------

function [errX, errY, err2D] = compute_bin_errors(decoded, actual, maxBins)
% Returns nTrials x maxBins matrices of squared errors (NaN = trial too short)
    nTrials = numel(decoded);
    errX  = NaN(nTrials, maxBins);
    errY  = NaN(nTrials, maxBins);
    err2D = NaN(nTrials, maxBins);

    for s = 1:nTrials
        dec = decoded{s};   % 2 x T_s
        act = actual{s};    % 2 x T_s
        T_s = size(dec, 2);
        bins = min(T_s, maxBins);

        errX(s,  1:bins) = (dec(1,1:bins) - act(1,1:bins)).^2;
        errY(s,  1:bins) = (dec(2,1:bins) - act(2,1:bins)).^2;
        err2D(s, 1:bins) = sum((dec(:,1:bins) - act(:,1:bins)).^2, 1);
    end
end


function [rmse_vec, sem_vec] = rmse_and_sem(sqErr)
% rmse_vec: 1 x maxBins RMSE ignoring NaN trials at each bin
% sem_vec:  1 x maxBins SEM of sqrt(sqErr) across valid trials
    nBins   = size(sqErr, 2);
    rmse_vec = zeros(1, nBins);
    sem_vec  = zeros(1, nBins);

    for b = 1:nBins
        col    = sqErr(:, b);
        valid  = col(~isnan(col));          % only trials that reach this bin
        n      = numel(valid);
        if n == 0
            rmse_vec(b) = NaN;
            sem_vec(b)  = NaN;
        else
            rmse_vec(b) = sqrt(mean(valid));
            % SEM of per-trial RMSE values
            perTrialRMSE = sqrt(valid);
            sem_vec(b)   = std(perTrialRMSE) / sqrt(n);
        end
    end
end


function shaded_plot(t, mu, sem, col, label)
% Plot mean line with ±1 SEM shaded band
    valid = ~isnan(mu) & ~isnan(sem);
    tv    = t(valid);
    muv   = mu(valid);
    semv  = sem(valid);

    upper = muv + semv;
    lower = muv - semv;

    fill([tv, fliplr(tv)], [upper, fliplr(lower)], col, ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none');
    plot(tv, muv, 'Color', col, 'LineWidth', 2, 'DisplayName', label);
end