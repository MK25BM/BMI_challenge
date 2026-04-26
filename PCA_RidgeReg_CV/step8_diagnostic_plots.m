% step8_diagnostic_plots.m
% =========================================================================
% Step 8: Diagnostic plots using best hyperparameters on held-out test set
%
% Trains on the full CV pool using optimal hyperparameters from Steps 5–7,
% evaluates on the test_set, and generates 4 diagnostic figures.
%
% Figures
%   1. Per-fold RMSE bar chart (at best lambda)
%   2. RMSE per reaching angle (all 8 directions)
%   3. RMSE over time (320, 340, ... ms)
%   4. Predicted vs Actual trajectories (2x4 grid, one per direction)
% =========================================================================

clear; clc;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

%% Load all results
load(fullfile(script_dir, 'cv_splits.mat'),             'cv_pool', 'test_set');
load(fullfile(script_dir, 'fold_indices.mat'),           'fold_indices', 'K');
load(fullfile(script_dir, 'lambda_cv_results.mat'),      'best_lambda', 'val_rmse_cv', 'lambda_grid');
load(fullfile(script_dir, 'pcaVarFrac_cv_results.mat'),  'best_pcaVarFrac');
load(fullfile(script_dir, 'Khist_cv_results.mat'),       'best_Khist');

fprintf('Best hyperparameters:\n');
fprintf('  lambda      = %g\n', best_lambda);
fprintf('  pcaVarFrac  = %g\n', best_pcaVarFrac);
fprintf('  Khist       = %d\n', best_Khist);

%% Train on full CV pool
fprintf('\nTraining on full CV pool...\n');
mp = cv_train_model(cv_pool, best_lambda, best_pcaVarFrac, best_Khist);

%% ---- Figure 1: Per-fold RMSE bar chart -----------------------------------
% Find row in val_rmse_cv corresponding to best_lambda
[~, best_li] = min(abs(lambda_grid - best_lambda));
fold_rmse = val_rmse_cv(best_li, :);

fig1 = figure('Name', 'Per-fold RMSE', 'Visible', 'on');
bar(1:K, fold_rmse, 'FaceColor', [0.3 0.6 0.9]);
hold on;
yline(mean(fold_rmse), 'r--', 'LineWidth', 2, ...
    'DisplayName', sprintf('Mean = %.2f mm', mean(fold_rmse)));
xlabel('Fold', 'FontSize', 12);
ylabel('RMSE (mm)', 'FontSize', 12);
title(sprintf('Per-fold RMSE at best \\lambda=%g', best_lambda), 'FontSize', 13);
legend('Location', 'best');
grid on;
savefig(fig1, fullfile(script_dir, 'Fig_per_fold_rmse.fig'));
saveas(fig1,  fullfile(script_dir, 'Fig_per_fold_rmse.png'));

%% ---- Evaluate test set ---------------------------------------------------
fprintf('Evaluating on test set...\n');

[nTest, nDir] = size(test_set);
binSize = mp.binSize;
tStart  = mp.tStart;

angles_deg = (0:nDir-1) * 45;

% For per-angle RMSE
rmse_per_dir = zeros(1, nDir);

% For RMSE over time — find common time grid
Tmax = 0;
for tr = 1:nTest
    for d = 1:nDir
        Tmax = max(Tmax, size(test_set(tr,d).spikes, 2));
    end
end
time_grid    = tStart : binSize : Tmax;
nTime        = numel(time_grid);
sq_sum_time  = zeros(1, nTime);
cnt_time     = zeros(1, nTime);

% For trajectory plot — store one example trial per direction
rng(42);
selected_trial_idx = randi(nTest);
traj_true_x  = cell(1, nDir);
traj_true_y  = cell(1, nDir);
traj_pred_x  = cell(1, nDir);
traj_pred_y  = cell(1, nDir);

total_sq = 0;
total_n  = 0;

for d = 1:nDir
    sq_dir = 0;
    n_dir  = 0;

    for tr = 1:nTest
        sp = test_set(tr,d).spikes;
        hp = test_set(tr,d).handPos;
        T  = size(sp, 2);

        if T < tStart; continue; end

        clear positionEstimator; %#ok<CLFUNC>

        t_vec   = tStart : binSize : T;
        pred_x  = zeros(1, numel(t_vec));
        pred_y  = zeros(1, numel(t_vec));
        true_x  = zeros(1, numel(t_vec));
        true_y  = zeros(1, numel(t_vec));

        for ti = 1:numel(t_vec)
            t = t_vec(ti);
            test_data.spikes  = sp(:, 1:t);
            test_data.trialId = test_set(tr,d).trialId;

            [xp, yp] = positionEstimator(test_data, mp);
            pred_x(ti) = xp;
            pred_y(ti) = yp;

            if t <= size(hp,2)
                true_x(ti) = hp(1, t);
                true_y(ti) = hp(2, t);

                err2 = (xp - hp(1,t))^2 + (yp - hp(2,t))^2;
                sq_dir   = sq_dir + err2;
                n_dir    = n_dir  + 1;
                total_sq = total_sq + err2;
                total_n  = total_n  + 1;

                % Time-resolved
                [~, tidx] = min(abs(time_grid - t));
                sq_sum_time(tidx) = sq_sum_time(tidx) + err2;
                cnt_time(tidx)    = cnt_time(tidx) + 1;
            end
        end

        % Store trajectory for selected_trial_idx
        if tr == selected_trial_idx
            traj_true_x{d} = true_x;
            traj_true_y{d} = true_y;
            traj_pred_x{d} = pred_x;
            traj_pred_y{d} = pred_y;
        end
    end

    if n_dir > 0
        rmse_per_dir(d) = sqrt(sq_dir / n_dir);
    end
end

final_rmse = sqrt(total_sq / total_n);
fprintf('\nFinal test RMSE: %.4f mm\n', final_rmse);

%% ---- Figure 2: RMSE per reaching angle -----------------------------------
fig2 = figure('Name', 'RMSE per angle', 'Visible', 'on');
bar(angles_deg, rmse_per_dir, 'FaceColor', [0.4 0.7 0.4]);
hold on;
yline(mean(rmse_per_dir), 'r--', 'LineWidth', 2, ...
    'DisplayName', sprintf('Mean = %.2f mm', mean(rmse_per_dir)));
xlabel('Reaching Angle (degrees)', 'FontSize', 12);
ylabel('RMSE (mm)', 'FontSize', 12);
title('Decoding RMSE per Reaching Direction (Test Set)', 'FontSize', 13);
xticks(angles_deg);
legend('Location', 'best');
grid on;
savefig(fig2, fullfile(script_dir, 'Fig_rmse_per_angle.fig'));
saveas(fig2,  fullfile(script_dir, 'Fig_rmse_per_angle.png'));

%% ---- Figure 3: RMSE over time -------------------------------------------
rmse_over_time = sqrt(sq_sum_time ./ max(cnt_time, 1));
rmse_over_time(cnt_time == 0) = NaN;

fig3 = figure('Name', 'RMSE over time', 'Visible', 'on');
plot(time_grid, rmse_over_time, 'b-o', 'LineWidth', 2);
xlabel('Time (ms)', 'FontSize', 12);
ylabel('RMSE (mm)', 'FontSize', 12);
title('Decoding RMSE vs Time into Trial (Test Set)', 'FontSize', 13);
grid on;
savefig(fig3, fullfile(script_dir, 'Fig_rmse_over_time.fig'));
saveas(fig3,  fullfile(script_dir, 'Fig_rmse_over_time.png'));

%% ---- Figure 4: Predicted vs Actual trajectories -------------------------
fig4 = figure('Name', 'Trajectories', 'Visible', 'on');
for d = 1:nDir
    subplot(2, 4, d); hold on;
    if ~isempty(traj_true_x{d})
        plot(traj_true_x{d}, traj_true_y{d}, 'b-',  'LineWidth', 2, 'DisplayName', 'Actual');
        plot(traj_pred_x{d}, traj_pred_y{d}, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Decoded');
    end
    title(sprintf('%d°', angles_deg(d)));
    axis equal; grid on;
    if d == 1; legend('Location', 'best'); end
end
sgtitle(sprintf('Predicted vs Actual Trajectories (Test Trial %d)', selected_trial_idx), 'FontSize', 13);
savefig(fig4, fullfile(script_dir, 'Fig_trajectories.fig'));
saveas(fig4,  fullfile(script_dir, 'Fig_trajectories.png'));

fprintf('All figures saved.\n');
