% step5_tune_r.m
% =========================================================================
% Step 5: Tune PCR components r via K-Fold Cross-Validation
%
% For each r in the grid, trains on K-1 folds and evaluates on the
% held-out fold. Also records training RMSE for the bias-variance plot.
%
% Fixed: M_pca = 170, M_lda = 7
%
% Outputs
%   r_cv_results.mat       — r_grid, val_rmse_cv, train_rmse_cv, optimal_r
%   fig_r_vs_rmse.png      — errorbar plot: mean±std val RMSE vs r
%   fig_r_bias_variance.png — train vs val RMSE (bias-variance tradeoff)
% =========================================================================

clear; clc;

%% Paths
script_dir  = fileparts(mfilename('fullpath'));
shared_dir  = fullfile(script_dir, '..', 'shared_cv_utils');
addpath(script_dir);   % ensure local cv_train_model / cv_evaluate_fold are found

%% Load data
load(fullfile(shared_dir, 'cv_splits.mat'),   'cv_pool');
load(fullfile(shared_dir, 'fold_indices.mat'), 'fold_indices', 'K');

[nPool, ~] = size(cv_pool);

%% Hyperparameter grid
r_grid = [2, 5, 10, 15, 20, 30, 40, 50];
nR     = numel(r_grid);

% Fixed hyperparameters for this step
M_pca_fixed = 170;
M_lda_fixed = 7;

%% Pre-allocate result matrices
val_rmse_cv   = zeros(nR, K);
train_rmse_cv = zeros(nR, K);

fprintf('Running K-Fold CV for r tuning (%d r values x %d folds)...\n', nR, K);
fprintf('Fixed: M_pca=%d, M_lda=%d\n\n', M_pca_fixed, M_lda_fixed);

for ri = 1:nR
    r_val = r_grid(ri);
    fprintf('  r = %-4d  [ ', r_val);

    for k = 1:K
        %% Build train/val sets for this fold
        val_mask   = fold_indices{k};   % nPool x nDir logical
        val_rows   = any(val_mask, 2);  % rows with at least one val entry

        val_data   = cv_pool(val_rows,  :);
        train_data = cv_pool(~val_rows, :);

        %% Train
        mp = cv_train_model(train_data, r_val, M_pca_fixed, M_lda_fixed);

        %% Validate
        val_rmse_cv(ri, k)   = cv_evaluate_fold(val_data,   mp);
        train_rmse_cv(ri, k) = cv_evaluate_fold(train_data, mp);

        fprintf('k%d(%.1f/%.1f) ', k, val_rmse_cv(ri,k), train_rmse_cv(ri,k));
    end
    fprintf(']\n');
end

%% Optimal r
mean_val_rmse = mean(val_rmse_cv, 2);
[~, best_ri]  = min(mean_val_rmse);
optimal_r     = r_grid(best_ri);

fprintf('\nr CV Results:\n');
for ri = 1:nR
    fprintf('  r=%-4d  mean_val_RMSE=%.4f   std=%.4f\n', ...
        r_grid(ri), mean_val_rmse(ri), std(val_rmse_cv(ri,:)));
end
fprintf('Optimal r: %d  (mean CV RMSE = %.4f mm)\n', optimal_r, mean_val_rmse(best_ri));

%% Save results
save(fullfile(script_dir, 'r_cv_results.mat'), ...
    'r_grid', 'val_rmse_cv', 'train_rmse_cv', 'optimal_r');
fprintf('\nSaved r_cv_results.mat\n');

%% ---- Figure 1: r vs CV RMSE (errorbar) ----------------------------------
mean_val = mean(val_rmse_cv, 2);
std_val  = std(val_rmse_cv,  0, 2);

fig1 = figure('Name', 'r CV RMSE', 'Visible', 'on');
errorbar(r_grid, mean_val, std_val, 'o-', 'LineWidth', 2, ...
    'MarkerFaceColor', [0.2 0.4 0.8]);
hold on;
xline(nPool - 1, 'r--', 'LineWidth', 1.5, ...
    'Label', 'Current (overfit risk)', 'LabelVerticalAlignment', 'bottom');
xlabel('r (PCR components)', 'FontSize', 12);
ylabel('Mean CV RMSE (mm)', 'FontSize', 12);
title('PCR Components r: CV RMSE vs r', 'FontSize', 13);
legend({'CV RMSE ± std', sprintf('Current r=%d', nPool-1)}, 'Location', 'best');
grid on;
saveas(fig1, fullfile(script_dir, 'fig_r_vs_rmse.png'));

%% ---- Figure 2: Bias-Variance tradeoff ------------------------------------
mean_train = mean(train_rmse_cv, 2);

fig2 = figure('Name', 'r Bias-Variance', 'Visible', 'on');
plot(r_grid, mean_train, 'b-o', 'LineWidth', 2, 'DisplayName', 'Train RMSE');
hold on;
plot(r_grid, mean_val,   'r-o', 'LineWidth', 2, 'DisplayName', 'Val RMSE');
xlabel('r (PCR components)', 'FontSize', 12);
ylabel('RMSE (mm)', 'FontSize', 12);
title('Bias-Variance Tradeoff: r (PCR components)', 'FontSize', 13);
legend('Location', 'best');
grid on;
saveas(fig2, fullfile(script_dir, 'fig_r_bias_variance.png'));

fprintf('Figures saved.\n');
