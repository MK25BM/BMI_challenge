% step6_tune_M_pca.m
% =========================================================================
% Step 6: Tune PCA dimensions fed into LDA (M_pca) via K-Fold CV
%
% Uses optimal_r from Step 5. Fixes M_lda = 7.
%
% Outputs
%   M_pca_cv_results.mat        — M_pca_grid, val_rmse_cv, train_rmse_cv, optimal_M_pca
%   fig_M_pca_vs_rmse.png       — errorbar plot: mean±std val RMSE vs M_pca
%   fig_M_pca_bias_variance.png — train vs val RMSE (bias-variance tradeoff)
% =========================================================================

clear; clc;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
shared_dir = fullfile(script_dir, '..', 'shared_cv_utils');
addpath(script_dir);

%% Load data and previous results
load(fullfile(shared_dir, 'cv_splits.mat'),   'cv_pool');
load(fullfile(shared_dir, 'fold_indices.mat'), 'fold_indices', 'K');
load(fullfile(script_dir, 'r_cv_results.mat'), 'optimal_r');

fprintf('Using optimal_r = %d (from Step 5)\n', optimal_r);

%% Hyperparameter grid
M_pca_grid  = [10, 20, 40, 60, 80, 100, 130, 170];
nM          = numel(M_pca_grid);

% Fixed hyperparameters for this step
M_lda_fixed = 7;

%% Pre-allocate result matrices
val_rmse_cv   = zeros(nM, K);
train_rmse_cv = zeros(nM, K);

fprintf('Running K-Fold CV for M_pca tuning (%d values x %d folds)...\n', nM, K);
fprintf('Fixed: r=%d, M_lda=%d\n\n', optimal_r, M_lda_fixed);

for mi = 1:nM
    M_pca_val = M_pca_grid(mi);
    fprintf('  M_pca = %-4d  [ ', M_pca_val);

    for k = 1:K
        %% Build train/val sets for this fold
        val_mask   = fold_indices{k};
        val_rows   = any(val_mask, 2);

        val_data   = cv_pool(val_rows,  :);
        train_data = cv_pool(~val_rows, :);

        %% Train
        mp = cv_train_model(train_data, optimal_r, M_pca_val, M_lda_fixed);

        %% Validate
        val_rmse_cv(mi, k)   = cv_evaluate_fold(val_data,   mp);
        train_rmse_cv(mi, k) = cv_evaluate_fold(train_data, mp);

        fprintf('k%d(%.1f/%.1f) ', k, val_rmse_cv(mi,k), train_rmse_cv(mi,k));
    end
    fprintf(']\n');
end

%% Optimal M_pca
mean_val_rmse = mean(val_rmse_cv, 2);
[~, best_mi]  = min(mean_val_rmse);
optimal_M_pca = M_pca_grid(best_mi);

fprintf('\nM_pca CV Results:\n');
for mi = 1:nM
    fprintf('  M_pca=%-4d  mean_val_RMSE=%.4f   std=%.4f\n', ...
        M_pca_grid(mi), mean_val_rmse(mi), std(val_rmse_cv(mi,:)));
end
fprintf('Optimal M_pca: %d  (mean CV RMSE = %.4f mm)\n', optimal_M_pca, mean_val_rmse(best_mi));

%% Save results
save(fullfile(script_dir, 'M_pca_cv_results.mat'), ...
    'M_pca_grid', 'val_rmse_cv', 'train_rmse_cv', 'optimal_M_pca');
fprintf('\nSaved M_pca_cv_results.mat\n');

%% ---- Figure 1: M_pca vs CV RMSE (errorbar) ------------------------------
mean_val = mean(val_rmse_cv, 2);
std_val  = std(val_rmse_cv,  0, 2);

fig1 = figure('Name', 'M_pca CV RMSE', 'Visible', 'on');
errorbar(M_pca_grid, mean_val, std_val, 'o-', 'LineWidth', 2, ...
    'MarkerFaceColor', [0.2 0.4 0.8]);
hold on;
xline(170, 'r--', 'LineWidth', 1.5, ...
    'Label', 'Current M\_pca=170', 'LabelVerticalAlignment', 'bottom');
xlabel('M\_pca (PCA dims into LDA)', 'FontSize', 12);
ylabel('Mean CV RMSE (mm)', 'FontSize', 12);
title('LDA Input PCA Dims M\_pca: CV RMSE vs M\_pca', 'FontSize', 13);
legend({'CV RMSE ± std', 'Current M\_pca=170'}, 'Location', 'best');
grid on;
saveas(fig1, fullfile(script_dir, 'fig_M_pca_vs_rmse.png'));

%% ---- Figure 2: Bias-Variance tradeoff ------------------------------------
mean_train = mean(train_rmse_cv, 2);

fig2 = figure('Name', 'M_pca Bias-Variance', 'Visible', 'on');
plot(M_pca_grid, mean_train, 'b-o', 'LineWidth', 2, 'DisplayName', 'Train RMSE');
hold on;
plot(M_pca_grid, mean_val,   'r-o', 'LineWidth', 2, 'DisplayName', 'Val RMSE');
xlabel('M\_pca (PCA dims into LDA)', 'FontSize', 12);
ylabel('RMSE (mm)', 'FontSize', 12);
title('Bias-Variance Tradeoff: M\_pca', 'FontSize', 13);
legend('Location', 'best');
grid on;
saveas(fig2, fullfile(script_dir, 'fig_M_pca_bias_variance.png'));

fprintf('Figures saved.\n');
