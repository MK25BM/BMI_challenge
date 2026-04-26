% step7_tune_Khist.m
% =========================================================================
% Step 7: Tune number of spike history bins (Khist) via K-Fold CV
%
% Uses the optimal lambda and pcaVarFrac from Steps 5–6.
% Sweeps Khist_grid and records validation RMSE per fold.
%
% Outputs
%   Khist_cv_results.mat
%   Fig_Khist_cv_rmse.fig / .png
% =========================================================================

clear; clc;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

%% Load data and previous results
load(fullfile(script_dir, 'cv_splits.mat'),             'cv_pool');
load(fullfile(script_dir, 'fold_indices.mat'),           'fold_indices', 'K');
load(fullfile(script_dir, 'lambda_cv_results.mat'),      'best_lambda');
load(fullfile(script_dir, 'pcaVarFrac_cv_results.mat'),  'best_pcaVarFrac');

fprintf('Using lambda = %g,  pcaVarFrac = %g\n', best_lambda, best_pcaVarFrac);

%% Hyperparameter grid
Khist_grid = [3, 5, 7, 9, 11, 15, 20];
nKH        = numel(Khist_grid);

%% Pre-allocate
val_rmse_kh = zeros(nKH, K);

fprintf('Running K-Fold CV for Khist tuning (%d values x %d folds)...\n', nKH, K);

for ki = 1:nKH
    kh = Khist_grid(ki);
    fprintf('  Khist = %-4d  [ ', kh);

    for k = 1:K
        val_mask   = fold_indices{k};
        val_rows   = any(val_mask, 2);
        val_data   = cv_pool(val_rows,  :);
        train_data = cv_pool(~val_rows, :);

        mp = cv_train_model(train_data, best_lambda, best_pcaVarFrac, kh);
        val_rmse_kh(ki, k) = cv_evaluate_fold(val_data, mp);

        fprintf('k%d(%.1f) ', k, val_rmse_kh(ki,k));
    end
    fprintf(']\n');
end

%% Optimal Khist
mean_kh_rmse = mean(val_rmse_kh, 2);
[~, best_ki] = min(mean_kh_rmse);
best_Khist   = Khist_grid(best_ki);
fprintf('\nOptimal Khist: %d  (mean CV RMSE = %.4f mm)\n', ...
    best_Khist, mean_kh_rmse(best_ki));

%% Save results  (saved by step7_tune_Khist.m)
save(fullfile(script_dir, 'Khist_cv_results.mat'), ...
    'Khist_grid', 'val_rmse_kh', 'best_Khist');
fprintf('Saved Khist_cv_results.mat\n');

%% ---- Figure: Khist vs CV RMSE --------------------------------------------
mean_kh = mean(val_rmse_kh, 2);
std_kh  = std(val_rmse_kh,  0, 2);

fig1 = figure('Name', 'Khist CV RMSE', 'Visible', 'on');
errorbar(Khist_grid, mean_kh, std_kh, '^-', 'LineWidth', 2, ...
    'MarkerFaceColor', [0.6 0.2 0.6]);
hold on;
xline(11, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Current Khist=11');
xlabel('Number of Spike History Bins (Khist)', 'FontSize', 12);
ylabel('Mean CV RMSE (mm)', 'FontSize', 12);
title('Spike History Depth: CV RMSE vs Khist', 'FontSize', 13);
legend({'CV RMSE ± std', 'Current Khist=11'}, 'Location', 'best');
grid on;
savefig(fig1, fullfile(script_dir, 'Fig_Khist_cv_rmse.fig'));
saveas(fig1,  fullfile(script_dir, 'Fig_Khist_cv_rmse.png'));

fprintf('Figure saved.\n');
