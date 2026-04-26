% step5_tune_lambda.m
% =========================================================================
% Step 5: Tune ridge regularisation parameter lambda via K-Fold CV
%
% For each lambda in the grid, trains on K-1 folds and evaluates on the
% held-out fold.  Also records training RMSE for the bias-variance plot.
%
% Outputs
%   lambda_cv_results.mat  — lambda_grid, val_rmse_cv, train_rmse_cv
%   Fig_lambda_cv_rmse.fig / .png
%   Fig_lambda_bias_variance.fig / .png
% =========================================================================

clear; clc;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);   % ensure local cv_train_model / cv_evaluate_fold are found

%% Load data
load(fullfile(script_dir, 'cv_splits.mat'),   'cv_pool');
load(fullfile(script_dir, 'fold_indices.mat'), 'fold_indices', 'K');

%% Hyperparameter grid
lambda_grid  = [0.01, 0.1, 1, 4, 10, 50, 100, 500];
nLambda      = numel(lambda_grid);

% Fixed hyperparameters for this step
pcaVarFrac_fixed = 0.1;
Khist_fixed      = 11;

%% Pre-allocate result matrices
val_rmse_cv   = zeros(nLambda, K);
train_rmse_cv = zeros(nLambda, K);

fprintf('Running K-Fold CV for lambda tuning (%d lambdas x %d folds)...\n', nLambda, K);

[nPool, nDir] = size(cv_pool);

for li = 1:nLambda
    lam = lambda_grid(li);
    fprintf('  lambda = %-8g  [ ', lam);

    for k = 1:K
        %% Build train/val sets for this fold
        val_mask   = fold_indices{k};   % nPool x nDir logical
        train_mask = ~val_mask;

        % Collect validation struct
        val_rows = any(val_mask, 2);       % rows that have at least one val entry
        % Build val_data: only rows where at least one direction is val
        % Use the full row for simplicity (stratified so all dirs present)
        val_data   = cv_pool(val_rows,  :);
        train_data = cv_pool(~val_rows, :);

        %% Train
        mp = cv_train_model(train_data, lam, pcaVarFrac_fixed, Khist_fixed);

        %% Validate
        val_rmse_cv(li, k)   = cv_evaluate_fold(val_data,   mp);
        train_rmse_cv(li, k) = cv_evaluate_fold(train_data, mp);

        fprintf('k%d(%.1f/%.1f) ', k, val_rmse_cv(li,k), train_rmse_cv(li,k));
    end
    fprintf(']\n');
end

%% Optimal lambda
mean_val_rmse = mean(val_rmse_cv, 2);
[~, best_li]  = min(mean_val_rmse);
best_lambda   = lambda_grid(best_li);
fprintf('\nOptimal lambda: %g  (mean CV RMSE = %.4f mm)\n', best_lambda, mean_val_rmse(best_li));

%% Save results  (saved by step5_tune_lambda.m)
save(fullfile(script_dir, 'lambda_cv_results.mat'), ...
    'lambda_grid', 'val_rmse_cv', 'train_rmse_cv', 'best_lambda');
fprintf('Saved lambda_cv_results.mat\n');

%% ---- Figure 1: Lambda vs CV RMSE ----------------------------------------
mean_val  = mean(val_rmse_cv, 2);
std_val   = std(val_rmse_cv,  0, 2);

fig1 = figure('Name', 'Lambda CV RMSE', 'Visible', 'on');
errorbar(log10(lambda_grid), mean_val, std_val, 'o-', 'LineWidth', 2, ...
    'MarkerFaceColor', [0.2 0.4 0.8]);
hold on;
xline(log10(4), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Current \lambda=4');
xlabel('log_{10}(\lambda)', 'FontSize', 12);
ylabel('Mean CV RMSE (mm)', 'FontSize', 12);
title('Ridge \lambda Tuning: CV RMSE vs \lambda', 'FontSize', 13);
legend({'CV RMSE ± std', 'Current \lambda=4'}, 'Location', 'best');
grid on;
savefig(fig1, fullfile(script_dir, 'Fig_lambda_cv_rmse.fig'));
saveas(fig1,  fullfile(script_dir, 'Fig_lambda_cv_rmse.png'));

%% ---- Figure 2: Bias-Variance tradeoff ------------------------------------
mean_train = mean(train_rmse_cv, 2);
std_train  = std(train_rmse_cv,  0, 2);

fig2 = figure('Name', 'Bias-Variance', 'Visible', 'on');
plot(log10(lambda_grid), mean_train, 'b-o', 'LineWidth', 2, 'DisplayName', 'Train RMSE');
hold on;
plot(log10(lambda_grid), mean_val,   'r-o', 'LineWidth', 2, 'DisplayName', 'Val RMSE');
xlabel('log_{10}(\lambda)', 'FontSize', 12);
ylabel('RMSE (mm)', 'FontSize', 12);
title('Bias-Variance Tradeoff', 'FontSize', 13);
legend('Location', 'best');
grid on;
savefig(fig2, fullfile(script_dir, 'Fig_lambda_bias_variance.fig'));
saveas(fig2,  fullfile(script_dir, 'Fig_lambda_bias_variance.png'));

fprintf('Figures saved.\n');
