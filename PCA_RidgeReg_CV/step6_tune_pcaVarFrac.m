% step6_tune_pcaVarFrac.m
% =========================================================================
% Step 6: Tune PCA variance fraction via K-Fold CV
%
% Uses the optimal lambda from Step 5.  Sweeps pcaVarFrac_grid and records
% validation RMSE per fold.
%
% Outputs
%   pcaVarFrac_cv_results.mat
%   Fig_scree.fig / .png
%   Fig_pcaVarFrac_cv_rmse.fig / .png
% =========================================================================

clear; clc;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

%% Load data
load(fullfile(script_dir, 'cv_splits.mat'),         'cv_pool');
load(fullfile(script_dir, 'fold_indices.mat'),       'fold_indices', 'K');
load(fullfile(script_dir, 'lambda_cv_results.mat'),  'best_lambda');

fprintf('Using optimal lambda = %g\n', best_lambda);

%% Hyperparameter grid
varFrac_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9];
nVF          = numel(varFrac_grid);
Khist_fixed  = 11;

%% Pre-allocate
val_rmse_vf = zeros(nVF, K);

fprintf('Running K-Fold CV for pcaVarFrac tuning (%d values x %d folds)...\n', nVF, K);

[nPool, ~] = size(cv_pool);

for vi = 1:nVF
    vf = varFrac_grid(vi);
    fprintf('  pcaVarFrac = %-5g  [ ', vf);

    for k = 1:K
        val_mask  = fold_indices{k};
        val_rows  = any(val_mask, 2);
        val_data  = cv_pool(val_rows,  :);
        train_data = cv_pool(~val_rows, :);

        mp = cv_train_model(train_data, best_lambda, vf, Khist_fixed);
        val_rmse_vf(vi, k) = cv_evaluate_fold(val_data, mp);

        fprintf('k%d(%.1f) ', k, val_rmse_vf(vi,k));
    end
    fprintf(']\n');
end

%% Optimal pcaVarFrac
mean_vf_rmse = mean(val_rmse_vf, 2);
[~, best_vi] = min(mean_vf_rmse);
best_pcaVarFrac = varFrac_grid(best_vi);
fprintf('\nOptimal pcaVarFrac: %g  (mean CV RMSE = %.4f mm)\n', ...
    best_pcaVarFrac, mean_vf_rmse(best_vi));

%% Save results  (saved by step6_tune_pcaVarFrac.m)
save(fullfile(script_dir, 'pcaVarFrac_cv_results.mat'), ...
    'varFrac_grid', 'val_rmse_vf', 'best_pcaVarFrac');
fprintf('Saved pcaVarFrac_cv_results.mat\n');

%% ---- Scree plot from full CV pool PCA ------------------------------------
% Collect all features from the CV pool for PCA
Khist_tmp  = 11;
binSize    = 20;
tStart     = 320;
[nPool, nDir] = size(cv_pool);
nNeurons   = size(cv_pool(1,1).spikes, 1);

Tmax = 0;
for tr = 1:nPool
    for d = 1:nDir
        Tmax = max(Tmax, size(cv_pool(tr,d).spikes, 2));
    end
end
times_tmp = tStart : binSize : Tmax;
nTime_tmp = numel(times_tmp);

PhiAll = [];
for d = 1:nDir
    for ti = 1:nTime_tmp
        tNow = times_tmp(ti);
        for tr = 1:nPool
            sp = cv_pool(tr,d).spikes;
            hp = cv_pool(tr,d).handPos;
            if tNow <= size(sp,2) && tNow <= size(hp,2)
                bNow = floor(tNow / binSize);
                feat = zeros(nNeurons, Khist_tmp);
                for idx = 1:Khist_tmp
                    bj = bNow - Khist_tmp + idx;
                    if bj >= 1
                        t1 = max(1, (bj-1)*binSize + 1);
                        t2 = min(tNow, min(size(sp,2), bj*binSize));
                        if t1 <= t2
                            feat(:,idx) = sum(sp(:, t1:t2), 2);
                        end
                    end
                end
                PhiAll = [PhiAll, feat(:)]; %#ok<AGROW>
            end
        end
    end
end

muPhi       = mean(PhiAll, 2);
PhiCentered = bsxfun(@minus, PhiAll, muPhi);
[~, S_full, ~] = svd(PhiCentered, 'econ');
varExp   = (diag(S_full).^2) / sum(diag(S_full).^2);
cumVarFull = cumsum(varExp);

nShow = min(60, numel(varExp));

fig1 = figure('Name', 'Scree', 'Visible', 'on');
subplot(1,2,1);
plot(1:nShow, varExp(1:nShow)*100, 'b-o', 'LineWidth', 1.5);
xlabel('Principal Component');
ylabel('% Variance Explained');
title('Scree Plot');
grid on;

subplot(1,2,2);
plot(1:nShow, cumVarFull(1:nShow)*100, 'b-o', 'LineWidth', 1.5);
hold on;
% Mark current 10% threshold
n10 = find(cumVarFull >= 0.1, 1);
n90 = find(cumVarFull >= 0.9, 1);
if ~isempty(n10); xline(n10, 'r--', 'LineWidth', 1.2, 'DisplayName', '10% threshold'); end
if ~isempty(n90); xline(n90, 'g--', 'LineWidth', 1.2, 'DisplayName', '90% variance'); end
xlabel('Number of Components');
ylabel('Cumulative % Variance');
title('Cumulative Variance Explained');
legend('Location', 'best');
grid on;

sgtitle('PCA of CV Pool — Full Spectrum');
savefig(fig1, fullfile(script_dir, 'Fig_scree.fig'));
saveas(fig1,  fullfile(script_dir, 'Fig_scree.png'));

%% ---- Figure 2: pcaVarFrac vs CV RMSE ------------------------------------
mean_vf = mean(val_rmse_vf, 2);
std_vf  = std(val_rmse_vf,  0, 2);

fig2 = figure('Name', 'pcaVarFrac CV RMSE', 'Visible', 'on');
errorbar(varFrac_grid, mean_vf, std_vf, 's-', 'LineWidth', 2, ...
    'MarkerFaceColor', [0.2 0.6 0.4]);
hold on;
xline(0.1, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Current = 0.1');
xlabel('PCA Variance Fraction Retained', 'FontSize', 12);
ylabel('Mean CV RMSE (mm)', 'FontSize', 12);
title('PCA Variance Fraction: CV RMSE vs pcaVarFrac', 'FontSize', 13);
legend({'CV RMSE ± std', 'Current = 0.1'}, 'Location', 'best');
grid on;
savefig(fig2, fullfile(script_dir, 'Fig_pcaVarFrac_cv_rmse.fig'));
saveas(fig2,  fullfile(script_dir, 'Fig_pcaVarFrac_cv_rmse.png'));

fprintf('Figures saved.\n');
