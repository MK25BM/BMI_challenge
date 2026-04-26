% step7_tune_M_lda.m
% =========================================================================
% Step 7: Tune LDA dimensions kept (M_lda) via K-Fold CV
%
% Uses optimal_r and optimal_M_pca from Steps 5–6.
%
% Outputs
%   M_lda_cv_results.mat        — M_lda_grid, val_rmse_cv, train_rmse_cv, optimal_M_lda
%   fig_M_lda_vs_rmse.png       — errorbar plot: mean±std val RMSE vs M_lda
%   fig_M_lda_eigenspectrum.png — LDA eigenvalue spectrum from full CV pool model
% =========================================================================

clear; clc;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
shared_dir = fullfile(script_dir, '..', 'shared_cv_utils');
addpath(script_dir);

%% Load data and previous results
load(fullfile(shared_dir, 'cv_splits.mat'),    'cv_pool');
load(fullfile(shared_dir, 'fold_indices.mat'),  'fold_indices', 'K');
load(fullfile(script_dir, 'r_cv_results.mat'),    'optimal_r');
load(fullfile(script_dir, 'M_pca_cv_results.mat'), 'optimal_M_pca');

fprintf('Using optimal_r = %d, optimal_M_pca = %d (from Steps 5-6)\n', ...
    optimal_r, optimal_M_pca);

%% Hyperparameter grid
M_lda_grid = [1, 2, 3, 4, 5, 6, 7];
nM         = numel(M_lda_grid);

%% Pre-allocate result matrices
val_rmse_cv   = zeros(nM, K);
train_rmse_cv = zeros(nM, K);

fprintf('Running K-Fold CV for M_lda tuning (%d values x %d folds)...\n', nM, K);
fprintf('Fixed: r=%d, M_pca=%d\n\n', optimal_r, optimal_M_pca);

for mi = 1:nM
    M_lda_val = M_lda_grid(mi);
    fprintf('  M_lda = %d  [ ', M_lda_val);

    for k = 1:K
        %% Build train/val sets for this fold
        val_mask   = fold_indices{k};
        val_rows   = any(val_mask, 2);

        val_data   = cv_pool(val_rows,  :);
        train_data = cv_pool(~val_rows, :);

        %% Train
        mp = cv_train_model(train_data, optimal_r, optimal_M_pca, M_lda_val);

        %% Validate
        val_rmse_cv(mi, k)   = cv_evaluate_fold(val_data,   mp);
        train_rmse_cv(mi, k) = cv_evaluate_fold(train_data, mp);

        fprintf('k%d(%.1f/%.1f) ', k, val_rmse_cv(mi,k), train_rmse_cv(mi,k));
    end
    fprintf(']\n');
end

%% Optimal M_lda
mean_val_rmse = mean(val_rmse_cv, 2);
[~, best_mi]  = min(mean_val_rmse);
optimal_M_lda = M_lda_grid(best_mi);

fprintf('\nM_lda CV Results:\n');
for mi = 1:nM
    fprintf('  M_lda=%d  mean_val_RMSE=%.4f   std=%.4f\n', ...
        M_lda_grid(mi), mean_val_rmse(mi), std(val_rmse_cv(mi,:)));
end
fprintf('Optimal M_lda: %d  (mean CV RMSE = %.4f mm)\n', optimal_M_lda, mean_val_rmse(best_mi));

%% Save results
save(fullfile(script_dir, 'M_lda_cv_results.mat'), ...
    'M_lda_grid', 'val_rmse_cv', 'train_rmse_cv', 'optimal_M_lda');
fprintf('\nSaved M_lda_cv_results.mat\n');

%% ---- Figure 1: M_lda vs CV RMSE (errorbar) ------------------------------
mean_val = mean(val_rmse_cv, 2);
std_val  = std(val_rmse_cv,  0, 2);

fig1 = figure('Name', 'M_lda CV RMSE', 'Visible', 'on');
errorbar(M_lda_grid, mean_val, std_val, 'o-', 'LineWidth', 2, ...
    'MarkerFaceColor', [0.2 0.4 0.8]);
hold on;
xline(7, 'r--', 'LineWidth', 1.5, ...
    'Label', 'Current M\_lda=7', 'LabelVerticalAlignment', 'bottom');
xlabel('M\_lda (LDA dimensions kept)', 'FontSize', 12);
ylabel('Mean CV RMSE (mm)', 'FontSize', 12);
title('LDA Dimensions M\_lda: CV RMSE vs M\_lda', 'FontSize', 13);
legend({'CV RMSE ± std', 'Current M\_lda=7'}, 'Location', 'best');
grid on;
saveas(fig1, fullfile(script_dir, 'fig_M_lda_vs_rmse.png'));

%% ---- Figure 2: LDA Eigenspectrum ----------------------------------------
% Train a model on full CV pool to extract LDA eigenvalues
fprintf('\nComputing LDA eigenspectrum on full CV pool...\n');

dt_class = 80;
t_class  = 320:dt_class:560;
M_pca_use = optimal_M_pca;

% Build classification features
[F_cls, labels, t_tag] = build_features(cv_pool, dt_class, t_class(end));

% Use the last (full-time) classification checkpoint for eigenspectrum
T_full = t_class(end);
X_sub  = F_cls(t_tag <= T_full, :);

% PCA
nComp_use = min(M_pca_use, size(X_sub, 2) - 1);
[~, mu_sub, Wpca_sub, ~] = pca_model_local(X_sub, nComp_use);

% Scatter matrices
[Sb_sub, Sw_sub] = scatter_matrices_local(X_sub, labels);

% Compute ALL LDA eigenvalues (not truncated to M_lda)
Wp     = Wpca_sub(:, 1:nComp_use);
Sw_pca = Wp' * Sw_sub * Wp;
Sb_pca = Wp' * Sb_sub * Wp;
[~, Llda] = eig(Sw_pca \ Sb_pca);
eig_vals   = sort(diag(Llda), 'descend');
eig_vals   = eig_vals(eig_vals > 0);   % keep positive eigenvalues only

fig2 = figure('Name', 'LDA Eigenspectrum', 'Visible', 'on');
bar(1:numel(eig_vals), eig_vals, 'FaceColor', [0.3 0.6 0.9]);
hold on;
xline(optimal_M_lda, 'r--', 'LineWidth', 2, ...
    'Label', sprintf('Optimal M\\_lda=%d', optimal_M_lda), ...
    'LabelVerticalAlignment', 'bottom');
xlabel('LDA Dimension', 'FontSize', 12);
ylabel('Eigenvalue (Discrimination Power)', 'FontSize', 12);
title('LDA Eigenvalue Spectrum (Full CV Pool)', 'FontSize', 13);
grid on;
saveas(fig2, fullfile(script_dir, 'fig_M_lda_eigenspectrum.png'));

fprintf('Figures saved.\n');


%% -----------------------------------------------------------------------
%  LOCAL HELPER FUNCTIONS (for eigenspectrum computation only)
%% -----------------------------------------------------------------------

function [X, labels, t_tag] = build_features(data, dt, T_end)
% Replicate organise_features logic locally for eigenspectrum computation.
    nNeurons = size(data(1,1).spikes, 1);
    T        = dt:dt:T_end;
    nBins    = numel(T);
    nTrials  = size(data, 1);
    nDirs    = size(data, 2);

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

    X1    = zeros(nNeurons*nBins, nTrials, nDirs);
    t_tag = zeros(1, nNeurons*nBins);
    for bi = 1:nBins
        rows = (bi-1)*nNeurons + (1:nNeurons);
        X1(rows,:,:) = X0(:,:,:,bi);
        t_tag(rows)  = T(bi);
    end

    X      = zeros(nNeurons*nBins, nTrials*nDirs);
    labels = zeros(1, nTrials*nDirs);
    for k = 1:nDirs
        cols         = (k-1)*nTrials + (1:nTrials);
        X(:, cols)   = X1(:,:,k);
        labels(cols) = k;
    end
end


function [N, mu, U, L] = pca_model_local(X, p)
    N  = size(X, 2);
    mu = mean(X, 2);
    A  = X - mu;
    S  = (A' * A) / N;
    [V, L]   = eig(S);
    p        = min(p, size(V,2));
    [~, idx] = maxk(diag(L), p);
    U        = A * V(:,idx);
    U        = U ./ sqrt(sum(U.^2));
    L        = L(idx,idx);
end


function [Sb, Sw] = scatter_matrices_local(X, labels)
    mu_all  = mean(X, 2);
    classes = unique(labels);
    mu_cls  = zeros(size(X,1), numel(classes));
    for ci = 1:numel(classes)
        mu_cls(:,ci) = mean(X(:, labels == classes(ci)), 2);
    end
    St = (X - mu_all) * (X - mu_all)';
    Sb = (mu_cls - mu_all) * (mu_cls - mu_all)';
    Sw = St - Sb;
end
