% step2_kfold_scaffold.m
% =========================================================================
% Step 2: Build stratified K-Fold indices over the CV pool
%
% Loads cv_splits.mat, creates K=5 stratified folds (equal representation
% of all 8 directions in every fold), and saves fold_indices.mat.
%
% Run this script ONCE after step1_holdout_split.m. Both PCA_RidgeReg_CV
% and LDA_PCR_CV load from this shared location for fair comparison.
%
% Uses basic MATLAB indexing — no Statistics Toolbox crossvalind.
% =========================================================================

clear; clc;

%% Load CV pool
script_dir = fileparts(mfilename('fullpath'));
load(fullfile(script_dir, 'cv_splits.mat'), 'cv_pool');

rng(42);  % Reproducibility

[nPool, nDir] = size(cv_pool);
K = 5;
fprintf('CV pool: %d trials x %d directions,  K = %d folds\n', nPool, nDir, K);

%% Build stratified fold indices
% Strategy: within each direction, assign fold labels round-robin, then
% collect trial indices per fold across all directions.
%
% fold_indices{k} is a logical matrix (nPool x nDir) that is TRUE for
% validation trials in fold k.

fold_indices = cell(K, 1);
for k = 1:K
    fold_indices{k} = false(nPool, nDir);
end

for d = 1:nDir
    % Assign trials in this direction to folds round-robin
    trial_order = randperm(nPool);   % shuffle within direction
    for tr_idx = 1:nPool
        k = mod(tr_idx - 1, K) + 1;
        fold_indices{k}(trial_order(tr_idx), d) = true;
    end
end

%% Print fold sizes to confirm balance
fprintf('\nFold sizes (validation trials per fold):\n');
fprintf('%-8s', 'Fold');
for d = 1:nDir
    fprintf('  Dir%d', d);
end
fprintf('  Total\n');

for k = 1:K
    fprintf('%-8d', k);
    for d = 1:nDir
        fprintf('  %4d', sum(fold_indices{k}(:,d)));
    end
    fprintf('  %5d\n', sum(fold_indices{k}(:)));
end

%% Save
save_path = fullfile(script_dir, 'fold_indices.mat');
save(save_path, 'fold_indices', 'K');
fprintf('\nSaved: %s\n', save_path);
