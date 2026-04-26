% step9_final_model.m
% =========================================================================
% Step 9: Train final LDA_PCR model on ALL available data
%
% Combines cv_pool + test_set = full trial array, then trains cv_train_model
% with the best hyperparameters found in Steps 5–7.
%
% Saves final_model_parameters.mat
% =========================================================================

clear; clc;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
shared_dir = fullfile(script_dir, '..', 'shared_cv_utils');
addpath(script_dir);

%% Load optimal hyperparameters
load(fullfile(script_dir, 'r_cv_results.mat'),    'optimal_r');
load(fullfile(script_dir, 'M_pca_cv_results.mat'), 'optimal_M_pca');
load(fullfile(script_dir, 'M_lda_cv_results.mat'), 'optimal_M_lda');

fprintf('Final Model Trained — LDA_PCR_CV\n');
fprintf('  r      = %d\n', optimal_r);
fprintf('  M_pca  = %d\n', optimal_M_pca);
fprintf('  M_lda  = %d\n', optimal_M_lda);

%% Load all data and combine
load(fullfile(shared_dir, 'cv_splits.mat'), 'cv_pool', 'test_set');

% Also load full monkeydata to reconstruct original trial order
data_path = fullfile(script_dir, '..', 'monkeydata_training.mat');
load(data_path, 'trial');

% Combine cv_pool and test_set to recover all trials
all_data = [cv_pool; test_set];

[nAll, nDir] = size(all_data);
fprintf('\nTraining on combined dataset: %d trials x %d directions\n', nAll, nDir);

%% Train final model
rng(42);
modelParameters = cv_train_model(all_data, optimal_r, optimal_M_pca, optimal_M_lda);

%% Save
save_path = fullfile(script_dir, 'final_model_parameters.mat');
save(save_path, 'modelParameters');

fprintf('\nSaved to: LDA_PCR_CV/final_model_parameters.mat\n');
fprintf('\n=== Summary ===\n');
fprintf('  Training trials  : %d x %d = %d\n', nAll, nDir, nAll*nDir);
fprintf('  r (PCR comps)    : %d\n', modelParameters.r);
fprintf('  Decode time grid : %d to %d ms (step %d ms)\n', ...
    modelParameters.t_reg(1), modelParameters.t_reg(end), modelParameters.dt_reg);
