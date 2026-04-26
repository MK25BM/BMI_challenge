% step9_final_model.m
% =========================================================================
% Step 9: Train final model on ALL available data with optimal hyperparameters
%
% Combines cv_pool + test_set = full trial array, then trains cv_train_model
% with the best hyperparameters found in Steps 5–7.
%
% Saves final_model_parameters.mat
% =========================================================================

clear; clc;

%% Paths
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

%% Load optimal hyperparameters
load(fullfile(script_dir, 'lambda_cv_results.mat'),      'best_lambda');
load(fullfile(script_dir, 'pcaVarFrac_cv_results.mat'),  'best_pcaVarFrac');
load(fullfile(script_dir, 'Khist_cv_results.mat'),       'best_Khist');

fprintf('=== Optimal Hyperparameters ===\n');
fprintf('  lambda     = %g\n', best_lambda);
fprintf('  pcaVarFrac = %g\n', best_pcaVarFrac);
fprintf('  Khist      = %d\n', best_Khist);

%% Load all data and combine
load(fullfile(script_dir, 'cv_splits.mat'), 'cv_pool', 'test_set');
all_data = [cv_pool; test_set];

[nAll, nDir] = size(all_data);
fprintf('\nTraining on combined dataset: %d trials x %d directions\n', nAll, nDir);

%% Train final model
rng(42);
modelParameters = cv_train_model(all_data, best_lambda, best_pcaVarFrac, best_Khist);

%% Save  (saved by step9_final_model.m)
save_path = fullfile(script_dir, 'final_model_parameters.mat');
save(save_path, 'modelParameters');

fprintf('\nFinal model saved: %s\n', save_path);
fprintf('\n=== Summary ===\n');
fprintf('  Training trials  : %d x %d = %d\n', nAll, nDir, nAll*nDir);
fprintf('  PCA components   : %d\n', modelParameters.PCA.nComp);
fprintf('  lambda           : %g\n', modelParameters.lambda);
fprintf('  Khist            : %d\n', modelParameters.Khist);
fprintf('  Decode time grid : %d to %d ms (step %d ms)\n', ...
    modelParameters.times(1), modelParameters.times(end), modelParameters.binSize);
