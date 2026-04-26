% step1_holdout_split.m
% =========================================================================
% Step 1: Create a fixed stratified holdout test set (15% per angle)
%
% Loads monkeydata_training.mat, holds out the last 15% of trials per
% angle as a fixed test set, and saves cv_pool (85%) and test_set (15%)
% to cv_splits.mat.
%
% Run this script once before any CV tuning. The test set must NEVER be
% used during hyperparameter search.
% =========================================================================

clear; clc;

%% Load data
data_path = fullfile(fileparts(mfilename('fullpath')), '..', 'monkeydata_training.mat');
load(data_path, 'trial');

rng(42);  % Reproducibility

[nTrials, nDir] = size(trial);
fprintf('Loaded trial array: %d trials x %d directions\n', nTrials, nDir);

%% Stratified 85/15 split (last 15% per angle held out)
holdout_frac = 0.15;
nHoldout = max(1, round(nTrials * holdout_frac));  % trials per angle to hold out
nPool    = nTrials - nHoldout;

fprintf('Per angle: %d CV pool trials, %d test trials\n', nPool, nHoldout);

% cv_pool  : first nPool trials per direction
% test_set : last  nHoldout trials per direction
cv_pool  = trial(1:nPool,    :);
test_set = trial(nPool+1:end, :);

%% Summary
totalTrials  = nTrials * nDir;
cvPoolSize   = size(cv_pool,  1) * nDir;
testSetSize  = size(test_set, 1) * nDir;

fprintf('\n--- Holdout Split Summary ---\n');
fprintf('Total trials   : %d\n', totalTrials);
fprintf('CV pool size   : %d  (%.0f%%)\n', cvPoolSize,  cvPoolSize/totalTrials*100);
fprintf('Test set size  : %d  (%.0f%%)\n', testSetSize, testSetSize/totalTrials*100);

%% Save
save_path = fullfile(fileparts(mfilename('fullpath')), 'cv_splits.mat');
save(save_path, 'cv_pool', 'test_set');
fprintf('\nSaved: %s\n', save_path);
