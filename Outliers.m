% Outliers exploration

clear ; close all; clc

load("/MATLAB Drive/monkey_bmi/monkeydata_training.mat");  

% show class and a summary of fields and their contained element classes
fprintf('Class of trial: %s\n', class(trial));
[Ntrials, Nangles] = size(trial);

fields = fieldnames(trial);

fprintf('Struct with %d elements, fields:\n', numel(trial));
disp(fields);
for k=1:numel(fields)
    fval = {trial.(fields{k})};
    types = cellfun(@class, fval, 'UniformOutput', false);
    fprintf('Field %s: classes: %s\n', fields{k}, strjoin(unique(types), ', '));
end

% Choose a trial and angle
trialNum = 1;
angleNum = 1;

% Extract spike matrix for this trial
spikes = trial(trialNum, angleNum).spikes;   % 98 x T
[numUnits, T] = size(spikes);

% --- 1. Compute mean firing rate per neuron (Hz) ---
firingRates = sum(spikes, 2) / (T/1000);   % spikes per second

figure;
histogram(firingRates, 20);
xlabel('Mean firing rate (Hz)');
ylabel('Number of neurons');
title('Neuron Firing Rate Distribution');


