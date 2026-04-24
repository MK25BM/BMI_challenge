% Outliers exploration

clear ; close all; clc

load("/MATLAB Drive/monkey_bmi/monkeydata_training.mat");  

% show class and a summary of fields and their contained element classes
fprintf('Class of trial: %s\n', class(trial));
[nTrials, nAngles] = size(trial);
% Get neuron count from one entry (should be same for all)


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
[nUnits, T] = size(spikes);

% --- 1. Compute mean firing rate per neuron (Hz) ---
firingRates = sum(spikes, 2) / (T/1000);   % spikes per second

figure;
histogram(firingRates, 20);
xlabel('Mean firing rate (Hz)');
ylabel('Number of neurons');
title('Neuron Firing Rate Distribution for Trial 1, Angle 1');

% --- 2. Visualise for all neurons ---

firingRates = zeros(nTrials, nAngles, nUnits); % 3D array

for tr = 1:nTrials
    for ang = 1:nAngles
        spikes = trial(tr, ang).spikes; % [nUnits x T]
        T = size(spikes,2);
        firingRates(tr, ang, :) = sum(spikes, 2) / (T/1000); % firing rate per neuron (Hz)
    end
end

% Average over trials and angles for each neuron
meanFR_perNeuron = mean(reshape(firingRates, [], nUnits), 1);
figure;
bar(meanFR_perNeuron);
xlabel('Neuron index');
ylabel('Mean firing rate (Hz)');
title('Mean Firing Rate Per Neuron (All Trials and Angles)');

% Distribution Across All Trials and Angles
allRates = firingRates(:); % convert to vector
figure;
histogram(allRates, 30);
xlabel('Firing rate (Hz)');
ylabel('Count');
title('Firing Rate Distribution Across All Trials and Angles');


%Firing Rates per Trial and Angle
avgFR_perTrialAndAngle = mean(firingRates, 3); % collapse neuron dim
figure;
imagesc(avgFR_perTrialAndAngle);
colorbar;
xlabel('Angle');
ylabel('Trial');
title('Mean Firing Rate (Averaged over Neurons)');
