% See what variables are inside (no full load)
clear ; close all; clc

load("/MATLAB Drive/monkey_bmi/monkeydata_training.mat");  

% show class and a summary of fields and their contained element classes
fprintf('Class: %s\n', class(trial));
[Ntrials, Nangles] = size(trial);

  %     training_data(n,k)              (n = trial id,  k = reaching angle)
  %     training_data(n,k).trialId      unique number of the trial
  %     training_data(n,k).spikes(i,t)  (i = neuron id, t = time)
  %     training_data(n,k).handPos(d,t) (d = dimension [1-3], t = time)

for n=1:Ntrials
  for k=1:Nangles
    s = trial(n,k).spikes;      % nUnits x T (binary 0/1 per ms)
    p = trial(n,k).handPos;     % 3 x T (mm)
    T = size(s,2);
  end
end 



fields = fieldnames(trial);
fprintf('Struct with %d elements, fields:\n', numel(trial));
disp(fields);
for k=1:numel(fields)
    fval = {trial.(fields{k})};
    types = cellfun(@class, fval, 'UniformOutput', false);
    fprintf('Field %s: classes: %s\n', fields{k}, strjoin(unique(types), ', '));
end
% Extract and display the first few entries of the 'spikes' field for the first trial

% Choose a trial and angle
trialNum = 1;
angleNum = 1;

% Extract spike matrix for this trial
spikes = trial(trialNum, angleNum).spikes;   % 98 x T
[numUnits, T] = size(spikes);

% --- 1. Compute mean firing rate per neuron (Hz) ---
firingRates = sum(spikes, 2) / (T/1000);   % spikes per second

% Normalize firing rates to [0,1]
frNorm = (firingRates - min(firingRates)) ./ (max(firingRates) - min(firingRates) + eps);

% --- 2. Build reversed colormap ---
% Start with a warm colormap (hot), then flip it so:
%   low firing rate → warm
%   high firing rate → cool
baseMap = hot(256);        % warm colors
revMap  = flipud(baseMap); % reversed: cool at high end

figure; hold on;

% --- 3. Plot raster with vertical ticks colored by firing rate ---
for unit = 1:numUnits
    spikeTimes = find(spikes(unit,:) == 1);

    % Map this neuron's firing rate to a color
    colorIdx = max(1, round(frNorm(unit) * 255));
    thisColor = revMap(colorIdx, :);

    % Draw vertical tick for each spike
    for t = spikeTimes
        plot([t t], [unit-0.4 unit+0.4], 'Color', thisColor, 'LineWidth', 1);
    end
end

% --- 4. Overlay movement onset (300 ms) ---
movementOnset = 300;
xline(movementOnset, 'r--', 'LineWidth', 1.5);

xlabel('Time (ms)');
ylabel('Neuron (unit index)');
title('Population Raster Colored by Per-Neuron Firing Rate (Warm = Low, Cool = High)');

ylim([0 numUnits+1]);
xlim([0 T]);
set(gca, 'YDir', 'reverse');

colorbar; colormap(revMap);


%%%%%%%%%%%%%%

% Pick a neuron to visualize
unit = 10;   % choose any unit 1–98

% Number of trials and angles
[numTrials, numAngles] = size(trial);

figure; hold on;

% Loop through all trials and angles
trialCounter = 0;
for ang = 1:numAngles
    for tr = 1:numTrials
        
        trialCounter = trialCounter + 1;  % raster row index
        
        % Extract spike train for this neuron
        spikes = trial(tr, ang).spikes(unit, :);   % 1 x T
        T = length(spikes);
        
        % Find spike times
        spikeTimes = find(spikes == 1);
        
        % Plot vertical ticks for each spike
        for t = spikeTimes
            plot([t t], [trialCounter-0.4 trialCounter+0.4], ...
                 'k', 'LineWidth', 1);
        end
    end
end

xlabel('Time (ms)');
ylabel('Trial Number');
title(['Multi-Trial Raster for Neuron ' num2str(unit)]);

ylim([0 trialCounter+1]);
xlim([0 T]);
set(gca, 'YDir', 'reverse');  % trial 1 at top

%%%%%%%%%%%%%%%%%

% 
% 
% % Preallocate a cell array for speed (tables grow slowly)
% rows = {};
% 
% rowIdx = 1;
% 
% [numTrials, numAngles] = size(trial);
% 
% for ang = 1:numAngles
%     for tr = 1:numTrials
% 
%         trData = trial(tr, ang);
%         spikes = trData.spikes;      % 98 x T
%         handPos = trData.handPos;    % 3 x T
%         T = size(spikes, 2);
% 
%         for unit = 1:size(spikes,1)
%             for t = 1:T
% 
%                 rows{rowIdx,1} = tr;                 % trial_num
%                 rows{rowIdx,2} = ang;                % trial_angle
%                 rows{rowIdx,3} = trData.trialId;     % trial_id
%                 rows{rowIdx,4} = unit;               % neuron_unit_num
%                 rows{rowIdx,5} = t;                  % spike_time_ms
%                 rows{rowIdx,6} = spikes(unit,t);     % spike_0_1
%                 rows{rowIdx,7} = handPos(1,t);       % hand_pos_x
%                 rows{rowIdx,8} = handPos(2,t);       % hand_pos_y
%                 rows{rowIdx,9} = handPos(3,t);       % hand_pos_z
% 
%                 rowIdx = rowIdx + 1;
%             end
%         end
%     end
% end
% 
% % Convert to table
% flatTable = cell2table(rows, 'VariableNames', ...
%     {'trial_num','trial_angle','trial_id','neuron_unit_num', ...
%      'spike_time_ms','spike_0_1','hand_pos_x','hand_pos_y','hand_pos_z'});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%    Histogram of trial durations (outliers???)
minDuration = inf;
maxDuration = -inf;

[numTrials, numAngles] = size(trial);

for ang = 1:numAngles
    for tr = 1:numTrials
        
        spikes = trial(tr, ang).spikes;   % 98 x T
        T = size(spikes, 2);              % duration in ms
        
        minDuration = min(minDuration, T);
        maxDuration = max(maxDuration, T);
    end
end

durations = [];

for ang = 1:numAngles
    for tr = 1:numTrials
        durations(end+1) = size(trial(tr, ang).spikes, 2);
    end
end

meanDuration = mean(durations);
stdDuration  = std(durations);

figure; hold on;

histogram(durations);
xlabel('Trial duration (ms)');
ylabel('Count');
title('Distribution of Trial Durations');

minDuration;
maxDuration;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Choose neuron and angle
unit = 10;
angleNum = 1;

[numTrials, ~] = size(trial);

% --- 1. Find max trial duration ---
maxT = 0;
for tr = 1:numTrials
    spikes = trial(tr, angleNum).spikes(unit,:);
    maxT = max(maxT, length(spikes));
end

% --- 2. Build padded spike matrix ---
allSpikes = zeros(numTrials, maxT);

for tr = 1:numTrials
    spikes = trial(tr, angleNum).spikes(unit,:);
    T = length(spikes);
    allSpikes(tr,1:T) = spikes;   % pad with zeros
end

% --- 3. Raw PSTH ---
rawPSTH = mean(allSpikes, 1);

% --- 4. Causal smoothing kernel ---
sigma = 20;
t = 0:5*sigma;
kernel = exp(-(t.^2)/(2*sigma^2));
kernel = kernel / sum(kernel);

% --- 5. Causal PSTH ---
psthCausal = conv(rawPSTH, kernel, 'same');

% --- 6. Plot ---
figure; hold on;
plot(rawPSTH, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
plot(psthCausal, 'b', 'LineWidth', 2);

xline(300, 'r--', 'LineWidth', 1.5);

xlabel('Time (ms)');
ylabel('Firing rate (spikes/ms)');
title(['Causal PSTH for Neuron ' num2str(unit) ', Angle ' num2str(angleNum)]);
legend('Raw PSTH', 'Causal Smoothed PSTH');