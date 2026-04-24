
clear ; close all; clc

load("/MATLAB Drive/monkey_bmi/monkeydata_training.mat");  
% Basic structure checks
[numTrials, numAngles] = size(trial);
fprintf('Trials: %d, Angles: %d\n', numTrials, numAngles);

% Check spike matrix dimensions for a representative trial
example = trial(1,1).spikes;
[numUnits, T] = size(example);
fprintf('Units per trial: %d\n', numUnits);
fprintf('Example trial duration: %d ms\n', T);

% Compute min/max/mean trial durations across dataset
durations = zeros(numTrials, numAngles);

for ang = 1:numAngles
    for tr = 1:numTrials
        durations(tr, ang) = size(trial(tr, ang).spikes, 2);
    end
end

fprintf('Min duration: %d ms\n', min(durations(:)));
fprintf('Max duration: %d ms\n', max(durations(:)));
fprintf('Mean duration: %.2f ms\n', mean(durations(:)));

% Trials: 100, Angles: 8
% Units per trial: 98
% Example trial duration: 672 ms
% Min duration: 571 ms
% Max duration: 975 ms
% Mean duration: 643.63 ms


% Plot hand trajectories for a few trials of one angle
figure; hold on;
angleNum=1;

for tr = 1:10   % first 10 trials for illustration
    handPos = trial(tr, angleNum).handPos;   % 3 x T
    plot(handPos(1,:), handPos(2,:), 'LineWidth', 1);
end

xlabel('Hand X');
ylabel('Hand Y');
title(sprintf('Hand Trajectories (first 10 trials), Angle %d', angleNum));
axis equal;
grid on;

%Plotting hand position over time

figure; hold on;

angleNum=4;

for tr = 1:10
    handPos = trial(tr, angleNum).handPos;
    T = size(handPos,2);
    plot(1:T, handPos(1,:), 'Color', [0.6 0.6 1]); % X position
end

xlabel('Time (ms)');
ylabel('Hand X position');
title(sprintf('Hand X Position Over Time (first 10 trials), Angle %d', angleNum));

figure; hold on;

for tr = 1:10
    handPos = trial(tr, angleNum).handPos;
    T = size(handPos,2);
    plot(1:T, handPos(2,:), 'Color', [0.6 0.6 1]); % Y position
end

xlabel('Time (ms)');
ylabel('Hand Y position');
title(sprintf('Hand Y Position Over Time (first 10 trials), Angle %d', angleNum));

% %%%% Choose a trial and angle to inspect raster
trialNum = 3;
angleNum = 4;

spikes = trial(trialNum, angleNum).spikes;   % units x time
[numUnits, T] = size(spikes);

figure; hold on;

for unit = 1:numUnits
    spikeTimes = find(spikes(unit,:) == 1);
    plot(spikeTimes, unit * ones(size(spikeTimes)), 'k.', 'MarkerSize', 6);
end

xlabel('Time (ms)');
ylabel('Neuron index');
title(sprintf('Population Raster: Trial %d, Angle %d', trialNum, angleNum));
set(gca, 'YDir', 'reverse');
xlim([0 T]);
ylim([0 numUnits+1]);