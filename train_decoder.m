function model = train_decoder(trial)
% train_decoder  Train a causal linear decoder from training data.
%   model = train_decoder(trial)
%
% Inputs:
%   trial - training struct array (Ntrials x Nangles) with fields:
%           .spikes (Nunits x T), .handPos (3 x T)
% Output:
%   model - struct with fields:
%           .W      - 2 x (Nunits * nLags + 1) weight matrix (bias included)
%           .binMs  - bin size in ms used to aggregate spikes
%           .nLags  - number of past bins used (history length)
%           .muX    - mean X used for centering
%           .muY    - mean Y used for centering
%           .units  - number of neural units

% -----------------------
% Parameters (tuneable)
binMs = 20;        % bin size in ms
historyMs = 100;   % how many ms of past to use (causal)
lambda = 1e2;      % ridge regularization strength
% -----------------------

nLags = ceil(historyMs / binMs);
% collect training examples across all trials and angles
[Ntrials, Nangles] = size(trial);
% determine units and time length from first trial
spikes0 = trial(1,1).spikes;
[nUnits, ~] = size(spikes0);

X_all = []; % design matrix
Y_all = []; % targets (2D: X and Y)

for n=1:Ntrials
  for k=1:Nangles
    s = trial(n,k).spikes;      % nUnits x T (binary 0/1 per ms)
    p = trial(n,k).handPos;     % 3 x T (mm)
    T = size(s,2);
    % bin spikes into binMs windows (non-overlapping)
    nbins = floor(T / binMs);
    spikes_binned = zeros(nUnits, nbins);
    pos_binned = zeros(2, nbins);
    for b=1:nbins
      idx = (b-1)*binMs + (1:binMs);
      spikes_binned(:,b) = sum(s(:,idx),2);
      % take the hand position at the last ms of the bin (causal alignment)
      pos_binned(:,b) = p(1:2, idx(end));
    end
    % build lagged features (causal: use current bin and previous nLags-1 bins)
    for b = nLags:nbins
      feat = [];
      for lag = 0:(nLags-1)
        feat = [feat; spikes_binned(:, b - lag)]; %#ok<AGROW>
      end
      X_all = [X_all; feat']; %#ok<AGROW>
      Y_all = [Y_all; pos_binned(:, b)']; %#ok<AGROW>
    end
  end
end

% center targets
mu = mean(Y_all,1);
Yc = Y_all - mu;

% add bias column to X
X_aug = [ones(size(X_all,1),1), X_all];

% ridge regression separately for X and Y
P = size(X_aug,2);
I = eye(P);
I(1,1) = 0; % do not regularize bias
W = (X_aug' * X_aug + lambda * I) \ (X_aug' * Yc);

% store model (transpose so W is 2 x features)
model.W = W';                 % 2 x P
model.binMs = binMs;
model.nLags = nLags;
model.muX = mu(1);
model.muY = mu(2);
model.units = nUnits;
model.historyMs = historyMs;
model.lambda = lambda;
end


