# 🧠 BMI Challenge — Team Monkey Idols (MK Version)

A Brain-Machine Interface (BMI) coursework project implementing neural decoding algorithms to estimate hand position from monkey motor cortex spiking data. Two decoding pipelines were developed and compared: **LDA + PCR** and **PCA + Ridge Regression**.

---

## 📌 Project Overview

This project tackles the BMI Challenge: given neural spike train recordings from a monkey's motor cortex (98 neurons) while it reaches in 8 directions, train a decoder that can continuously estimate hand position (x, y) in real time.

The decoder is called iteratively, receiving progressively longer spike windows (from 320 ms onwards in 20 ms steps), and must return the estimated (x, y) hand position at each timestep.

---

## 📁 Repository Structure

```
BMI_challenge/
│
├── monkeydata_training.mat          # Full training dataset (trials × angles)
│
├── Explore.m                        # EDA: raster plots, PSTHs, trial duration analysis
├── Explore_1.m                      # Additional exploratory analysis
├── Outliers.m                       # Outlier detection in trial data
│
├── continuousEstimator0.m           # Baseline template (training + estimation stubs)
│
├── LDA_PCR/                         # Approach 1: LDA for direction classification + PCR regression
│   ├── positionEstimatorTraining.m  # Train LDA classifier + PCR position model
│   ├── positionEstimator.m          # Decode hand position using LDA + PCR
│   └── testFunction_for_students_MTb.m  # Test harness
│
├── PCA_RidgeReg/                    # Approach 2: PCA dimensionality reduction + Ridge Regression
│   ├── positionEstimatorTraining.m  # Train PCA + Ridge Regression model
│   ├── positionEstimator.m          # Decode hand position using PCA + Ridge
│   └── testFunction_for_students_MTb.m  # Test harness
│
├── Fig 1.0.jpeg – Fig 1.5.jpeg      # EDA figures (rasters, PSTHs, distributions)
├── PCA_RidgeReg/Fig 2.jpeg – Fig 6.jpeg  # PCA/Ridge model result figures
│
├── CompetitionDocument.pdf          # Official competition specification
└── Paper IEEE_MK.pdf  # Team coursework report
```

---

## 🗂️ Dataset

**`monkeydata_training.mat`** — contains a `trial` struct array of size `(N_trials × 8 angles)`: 

| Field | Description |
|---|---|
| `trial(n,k).trialId` | Unique trial identifier |
| `trial(n,k).spikes(i,t)` | Binary spike matrix — 98 neurons × T ms |
| `trial(n,k).handPos(d,t)` | Hand position — 3D × T ms (x, y, z in mm) |

- **98 neurons**, **8 reaching angles**, variable trial durations
- Movement onset at **300 ms**

---

## 🤖 Decoding Approaches

### 1. LDA + PCR (`LDA_PCR/`)

- **Linear Discriminant Analysis (LDA)** classifies the intended reach direction from neural population activity
- **Principal Component Regression (PCR)** then maps PCA-reduced spike features to continuous (x, y) hand position
- Direction-specific regression models are trained per angle class

### 2. PCA + Ridge Regression (`PCA_RidgeReg/`)

- **PCA** reduces the 98-dimensional spike data to a lower-dimensional feature space
- **Ridge Regression** maps PCA features to (x, y) hand position with L2 regularisation to prevent overfitting
- A single unified regression model is used across all angles

---

## 🔍 Exploratory Analysis (`Explore.m`)

The `Explore.m` script includes:

- **Population raster plot** — coloured by per-neuron firing rate (warm = low, cool = high)
- **Single-neuron raster** — across all trials and angles
- **Trial duration histogram** — to detect potential outliers
- **Causal PSTH** — per neuron per angle, smoothed with a Gaussian kernel (σ = 20 ms)

---

## 🚀 Getting Started

### Requirements

- MATLAB R2020a or later
- No external toolboxes required (uses built-in `eig`, `svd`, `lsqminnorm`, etc.)

### Running the Decoder

1. **Choose an approach** — navigate to either `LDA_PCR/` or `PCA_RidgeReg/`
2. **Load training data:**
   ```matlab
   load('monkeydata_training.mat');
   ```
3. **Run the test harness:**
   ```matlab
   testFunction_for_students_MTb
   ```
   This will call `positionEstimatorTraining` to fit the model and then evaluate `positionEstimator` iteratively on held-out trials.

### Function Signatures

```matlab
% Train the model
modelParameters = positionEstimatorTraining(training_data);

% Predict hand position at each timestep
[x, y] = positionEstimator(test_data, modelParameters);
```

---

## 📊 Results

Result figures are stored in `PCA_RidgeReg/` (`Fig 2.jpeg` – `Fig 6.jpeg`) and the root directory (`Fig 1.0.jpeg` – `Fig 1.5.jpeg`). Full analysis and performance metrics are documented in the team report PDF.

---

## 📄 References

- Competition specification: `CompetitionDocument.pdf`
- Team report: `BMI_CW_for_team_Monkey_Idols__MK_version_.pdf`
