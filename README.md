# LSTM-AE Anomaly Detection for Semiconductor Etch

PyTorch implementation of an LSTM autoencoder that catches etch process faults
which final-value SPC misses. The models never see real data during training:
real wafers from a LAM 9600 metal etcher are used only to extract statistics
for a synthetic data generator, and again at the very end for validation.

## Results

Final validation on real data (20 induced faults, 43 held-out normal wafers
that were never used for statistics):

| Method | Fault recall | False alarms | AUC |
|---|---|---|---|
| SPC X-bar (final value) | 25% | 0.0% | - |
| Dense AE | 75% | 2.3% | 0.923 |
| LSTM-AE | 75% | 4.7% | 0.880 |

All 8 faults on monitored sensors (Pr, Cl2, He) were detected, including the
smallest one (Pr +1). Faults on unmonitored sensors (TCP/RF) were still caught
58% of the time, because they perturb the pressure control loop and show up
indirectly.

On the synthetic benchmark (5 seeds): LSTM-AE F1 = 0.725 ± 0.013, vs 0.137 for
SPC X-bar and 0.057 for Isolation Forest.

![SPC blind spot](figures/04_spc_blind_spot.png)

A few things I learned along the way:

1. The win comes from looking at the whole trajectory instead of the final
   value. Both autoencoders get 75% recall vs SPC's 25%; the model family
   matters less than the approach.
2. A plain Dense AE matches the LSTM-AE here (and slightly beats it on AUC).
   Since the wafers are resampled to fixed length with locked profiles,
   position-specific weights are enough. What the LSTM buys you is operational:
   variable-length input and a path to streaming detection.
3. Synthetic benchmarks lie until you check them against real data. The v1
   generator (hand-designed ramps, no quantization, fixed length) scored
   F1 0.86 on its own test set but produced 100% false alarms on real wafers.
   Rebuilding it from richer real statistics dropped that to 11.6% with the
   same direct threshold transfer, still without training on real data.

## How it works

**Statistics extraction** (`01_sensor_stats.py`): the 107 real normal wafers
are split 60/40. From the 60% set, each sensor gets a mean waveform profile
(ramp direction, step-transition transient), within-wafer residual std and
lag-1 autocorrelation, between-wafer offset std, quantization step, transient
amplitude, and the process-length distribution. The 40% holdout is only used
as final-validation negatives.

**Synthetic generator** (`02_generate_synthetic.py`): each wafer is a
time-rescaled mean profile (random length 95-112) plus between-wafer offset,
AR(1) residual noise, and integer quantization. Three anomaly types, all
designed to end on target so final-value SPC can't see them, each hitting 1-2
random sensors:

- A: ramp too fast — transient compressed 2.5-4x in time, with overshoot
  ringing (only on sensors that have a real transient)
- B: mid-process oscillation — a 2.5-4 sigma damped burst that recovers before
  the end
- C: slow drift — linear drift that stays inside the ±3 sigma control limits

**Model selection**: AE detection quality is not monotonic in training epochs,
because a fully converged AE reconstructs anomalies too. So checkpoints (every
20 epochs), smoothing window, peak calibration, and threshold rule
(mean+3 sigma vs p99) are grid-selected by F1 on a held-out synthetic anomaly
validation set. The test set is touched once per seed, and results are
reported over 5 seeds. The anomaly score is the max of the smoothed
per-timestep reconstruction error, so a localized anomaly doesn't get diluted
by averaging over the whole wafer.

## Usage

```bash
pip install torch scipy scikit-learn pandas matplotlib

python 01_sensor_stats.py       # extract statistics from real normals
python 02_generate_synthetic.py # build the synthetic benchmark
python 03_train_lstm_ae.py      # 5-seed training, ~15-25 min on CPU, resumable
python 04_compare_methods.py    # SPC / Dense AE / Isolation Forest / LSTM-AE
python 05_validate_real_data.py # final validation on held-out real wafers
```

## Data

LAM 9600 Metal Etcher dataset (Eigenvector Research): 108 normal + 21 faulty
wafers, 21 engineering variables. Put `MACHINE_Data.mat` at the path set in
`config.py`; the dataset itself is not included in this repo.

The monitored sensors were picked from an equipment-control point of view:
controlled variable (Pressure), actuator (Vat Valve), cooling loop (He Press),
flow loop (Cl2 Flow). The idea is that a closed-loop story explains the
detections without needing any plasma chemistry.

## TODO

- Streaming detection with an LSTM forecaster, so alarms fire mid-process
  instead of per-wafer
- Add TCP/RF power to the monitored set and measure the coverage/recall
  trade-off
- Try a Transformer autoencoder
- Edge deployment on a Raspberry Pi (ONNX, quantization)
