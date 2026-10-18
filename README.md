# DitherCal-SAR

Implementation of **DitherCal-SAR: Label-Free Complementary-Dither Background Calibration for Linearity Recovery in SAR ADCs under PVT Drift**.

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, SciPy 1.15.3 and scikit-learn 1.6.1; neural entries use the installed PyTorch 2.7.1+cu118. Checks cover Python syntax, TOML/JSON and CLI options against parser definitions. Training and evaluation were not run.

## Data and preprocessing

The included SAR behavioral model generates complementary-dither code pairs with one held input per pair. All amplitudes are nominal LSB units. `pairs.npz` stores codes and per-window converter states; `config.json` records generation settings. True physical states are used only by the behavioral evaluator and privileged baselines.

```bash
dithercal generate --output data/main --bits 10 --mismatch 0.01 --offset 0.35 --noise 0.2 --dither 0.5 --windows 100 --batch 512 --seed 0
```

## Training and inference

```bash
dithercal train --data data/main --output runs/full --variant full --initial-windows 60 --tracking-duty 0.25 --seed 0
```

## Evaluation

```bash
dithercal evaluate --model runs/full/calibration.joblib --output runs/full/evaluation --samples 16384 --tone 127 --static-points 262144 --seed 400000
```

## Ablations

Variants: `full`, `single`, `no_mono`, `no_prior`, `no_smooth`, `no_gate`, `uncalibrated`, `lut`, `least_squares`, `mlp`, `oracle`. Reuse the same generated converter and evaluation seed. Foreground controls consume labeled ramp records; `oracle` reads physical states.

```bash
dithercal train --data data/main --output runs/no_mono --variant no_mono --seed 0
dithercal evaluate --model runs/no_mono/calibration.joblib --output runs/no_mono/evaluation --seed 400000
```

## Outputs

`runs/full/calibration.joblib` retains weight snapshots and run accounting; `history.json` records updates, excitation and optimization. Evaluation writes `metrics.json`, `tracking.json` and `curves.npz`; plot writes `comparison.csv` and curve PNGs.

```bash
dithercal plot --input runs --output runs/figures
```
