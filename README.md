# DitherCal-SAR

Implementation of **DitherCal-SAR: Label-Free Complementary-Dither Background Calibration for Linearity Recovery in SAR ADCs under PVT Drift**. The calibration loop learns digital bit weights from two conversions of the same held input with opposite known dither. Excitation gating and monotonicity, binary-prior, and temporal penalties control the update.

## Install

Use Python 3.10 or later:

```bash
python -m venv .venv
```

Activate using `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then:

```bash
python -m pip install -e .
```

## Generate converter observations

```bash
dithercal generate --output data/main --bits 10 --mismatch 0.01 --offset 0.35 --noise 0.2 --dither 0.5 --windows 100 --batch 512 --seed 0
```

The data source is the included SAR behavioral model. Physical weights are nominal binary weights with fixed Gaussian relative mismatch. The comparator adds an offset and independent input-referred Gaussian noise for each conversion. Successive approximation compares trial DAC sums in descending bit order. Within every pair the input is held constant. Positive and negative conversions use the same physical converter state.

All internal amplitudes use nominal LSB units. For ten bits, a normalized differential input in [-1, 1] corresponds to [-512, 512] LSB. The supplied background inputs are uniform over 89% of that range. Reference drift scales the physical weights linearly over windows; offset drift is additive.

`pairs.npz` contains `positive` and `negative` integer codes, independent signed one-sided observations, nominal weights, and per-window physical states for privileged baselines. `config.json` records generator settings. The full background update reads paired codes and known dither; it does not read input labels or true weights.

For captured paired codes, use the same W × B arrays in an NPZ archive and the associated `config.json`. Physical states are required only for the supplied behavioral evaluator and oracle/foreground baselines. Hardware measurements require corresponding held-out sine/ramp records for those evaluations.

## Calibrate

```bash
dithercal train --data data/main --output runs/full --variant full --initial-windows 60 --tracking-duty 0.25
```

The pair residual is `(bits_positive - bits_negative) @ weights - 2*dither`. Digital bias cancels algebraically and stays at the nominal midpoint convention; the background loop does not estimate absolute comparator offset.

An update is admitted when the median absolute integer-code difference lies between 1 and 4. The finite monotonicity penalty acts on every adjacent code step. The binary prior penalizes relative weight deviations from nominal, and temporal smoothness penalizes relative changes from the previous estimate. Their coefficients are 0.20, 0.05, and 0.10. Each accepted window uses at most 40 analytic-gradient L-BFGS iterations, adjustable with `--iterations`. An iteration-limit termination is recorded; other optimizer failures raise an error.

The first 60 windows admit every dither batch. Subsequent windows independently admit a batch with probability 0.25. Change `--initial-windows` and `--tracking-duty` to alter that schedule. Conversion accounting includes both members of accepted paired acquisitions, including batches subsequently rejected by the excitation gate. It excludes foreground/evaluation records from background-conversion counts.

`calibration.joblib` stores the actual weight snapshots and accounting, and `history.json` records excitation, accepted updates, pair-design rank, optimizer termination, and objective values. The rank diagnostic is separate from the excitation heuristic.

## Evaluate all converter metrics

```bash
dithercal evaluate --model runs/full/calibration.joblib --output runs/full/evaluation --samples 16384 --tone 127 --static-points 262144
dithercal plot --input runs --output runs/figures
```

Evaluation uses independent noise seeds. Dynamic records contain a coherent sine at 89% full-scale amplitude. The FFT tone index must be coprime with the record length and below Nyquist. SNDR excludes DC and the fundamental from noise-plus-distortion power; SFDR compares the fundamental with the strongest remaining bin. ENOB is `(SNDR - 1.76) / 6.02`.

Static evaluation uses a separate noiseless dense ramp. Decoded outputs are mapped to the fixed nominal integer output grid. The histogram estimates corrected output-code widths, DNL is width minus one LSB, and INL is cumulative DNL with an endpoint linear trend removed. Rail codes are excluded from reported maxima. Missing corrected codes and reversed decoded steps are reported explicitly.

| Output | Content |
| --- | --- |
| `metrics.json` | SNDR, SFDR, ENOB, maximum INL/DNL, missing codes, oracle-gap recovery, convergence, conversion counts, storage, arithmetic and runtime |
| `tracking.json` | Actual per-window SNDR/SFDR/ENOB and decode time |
| `curves.npz` | Spectra, final INL, and final DNL arrays |
| `comparison.csv` | Evaluated method comparison generated by `plot` |
| `*_curves.png` | Drift tracking, spectrum, INL, and DNL panels |

`sndr_gap_recovery` compares the final calibrated, nominal, and true-weight decoders on identical acquired test codes. Convergence is the earliest window after which all remaining SNDR values stay within `--convergence-db`, default 0.5 dB, below the final value. The output gives the window, held-input count, and actual conversion count separately.

`decode_macs_per_conversion` counts linear or MLP decoder matrix MACs; LUT lookup has zero decoder MACs. `update_matrix_macs_per_held_input` counts the design and code-step matrix products performed by recorded optimizer gradient evaluations, averaged over all supplied held inputs. Scalar regularization, optimizer bookkeeping, and analog conversion are separate costs.

## Baselines and component ablations

```bash
dithercal train --data data/main --output runs/uncalibrated --variant uncalibrated
dithercal evaluate --model runs/uncalibrated/calibration.joblib --output runs/uncalibrated/evaluation
dithercal train --data data/main --output runs/least_squares --variant least_squares --foreground-samples 65536
dithercal evaluate --model runs/least_squares/calibration.joblib --output runs/least_squares/evaluation
dithercal train --data data/main --output runs/no_mono --variant no_mono
dithercal evaluate --model runs/no_mono/calibration.joblib --output runs/no_mono/evaluation
```

Repeat the train/evaluate commands with the following variants:

| Variant | Information and update |
| --- | --- |
| `full` | Complementary pairs with all three penalties and excitation gating |
| `single` | One signed-dither conversion per held input; centered dither/output covariance objective |
| `no_mono` | Zero monotonicity coefficient |
| `no_prior` | Zero binary-prior coefficient |
| `no_smooth` | Zero temporal-smoothness coefficient |
| `no_gate` | All acquired paired batches can update |
| `uncalibrated` | Fixed nominal binary decoder |
| `lut` | Foreground ramp code-density midpoint correction |
| `least_squares` | Foreground labeled-ramp fit of weights and bias |
| `mlp` | Foreground supervised 64/32 hidden-unit tanh network |
| `oracle` | Per-window true physical weights and offset |

Foreground models remain fixed during drift. They receive `--foreground-samples` labeled ramp observations; the LUT uses only the ramp histogram. `--mlp-epochs` controls the supervised optimizer's iteration limit. Single-sided calibration uses independence of input and random signed dither; its pair-specific gate is disabled because it acquires no negative counterpart. It provides a concrete one-sided control for the under-specified baseline in the manuscript.

For the mismatch sweep, regenerate with `--mismatch 0.002`, `0.005`, `0.01`, and `0.015`. For dither ablations, use `--dither 0.25`, `0.5`, or `1.5`. For drift tracking, change `--drift` and `--offset-drift`. Use seeds 0 through 31 for the stated 32-seed protocol, keep each seed in its own directory, and compare methods on the same generated converter. No experiment table is populated before these commands run.

## Implementation conventions

The manuscript leaves the exact drift waveform, optimizer, input distribution, static code-remapping convention, and one-sided objective unspecified. This implementation fixes them explicitly above. Pair acquisition uses independent conversion noise while retaining the same held input. Gaussian mismatch acts on effective bit weights. The default normalized gain drift is 0.002 end-to-end and comparator drift is 0.2 LSB end-to-end.

The binary-prior and temporal penalties use relative weight differences so high-order bits do not dominate solely by scale. Monotonicity remains a finite penalty, and the evaluator reports reversals or missing codes instead of hiding them. The privileged oracle accesses the physical model only as a comparison decoder. CI checks syntax and package construction without running calibration or experiments.
