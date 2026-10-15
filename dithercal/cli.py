import argparse
import copy
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor

from .core import SAR, bits, decode, dynamic_metrics, static_metrics, update


VARIANTS = ["full", "single", "no_mono", "no_prior", "no_smooth", "no_gate",
            "uncalibrated", "lut", "least_squares", "mlp", "oracle"]


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def generate(args):
    if not 2 <= args.bits <= 16:
        raise ValueError("Resolution must be between 2 and 16 bits")
    adc = SAR.create(args.bits, args.mismatch, args.offset, args.noise, args.seed)
    nominal = 2. ** np.arange(args.bits)
    rng = np.random.default_rng(args.seed + 100000)
    full_scale = 2 ** (args.bits - 1)
    inputs = rng.uniform(-.89 * full_scale, .89 * full_scale, (args.windows, args.batch))
    positive, negative = [], []
    physical, offsets = [], []
    for window in range(args.windows):
        phase = window / max(args.windows - 1, 1)
        current = SAR(adc.weights * (1 + args.drift * phase),
                      adc.offset + args.offset_drift * phase, adc.noise)
        positive.append(current.convert(inputs[window] + args.dither, rng))
        negative.append(current.convert(inputs[window] - args.dither, rng))
        physical.append(current.weights)
        offsets.append(current.offset)
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 200000)
    single_dither = rng.choice([-args.dither, args.dither], size=inputs.shape)
    single = np.stack([SAR(physical[i], offsets[i], adc.noise).convert(inputs[i] + single_dither[i], rng)
                       for i in range(args.windows)])
    np.savez_compressed(target / "pairs.npz", positive=np.stack(positive), negative=np.stack(negative),
                        single=single, single_dither=single_dither,
                        physical=np.stack(physical), offsets=offsets, nominal=nominal)
    dump(target / "config.json", vars(args))


def train(args):
    root = Path(args.data)
    config = json.loads((root / "config.json").read_text())
    with np.load(root / "pairs.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    nominal = data["nominal"]
    resolution = len(nominal)
    model = {"kind": "linear", "weights": nominal.copy(), "bias": -.5 * 2 ** resolution + .5}
    history = []
    snapshots = []
    start = time.perf_counter()
    conversions = 0
    rng = np.random.default_rng(args.seed + 300000)
    if args.variant in {"least_squares", "lut", "mlp"}:
        adc = SAR(data["physical"][0], data["offsets"][0], config["noise"])
        full_scale = 2 ** (resolution - 1)
        labels = np.linspace(-full_scale, full_scale, args.foreground_samples, endpoint=False)
        codes = adc.convert(labels, rng)
        conversions = len(labels)
        design = np.c_[bits(codes, resolution), np.ones(len(codes))]
        if args.variant == "least_squares":
            fit = np.linalg.lstsq(design, labels, rcond=None)[0]
            model.update(weights=fit[:-1], bias=float(fit[-1]))
        elif args.variant == "lut":
            counts = np.bincount(codes, minlength=2 ** resolution)
            midpoints = np.cumsum(counts) - .5 * counts
            model = {"kind": "lut", "values": -full_scale + midpoints / len(labels) * 2 * full_scale}
        else:
            network = MLPRegressor(hidden_layer_sizes=(64, 32), activation="tanh", max_iter=args.mlp_epochs,
                                   random_state=args.seed).fit(bits(codes, resolution), labels / full_scale)
            model = {"kind": "mlp", "network": network, "resolution": resolution, "scale": full_scale}
    accepted_count = 0
    for window, (positive, negative) in enumerate(zip(data["positive"], data["negative"])):
        active = window < args.initial_windows or rng.random() < args.tracking_duty
        diagnostics = {"accepted": False}
        if args.variant in {"full", "single", "no_mono", "no_prior", "no_smooth", "no_gate"} and active:
            if args.variant == "single":
                positive = data["single"][window]
                dither = data["single_dither"][window]
                negative = positive
                variant = "single"
                conversions += len(positive)
            else:
                dither = np.full(len(positive), config["dither"])
                variant = args.variant
                conversions += 2 * len(positive)
            model["weights"], diagnostics = update(positive, negative, dither,
                model["weights"], nominal, variant, args.iterations)
            accepted_count += int(diagnostics["accepted"])
        elif args.variant == "oracle":
            model["weights"] = data["physical"][window].copy()
            model["bias"] = -.5 * 2 ** resolution + .5 - data["offsets"][window]
        snapshots.append(copy.deepcopy(model))
        history.append({"window": window, "conversions": conversions,
                        "held_inputs": (window + 1) * len(positive), **diagnostics})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": snapshots, "variant": args.variant,
                 "training_seconds": time.perf_counter() - start, "conversions": conversions,
                 "accepted_windows": accepted_count, "config": config,
                 "update_matrix_macs": sum(row.get("update_matrix_macs", 0) for row in history),
                 "conversion_counts": [row["conversions"] for row in history]}, output / "calibration.joblib")
    dump(output / "history.json", history)


def evaluate(args):
    bundle = joblib.load(args.model)
    config = bundle["config"]
    resolution = config["bits"]
    adc = SAR.create(resolution, config["mismatch"], config["offset"], config["noise"], config["seed"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    full_scale = 2 ** (resolution - 1)
    sine = .89 * full_scale * np.sin(2 * np.pi * args.tone * np.arange(args.samples) / args.samples)
    records, spectra = [], []
    for index, model in enumerate(bundle["models"]):
        phase = index / max(len(bundle["models"]) - 1, 1)
        physical = SAR(adc.weights * (1 + config["drift"] * phase),
                       adc.offset + config["offset_drift"] * phase, adc.noise)
        rng = np.random.default_rng(args.seed + index)
        codes = physical.convert(sine, rng)
        start = time.perf_counter()
        values = decode(codes, model)
        latency = time.perf_counter() - start
        metrics, spectrum = dynamic_metrics(values, args.tone)
        records.append({"window": index, **metrics, "decode_us_per_sample": latency * 1e6 / len(codes)})
        spectra.append(spectrum)
    final = bundle["models"][-1]
    linearity, inl, dnl = static_metrics(physical, final, args.static_points)
    baseline_model = {"kind": "linear", "weights": 2. ** np.arange(resolution), "bias": -full_scale + .5}
    uncalibrated = dynamic_metrics(decode(codes, baseline_model), args.tone)[0]
    oracle_model = {"kind": "linear", "weights": physical.weights, "bias": -full_scale + .5 - physical.offset}
    oracle = dynamic_metrics(decode(codes, oracle_model), args.tone)[0]
    denominator = oracle["sndr_db"] - uncalibrated["sndr_db"]
    gap = (records[-1]["sndr_db"] - uncalibrated["sndr_db"]) / denominator if abs(denominator) > 1e-8 else None
    final_sndr = records[-1]["sndr_db"]
    convergence = None
    for index in range(len(records)):
        if min(row["sndr_db"] for row in records[index:]) >= final_sndr - args.convergence_db:
            convergence = index
            break
    if final["kind"] == "linear":
        storage, decode_macs = len(final["weights"]) + 1, len(final["weights"])
    elif final["kind"] == "lut":
        storage, decode_macs = len(final["values"]), 0
    else:
        network = final["network"]
        storage = sum(w.size for w in network.coefs_) + sum(b.size for b in network.intercepts_)
        decode_macs = sum(w.size for w in network.coefs_)
    summary = {"variant": bundle["variant"], **records[-1], **linearity,
               "sndr_gap_recovery": gap, "convergence_window": convergence,
               "convergence_held_inputs": (convergence + 1) * config["batch"] if convergence is not None else None,
               "convergence_conversions": bundle["conversion_counts"][convergence] if convergence is not None else None,
               "conversions": bundle["conversions"], "accepted_windows": bundle["accepted_windows"],
               "training_seconds": bundle["training_seconds"], "storage_values": storage,
               "update_matrix_macs_per_held_input": bundle["update_matrix_macs"] / (len(bundle["models"]) * config["batch"]),
               "decode_macs_per_conversion": decode_macs}
    dump(output / "metrics.json", summary)
    dump(output / "tracking.json", records)
    np.savez_compressed(output / "curves.npz", spectrum=np.stack(spectra), inl=inl, dnl=dnl)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = list(Path(args.input).rglob("metrics.json"))
    if not paths:
        raise ValueError("No evaluated metrics found")
    rows = [json.loads(p.read_text()) for p in paths]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for key in ["sndr_db", "sfdr_db", "enob", "max_inl_lsb", "max_dnl_lsb"]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([str(p.parent.relative_to(args.input)) for p in paths], [row[key] for row in rows], color="#4c8496")
        ax.set_ylabel(key)
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(output / f"{key}.png", dpi=180)
        plt.close(fig)
    for path in paths:
        name = str(path.parent.relative_to(args.input)).replace("\\", "_").replace("/", "_")
        tracking = json.loads((path.parent / "tracking.json").read_text())
        curves = np.load(path.parent / "curves.npz")
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot([r["window"] for r in tracking], [r["sndr_db"] for r in tracking])
        axes[0, 0].set(xlabel="Update window", ylabel="SNDR (dB)")
        spectrum = curves["spectrum"][-1]
        axes[0, 1].plot(10 * np.log10(np.maximum(spectrum / max(spectrum.max(), 1e-30), 1e-15)))
        axes[0, 1].set(xlabel="FFT bin", ylabel="Relative power (dBc)")
        axes[1, 0].plot(curves["inl"])
        axes[1, 0].set(xlabel="Corrected code", ylabel="INL (LSB)")
        axes[1, 1].plot(curves["dnl"])
        axes[1, 1].set(xlabel="Corrected code", ylabel="DNL (LSB)")
        fig.tight_layout()
        fig.savefig(output / f"{name}_curves.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="dithercal")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--output", required=True)
    for key, value in [("bits", 10), ("windows", 100), ("batch", 512), ("seed", 0)]:
        p.add_argument(f"--{key}", type=int, default=value)
    for key, value in [("mismatch", .01), ("offset", .35), ("noise", .2), ("dither", .5),
                       ("drift", .002), ("offset-drift", .2)]:
        p.add_argument(f"--{key}", type=float, default=value)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=VARIANTS, default="full")
    for key, value in [("seed", 0), ("initial-windows", 60), ("iterations", 40),
                       ("foreground-samples", 65536), ("mlp-epochs", 200)]:
        p.add_argument(f"--{key}", type=int, default=value)
    p.add_argument("--tracking-duty", type=float, default=.25)
    p = sub.add_parser("evaluate")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    for key, value in [("samples", 16384), ("tone", 127), ("static-points", 262144), ("seed", 400000)]:
        p.add_argument(f"--{key}", type=int, default=value)
    p.add_argument("--convergence-db", type=float, default=.5)
    p = sub.add_parser("plot")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
