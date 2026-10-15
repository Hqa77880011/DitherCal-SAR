from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def bits(codes, resolution):
    return ((np.asarray(codes, dtype=np.int64)[..., None] >> np.arange(resolution)) & 1).astype(float)


@dataclass
class SAR:
    weights: np.ndarray
    offset: float
    noise: float

    @classmethod
    def create(cls, resolution=10, mismatch=.01, offset=.35, noise=.2, seed=0):
        rng = np.random.default_rng(seed)
        return cls(2. ** np.arange(resolution) * (1 + mismatch * rng.normal(size=resolution)), offset, noise)

    def convert(self, inputs, rng, noisy=True):
        resolution = len(self.weights)
        level = np.asarray(inputs) + 2 ** (resolution - 1) + self.offset
        if noisy:
            level = level + rng.normal(0, self.noise, size=level.shape)
        total = np.zeros_like(level)
        codes = np.zeros(level.shape, dtype=np.int64)
        for k in range(resolution - 1, -1, -1):
            accept = total + self.weights[k] <= level
            total = total + accept * self.weights[k]
            codes = codes | (accept.astype(np.int64) << k)
        return codes


def update(positive, negative, dither, previous, nominal, variant="full",
           iterations=40, lambda_m=.2, lambda_p=.05, lambda_s=.1):
    resolution = len(previous)
    code_delta = np.abs(positive - negative)
    accepted = variant in {"no_gate", "single"} or 1 <= np.median(code_delta) <= 4
    delta = bits(positive, resolution) - bits(negative, resolution)
    rank = int(np.linalg.matrix_rank(delta))
    if not accepted:
        return previous.copy(), {"accepted": False, "pair_rank": rank, "median_excitation": float(np.median(code_delta))}
    steps = np.diff(bits(np.arange(2 ** resolution), resolution), axis=0)
    if variant == "no_mono":
        lambda_m = 0.
    if variant == "no_prior":
        lambda_p = 0.
    if variant == "no_smooth":
        lambda_s = 0.
    if variant == "single":
        centered = dither - dither.mean()
        design = np.mean(centered[:, None] * bits(positive, resolution), axis=0, keepdims=True)
        target = np.array([np.mean(centered * dither)])
    else:
        design, target = delta, 2 * dither

    def objective(weights):
        residual = design @ weights - target
        reversed_steps = np.minimum(steps @ weights, 0)
        loss = np.mean(residual ** 2) + lambda_m * np.mean(reversed_steps ** 2)
        prior = (weights - nominal) / nominal
        smooth = (weights - previous) / nominal
        loss += lambda_p * np.mean(prior ** 2) + lambda_s * np.mean(smooth ** 2)
        gradient = 2 * design.T @ residual / len(residual)
        gradient += 2 * lambda_m * steps.T @ reversed_steps / len(steps)
        gradient += 2 * (lambda_p * prior + lambda_s * smooth) / nominal / resolution
        return float(loss), gradient

    result = minimize(objective, previous, jac=True, method="L-BFGS-B",
                      options={"maxiter": iterations, "ftol": 1e-11})
    if not result.success and result.status != 1:
        raise RuntimeError(f"Calibration optimizer failed: {result.message}")
    return result.x, {"accepted": True, "pair_rank": rank,
                      "median_excitation": float(np.median(code_delta)),
                      "loss": float(result.fun), "optimizer_converged": bool(result.success),
                      "iterations": int(result.nit), "gradient_evaluations": int(result.njev),
                      "update_matrix_macs": int(2 * resolution * (len(design) + len(steps)) * result.njev)}


def decode(codes, model):
    if model["kind"] == "lut":
        return model["values"][codes]
    if model["kind"] == "mlp":
        return model["network"].predict(bits(codes, model["resolution"])) * model["scale"]
    return bits(codes, len(model["weights"])) @ model["weights"] + model["bias"]


def dynamic_metrics(values, tone):
    values = np.asarray(values, float)
    n = len(values)
    if tone <= 0 or tone >= n // 2 or np.gcd(tone, n) != 1:
        raise ValueError("Use a coherent tone coprime with record length and below Nyquist")
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    spectrum[1:-1] *= 2
    spectrum[0] = 0
    signal = spectrum[tone]
    remaining = spectrum.copy()
    remaining[tone] = 0
    noise_distortion = remaining.sum()
    sndr = 10 * np.log10(max(signal, 1e-30) / max(noise_distortion, 1e-30))
    sfdr = 10 * np.log10(max(signal, 1e-30) / max(remaining.max(), 1e-30))
    return {"sndr_db": float(sndr), "sfdr_db": float(sfdr), "enob": float((sndr - 1.76) / 6.02)}, spectrum


def static_metrics(adc, model, points=262144):
    resolution = len(adc.weights)
    full_scale = 2 ** (resolution - 1)
    ramp = np.linspace(-full_scale, full_scale, points, endpoint=False)
    raw = adc.convert(ramp, np.random.default_rng(0), noisy=False)
    decoded = decode(raw, model)
    corrected = np.clip(np.floor(decoded + full_scale), 0, 2 ** resolution - 1).astype(int)
    counts = np.bincount(corrected, minlength=2 ** resolution)
    widths = counts / (points / 2 ** resolution)
    dnl = widths - 1
    inl = np.cumsum(dnl)
    interior = slice(1, -1)
    trend = inl[1] + (np.arange(len(inl)) - 1) * (inl[-2] - inl[1]) / (len(inl) - 3)
    inl = inl - trend
    return {"max_inl_lsb": float(abs(inl[interior]).max()),
            "max_dnl_lsb": float(abs(dnl[interior]).max()),
            "missing_codes": int(np.sum(counts[interior] == 0)),
            "reversed_decoded_steps": int(np.sum(np.diff(decoded) < 0))}, inl, dnl
