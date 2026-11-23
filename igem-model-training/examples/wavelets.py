"""Detect a known transient in a simulated leaf-potential signal."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def haar_decompose(signal, levels):
    approximation = signal.copy()
    details = []
    for _ in range(levels):
        even = approximation[::2]
        odd = approximation[1::2]
        details.append((even - odd) / np.sqrt(2))
        approximation = (even + odd) / np.sqrt(2)
    return approximation, details


def haar_reconstruct(approximation, details):
    for detail in reversed(details):
        values = np.empty(2 * approximation.size)
        values[::2] = (approximation + detail) / np.sqrt(2)
        values[1::2] = (approximation - detail) / np.sqrt(2)
        approximation = values
    return approximation


def soft_threshold(values, threshold):
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0)


def step_scores(signal, half_width):
    # A sliding Haar contrast places a score at every eligible split.
    positions = np.arange(half_width, signal.size - half_width + 1)
    totals = np.concatenate(([0.0], np.cumsum(signal)))
    before = (totals[positions] - totals[positions - half_width]) / half_width
    after = (totals[positions + half_width] - totals[positions]) / half_width
    return positions, after - before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--noise", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("build/wavelets"))
    args = parser.parse_args()
    if not np.isfinite(args.noise) or args.noise < 0:
        parser.error("noise must be finite and nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    samples = np.arange(512)
    step_location = 237
    clean = 0.13 * np.sin(samples / 31) + 0.9 * (samples >= step_location)
    noisy = clean + rng.normal(0, args.noise, samples.size)
    approximation, details = haar_decompose(noisy, levels=4)
    np.testing.assert_allclose(haar_reconstruct(approximation, details), noisy, atol=1e-12)
    centered_finest = details[0] - np.median(details[0])
    noise_estimate = np.median(np.abs(centered_finest)) / 0.67448975
    threshold = noise_estimate * np.sqrt(2 * np.log(samples.size))
    denoised = haar_reconstruct(approximation, [soft_threshold(detail, threshold) for detail in details])
    locations, score = step_scores(noisy, half_width=12)
    detected = locations[np.argmax(np.abs(score))]
    print(f"RMSE raw={np.sqrt(np.mean((noisy-clean)**2)):.3f}, denoised={np.sqrt(np.mean((denoised-clean)**2)):.3f}")
    print(f"Known synthetic step {step_location}; detected sample {detected}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
    axes[0].plot(samples, noisy, color="#b0b6c3", label="observed")
    axes[0].plot(samples, denoised, label="Haar denoising")
    axes[0].plot(samples, clean, ls="--", label="synthetic truth")
    axes[0].set_ylabel("Simulated potential (a.u.)")
    axes[0].legend()
    axes[1].plot(locations, score)
    axes[1].axvline(detected, color="#cc5737", ls="--", label="detected change")
    axes[1].set(xlabel="Sample index", ylabel="Sliding Haar contrast")
    axes[1].legend()
    fig.savefig(args.output / "wavelets.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
