"""Estimate the above-threshold area of a synthetic optical beacon."""
import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def estimate_visibility(n, seed, side_m, spread_m, peak, threshold):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-side_m / 2, side_m / 2, size=(n, 2))
    radius_squared = np.sum(points**2, axis=1)
    light = peak * np.exp(-radius_squared / (2 * spread_m**2))
    detected = light >= threshold
    probability = detected.mean()
    # Wilson interval for a binomial proportion, including p = 0 and p = 1.
    z = 1.959963984540054
    denominator = 1 + z**2 / n
    center = (probability + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(probability * (1 - probability) / n + z**2 / (4 * n**2)) / denominator
    return points, detected, probability, max(0, center - half_width), min(1, center + half_width)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--spread-m", type=float, default=18)
    parser.add_argument("--peak", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("build/monte_carlo"))
    args = parser.parse_args()
    if args.samples < 1 or not all(
        math.isfinite(value) and value > 0
        for value in (args.spread_m, args.peak, args.threshold)
    ):
        parser.error("samples, spread, peak and threshold must be finite and positive")

    side_m = 100
    points, detected, p, low, high = estimate_visibility(
        args.samples, args.seed, side_m, args.spread_m, args.peak, args.threshold
    )
    radius = args.spread_m * math.sqrt(max(0, 2 * math.log(args.peak / args.threshold)))
    analytic = math.pi * radius**2 / side_m**2 if radius <= side_m / 2 else None
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "visibility.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["samples", "seed", "coverage", "wilson_low", "wilson_high", "analytic_coverage"])
        writer.writerow([args.samples, args.seed, p, low, high, analytic])
    print(f"Coverage {p:.4f}; 95% Monte Carlo interval [{low:.4f}, {high:.4f}]")
    if analytic is not None:
        print(f"Analytic coverage {analytic:.4f}")
    else:
        print("The disk crosses the square boundary; the simple area formula does not apply.")

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sample = points[:3500]
    ax.scatter(*sample.T, c=np.where(detected[:3500], "#147a72", "#a9b1bc"), s=5)
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="#22344d"))
    ax.set(xlim=(-50, 50), ylim=(-50, 50), xlabel="x (m)", ylabel="y (m)", title="Simulated detection region")
    ax.set_aspect("equal")
    fig.savefig(args.output / "coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
