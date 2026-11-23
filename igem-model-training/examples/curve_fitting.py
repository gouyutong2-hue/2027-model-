"""Reconstruct Fujian-United's activity curves from published numerical inputs."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d


DATA = Path(__file__).resolve().parents[1] / "data" / "fujian_phytase_activity.csv"
ENZYMES = ("AppA", "An_phy33")


def models(variable, x, y):
    if variable == "temperature_C":
        return {"linear fit": Polynomial.fit(x, y, 1), "quartic fit": Polynomial.fit(x, y, 4)}
    return {
        "nearest": interp1d(x, y, kind="nearest", bounds_error=True),
        "linear": interp1d(x, y, kind="linear", bounds_error=True),
        "PCHIP": PchipInterpolator(x, y, extrapolate=False),
        "cubic spline": CubicSpline(x, y, extrapolate=False),
    }


def interior_validation_rmse(variable, method, x, y):
    predictions = []
    for omitted in range(1, len(x) - 1):
        keep = np.arange(len(x)) != omitted
        predict = models(variable, x[keep], y[keep])[method]
        predictions.append(float(predict(x[omitted])))
    return np.sqrt(np.mean((np.array(predictions) - y[1:-1]) ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/curve_fitting"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with DATA.open(encoding="utf-8", newline="") as stream:
        data = list(csv.DictReader(stream))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    results = []
    for row_index, (variable, label) in enumerate([("temperature_C", "Temperature (°C)"), ("pH", "pH")]):
        rows = [row for row in data if row["variable"] == variable]
        x = np.array([float(row["value"]) for row in rows])
        grid = np.linspace(x[0], x[-1], round((x[-1] - x[0]) * 10) + 1)
        for column_index, enzyme in enumerate(ENZYMES):
            y = np.array([float(row[enzyme]) for row in rows])
            ax = axes[row_index, column_index]
            for method, predict in models(variable, x, y).items():
                predicted = predict(grid)
                best = np.argmax(predicted)
                rmse = interior_validation_rmse(variable, method, x, y)
                results.append([variable, enzyme, method, grid[best], predicted[best], rmse])
                style = "steps-post" if method == "nearest" else "default"
                ax.plot(grid, predicted, label=method, drawstyle=style)
                if method in ("cubic spline", "quartic fit"):
                    print(f"{enzyme} {method}: x={grid[best]:.1f}, peak={predicted[best]:.5f}, CV RMSE={rmse:.3f}")
            ax.scatter(x, y, color="#172f41", zorder=3, s=16, label="reported inputs")
            ax.set(title=enzyme, xlabel=label, ylabel="Relative enzyme activity (team scale)")
            ax.legend(fontsize=8)

    with (args.output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variable", "enzyme", "method", "grid_peak_x", "grid_peak_y", "interior_loo_rmse"])
        writer.writerows(results)
    fig.savefig(args.output / "activity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
