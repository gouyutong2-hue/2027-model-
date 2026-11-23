"""Reanalyze iGEM InterLab data and compare basic statistical tests."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DATA = Path(__file__).resolve().parents[1] / "data" / "beal_2018_s1_bulk.csv"
DEVICES = ("strong device", "medium device", "weak device")


def values(rows, field):
    return np.array([float(row.get(field, "") or "NaN") for row in rows])


def geometric_statistics(data):
    log_data = np.log(data[np.isfinite(data) & (data > 0)])
    return log_data.size, np.exp(log_data.mean()), np.exp(log_data.std(ddof=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/statistics"))
    parser.add_argument("--center", type=float, default=3.02, help="reported or alternate positive-control center")
    args = parser.parse_args()
    if not np.isfinite(args.center) or args.center <= 0:
        parser.error("center must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with DATA.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    positive_control = values(rows, "positive control mean (uM FITC/OD)")
    negative_control = values(rows, "negative control mean (uM FITC/OD)")
    lower, upper = args.center / 3, args.center * 3
    passed = (
        np.isfinite(positive_control)
        & np.isfinite(negative_control)
        & (positive_control >= lower)
        & (positive_control <= upper)
        & (negative_control >= -0.01)
        & (negative_control <= 0.5)
    )
    print(f"Bulk datasets: {len(rows)}; passed controls: {passed.sum()}")
    print(f"Positive-control center used={args.center:g}; median in distributed CSV={np.nanmedian(positive_control):g}")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    summary = []
    histogram_data = []
    for name in DEVICES:
        data = values(rows, f"{name} mean (uM FITC/OD)")
        for treatment, observations in [
            ("calibrated", data),
            ("calibrated and filtered", data[passed]),
            ("normalized and filtered", data[passed] / positive_control[passed]),
        ]:
            summary.append((name, treatment, len(observations), *geometric_statistics(observations)))
        plotted_data = data[passed & np.isfinite(data) & (data > 0)]
        histogram_data.append(np.log10(plotted_data))
    if any(values.size for values in histogram_data):
        histogram_bins = np.histogram_bin_edges(np.concatenate(histogram_data), bins=12)
        for name, observations in zip(DEVICES, histogram_data):
            axes[0].hist(observations, bins=histogram_bins, histtype="step", label=name)

    with (args.output / "device_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["device", "treatment", "n_rows", "n_positive", "geometric_mean", "geometric_sd"])
        writer.writerows(summary)
    axes[0].set(xlabel="log10 calibrated fluorescence (µM FITC / OD)", ylabel="Datasets")
    axes[0].legend(fontsize=8)

    # Teaching comparison: the strong and medium measurements are paired by lab.
    strong = values(rows, "strong device mean (uM FITC/OD)")
    medium = values(rows, "medium device mean (uM FITC/OD)")
    matched = passed & np.isfinite(strong) & np.isfinite(medium) & (strong > 0) & (medium > 0)
    log_ratio = np.log(strong[matched] / medium[matched])
    n = len(log_ratio)
    if n < 8:
        raise ValueError("At least eight positive paired measurements are needed for this lesson.")
    mean = log_ratio.mean()
    sd = log_ratio.std(ddof=1)
    ci = stats.t.interval(.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
    sd_ci = np.sqrt((n - 1) * sd**2 / stats.chi2.ppf([.975, .025], n - 1))
    print(f"Paired teaching analysis: n={n}, geometric strong/medium={np.exp(mean):.3f}, 95% CI={np.exp(ci).round(3)}")
    print(f"Log ratio sample SD={sd:.3f}; normal-theory 95% SD interval={sd_ci.round(3)}")
    print(f"Log ratio median={np.median(log_ratio):.3f}; variance={sd**2:.3f}; range={np.ptp(log_ratio):.3f}")
    print(f"Log ratio skewness={stats.skew(log_ratio, bias=False):.3f}; excess kurtosis={stats.kurtosis(log_ratio, bias=False):.3f}")
    sign_nonzero = log_ratio[log_ratio != 0]
    paired = stats.ttest_1samp(log_ratio, 0)
    signed_rank = stats.wilcoxon(log_ratio)
    sign = stats.binomtest(np.count_nonzero(sign_nonzero > 0), len(sign_nonzero))
    median_ci = stats.bootstrap((log_ratio,), np.median, n_resamples=9999, method="percentile", random_state=2027)
    ratio_ci = np.exp([median_ci.confidence_interval.low, median_ci.confidence_interval.high])
    print(f"Median paired ratio={np.exp(np.median(log_ratio)):.3f}; bootstrap 95% CI={ratio_ci.round(3)}")
    print(f"Paired t p={paired.pvalue:.4g}; signed-rank p={signed_rank.pvalue:.4g}; sign p={sign.pvalue:.4g}")
    print(f"Normality check (D'Agostino K²) p={stats.normaltest(log_ratio).pvalue:.4g}")
    stats.probplot(log_ratio, dist="norm", plot=axes[1])
    axes[1].set(title="Normal Q–Q of paired log ratios", xlabel="Normal quantile", ylabel="Observed log ratio")

    # These two samples are generated independently on purpose; assays above are paired.
    rng = np.random.default_rng(2027)
    control = rng.normal(0, 1, 80)
    alternate = rng.normal(.6, 1.4, 65)
    z = control.mean() / (1 / np.sqrt(len(control)))
    z_p = 2 * stats.norm.sf(abs(z))
    welch = stats.ttest_ind(control, alternate, equal_var=False)
    pooled = stats.ttest_ind(control, alternate, equal_var=True)
    ranks = stats.mannwhitneyu(control, alternate, alternative="two-sided")
    edges = np.r_[-np.inf, stats.norm.ppf(np.arange(1, 8) / 8), np.inf]
    counts, _ = np.histogram(control, bins=edges)
    gof = stats.chisquare(counts, np.full(8, len(control) / 8))
    variance_ratio = np.var(control, ddof=1) / np.var(alternate, ddof=1)
    f_cdf = stats.f.cdf(variance_ratio, len(control) - 1, len(alternate) - 1)
    f_p = 2 * min(f_cdf, 1 - f_cdf)
    print(f"Synthetic independent examples: known-SD z p={z_p:.4f}; chi-square GOF p={gof.pvalue:.4f}; F p={f_p:.4f}")
    print(f"Welch p={welch.pvalue:.4g}; pooled t p={pooled.pvalue:.4g}; rank sum p={ranks.pvalue:.4g}")

    fig.savefig(args.output / "interlab.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
