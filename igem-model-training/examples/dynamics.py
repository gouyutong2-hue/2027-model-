"""Run the Chap01 dynamics labs with Python, NumPy, SciPy and Matplotlib."""

import argparse
import csv
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA = Path(__file__).resolve().parents[1] / "data"


def trajectory(rhs, time, initial, method="RK45"):
    solution = solve_ivp(
        rhs, (time[0], time[-1]), initial, t_eval=time,
        method=method, rtol=1e-8, atol=1e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)


def michaelis_menten(substrate, km, vmax):
    return vmax * substrate / (km + substrate)


def enzyme_lab(out):
    # Figure 6 published curve, TAS_Taipei 2021. No fitted data points here.
    source_km, source_vmax = 0.3762, 0.0345

    # Simulated colorimeter observations. A = blank + calibration * product.
    rng = np.random.default_rng(22)
    substrate = np.geomspace(0.08, 3.0, 12)
    time = np.linspace(0.0, 1.0, 7)
    calibration, blank = 2.1, 0.025
    true_km, true_vmax = 0.40, 0.036
    rates = michaelis_menten(substrate, true_km, true_vmax)
    absorbance = blank + calibration * rates[:, None] * time
    absorbance += rng.normal(0, 0.0018, absorbance.shape)
    estimated_rates = np.array([
        np.polyfit(time, (row - blank) / calibration, 1)[0]
        for row in absorbance
    ])
    slope, intercept = np.polyfit(1 / substrate, 1 / estimated_rates, 1)
    reciprocal = np.array([slope / intercept, 1 / intercept])
    direct, _ = curve_fit(michaelis_menten, substrate, estimated_rates, bounds=(0, np.inf))
    print(f"Enzyme simulated true Km/Vmax: {true_km:.3g}, {true_vmax:.3g}")
    for label, fit in [("double reciprocal", reciprocal), ("direct fit", direct)]:
        rmse = np.sqrt(np.mean((estimated_rates - michaelis_menten(substrate, *fit)) ** 2))
        print(f"  {label}: {fit[0]:.4f}, {fit[1]:.5f}; rate RMSE={rmse:.5f} mM/min")

    # A distinct simulated full reaction: E + S <-> ES -> E + P.
    # State units mM; rates k_on in 1/(mM min), k_off and k_cat in 1/min.
    k_on, k_off, k_cat = 8.0, 2.0, 1.2
    e_total, s_initial = 0.030, 0.6

    def elementary(_time, state):
        e, s, es, _p = state
        binding = k_on * e * s - k_off * es
        catalysis = k_cat * es
        return [-binding + catalysis, -binding, binding - catalysis, catalysis]

    kinetics_time = np.linspace(0, 18, 301)
    e, s, es, p = trajectory(elementary, kinetics_time, [e_total, s_initial, 0.0, 0.0])
    np.testing.assert_allclose(e + es, e_total, atol=1e-9)
    np.testing.assert_allclose(s + es + p, s_initial, atol=1e-9)
    km, vmax = (k_off + k_cat) / k_on, k_cat * e_total
    reduced = trajectory(
        lambda _t, state: [-michaelis_menten(state[0], km, vmax)],
        kinetics_time, [s_initial],
    )[0]

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    grid = np.linspace(0, 3, 200)
    ax[0].plot(grid, michaelis_menten(grid, source_km, source_vmax), label="Published-fit redraw")
    ax[0].scatter([source_km], [source_vmax / 2], label="Computed half rate")
    ax[0].set(title="TAS Taipei 2021, Fig. 6 equation", xlabel="Substrate (mM)", ylabel="Rate (mM/min)")
    ax[1].scatter(substrate, estimated_rates, s=22, label="Simulated observations")
    ax[1].plot(grid, michaelis_menten(grid, *reciprocal), "--", label="Double reciprocal")
    ax[1].plot(grid, michaelis_menten(grid, *direct), label="Direct fit")
    ax[1].set(title="Fit on simulated data", xlabel="Substrate (mM)", ylabel="Rate (mM/min)")
    ax[2].plot(kinetics_time, p, label="Elementary reactions")
    ax[2].plot(kinetics_time, s_initial - reduced, "--", label="Quasi-steady-state model")
    ax[2].plot(kinetics_time, es, label="Enzyme-substrate complex")
    ax[2].set(title="Simulated mass balances", xlabel="Time (min)", ylabel="Concentration (mM)")
    for panel in ax:
        panel.legend(fontsize=8)
    save(fig, out, "dynamics-enzyme")


def hill(input_level, half_effect, steepness):
    ratio = (input_level / half_effect) ** steepness
    return ratio / (1 + ratio)


def reporter_lab(out):
    # Dimensionless inputs and output units. All values are teaching assumptions.
    time = np.linspace(0, 24, 241)
    inputs = [0.3, 1.0, 3.0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    dose = np.geomspace(0.03, 10, 200)
    for exponent in [1, 2, 4]:
        ax[0].semilogx(dose, hill(dose, 1, exponent), label=f"n={exponent}")

    for level in inputs:
        stimulus = hill(level, 1, 3)

        def reporter(_time, state):
            mrna, dark, mature = state
            maturation = 0.35 * dark
            return [1.4 * stimulus - 0.9 * mrna,
                    0.8 * mrna - 0.16 * dark - maturation,
                    maturation - 0.10 * mature]

        signal = trajectory(reporter, time, [0, 0, 0])[2]
        ax[1].plot(time, signal, label=f"Input {level:g}")
        print(f"Reporter simulated input {level:g}: terminal fluorescence {signal[-1]:.3f} a.u.")

    # Continuous interpolation of the truth table (A or B) and not C.
    # The input activities are independent and dimensionless in this exercise.
    def regulator(activities):
        expression = 0.0
        for active in product((0, 1), repeat=3):
            weight = np.prod(np.where(active, activities, 1 - activities))
            on = (active[0] or active[1]) and not active[2]
            expression += float(on) * weight
        return expression

    for states in product((0, 1), repeat=3):
        assert regulator(np.array(states)) == float((states[0] or states[1]) and not states[2])
    for c in [0.05, 0.5, 0.95]:
        a_range = np.linspace(0, 1, 101)
        ax[2].plot(a_range, [regulator(np.array([a, 0.35, c])) for a in a_range],
                   label=f"C activity {c:g}")
    ax[0].set(title="Hill dose response", xlabel="Relative input", ylabel="Activity")
    ax[1].set(title="Simulated mature reporter", xlabel="Time (teaching units)", ylabel="Signal (a.u.)")
    ax[2].set(title="BoolODE-style truth-table interpolation", xlabel="A activity", ylabel="Expression (0-1)")
    for panel in ax:
        panel.legend(fontsize=8)
    save(fig, out, "dynamics-reporter")


def growth_lab(out):
    time = np.linspace(0, 30, 301)
    initial, mu_max, limit, half = 0.05, 0.65, 1.6, 0.25
    start_substrate, yield_x_s = 1.20, 0.8

    def model(name):
        def rhs(_time, state):
            live, dead, lysed, substrate = state
            if name == "Logistic":
                growth = mu_max * live * (1 - live / limit)
                return [growth, 0, 0, 0]

            # Only tiny solver excursions across the S=0 boundary are clamped.
            substrate = max(substrate, 0)
            if name == "Contois":
                denominator = half * live + substrate
                rate = mu_max * substrate / denominator if denominator > 0 else 0
            else:
                rate = mu_max * substrate / (half + substrate)
            if name == "Hybrid + death":
                rate *= max(0, 1 - live / limit)
            growth = rate * live
            death = 0.025 * live if name == "Hybrid + death" else 0
            lysis = 0.08 * dead
            return [growth - death, death - lysis, lysis, -growth / yield_x_s]
        return rhs

    fig, panels = plt.subplots(2, 2, figsize=(10, 7))
    ax = panels.ravel()
    for name in ["Logistic", "Monod", "Contois", "Hybrid + death"]:
        live, dead, lysed, substrate = trajectory(
            model(name), time, [initial, 0, 0, start_substrate],
        )
        assert np.min([live, dead, lysed, substrate]) > -1e-8
        if name != "Logistic":
            np.testing.assert_allclose(live + dead + lysed + yield_x_s * substrate,
                                       initial + yield_x_s * start_substrate, atol=1e-8)
        ax[0].plot(time, live, label=name)
        ax[1].plot(time, substrate, label=name)
        if name == "Hybrid + death":
            ax[2].plot(time, live, label="Live")
            ax[2].plot(time, dead, label="Dead")
            ax[2].plot(time, lysed, label="Lysed biomass equivalent")
        print(f"Growth simulated {name}: terminal live biomass={live[-1]:.3f} g/L")
    early_time = time[time <= 5]
    ax[0].plot(early_time, initial * np.exp(mu_max * early_time), ":", label="Exponential, early")
    # A local sensitivity sweep in normalized nutrient units at fixed light.
    # n and p are each an initial concentration divided by its own Monod K.
    nutrient = np.geomspace(0.1, 20, 121)
    lower, higher = 0.5, 3.0
    fixed_light = 0.8
    ax[3].semilogx(nutrient, mu_max * fixed_light * nutrient / (1 + nutrient) * higher / (1 + higher),
                   label="Vary N; P fixed at 3")
    ax[3].semilogx(nutrient, mu_max * fixed_light * lower / (1 + lower) * nutrient / (1 + nutrient),
                   label="Vary P; N fixed at 0.5")
    ax[0].set(title="Simulated growth models", xlabel="Time (h)", ylabel="Live biomass (g/L)")
    ax[1].set(title="Simulated substrate balance", xlabel="Time (h)", ylabel="Substrate (g/L)")
    ax[2].set(title="Hybrid model compartments", xlabel="Time (h)", ylabel="Biomass equivalents (g/L)")
    ax[3].set(title="Simulated initial Monod sensitivity", xlabel="N or P relative to its own K",
              ylabel="Specific growth rate (1/h)")
    for panel in ax:
        panel.legend(fontsize=8)
    save(fig, out, "dynamics-growth")


def fixed_step(rhs, time, initial, method):
    values = np.empty((len(time), len(initial)))
    values[0] = initial
    for i, (t, h) in enumerate(zip(time[:-1], np.diff(time))):
        y = values[i]
        k1 = np.asarray(rhs(t, y))
        if method == "Euler":
            values[i + 1] = y + h * k1
        elif method == "Heun":
            k2 = np.asarray(rhs(t + h, y + h * k1))
            values[i + 1] = y + h * (k1 + k2) / 2
        elif method == "RK4":
            k2 = np.asarray(rhs(t + h / 2, y + h * k1 / 2))
            k3 = np.asarray(rhs(t + h / 2, y + h * k2 / 2))
            k4 = np.asarray(rhs(t + h, y + h * k3))
            values[i + 1] = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        else:
            raise ValueError(f"Unsupported method {method}")
    return values.T


def solver_lab(out):
    # Exact decay solution y(t) = exp(-t) provides an independent reference.
    rhs = lambda _t, y: -y
    steps = np.array([20, 40, 80, 160, 320])
    step_size = 4 / steps
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4))
    for method in ["Euler", "Heun", "RK4"]:
        error = [
            abs(fixed_step(rhs, np.linspace(0, 4, n + 1), [1.0], method)[0, -1] - np.exp(-4))
            for n in steps
        ]
        ax[0].loglog(step_size, error, ".-", label=method)
        print(f"Solver {method}: observed order {np.polyfit(np.log(step_size), np.log(error), 1)[0]:.2f}")
    ax[0].loglog(step_size, abs((1 + step_size) ** (-steps) - np.exp(-4)),
                 ".-", label="Backward Euler")
    ax[0].loglog(step_size,
                 abs(((1 - step_size / 2) / (1 + step_size / 2)) ** steps - np.exp(-4)),
                 ".-", label="Implicit trapezoid")
    # For y'=-20y, Euler with h=0.2 has amplification |1-20h|=3.
    count = np.arange(11)
    h = 0.2
    for label, factor in [("Euler", 1 - 20 * h), ("Backward Euler", 1 / (1 + 20 * h)),
                          ("Implicit trapezoid", (1 - 10 * h) / (1 + 10 * h))]:
        ax[1].semilogy(h * count, abs(factor) ** count, ".-", label=label)
    ax[1].semilogy(h * count, np.exp(-20 * h * count), "--", label="Exact")
    ax[0].set(title="Error against exact decay", xlabel="Step size", ylabel="Absolute terminal error")
    ax[1].set(title="Large-step stability check", xlabel="Time", ylabel="Absolute state")
    for panel in ax:
        panel.legend(fontsize=8)
    save(fig, out, "dynamics-solvers")


def community_lab(out):
    with (DATA / "gause-1935-figure3.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    prey = [row for row in rows if row["Species"] == "Saccharomyces exiguus"]
    predator = [row for row in rows if row["Species"] == "Paramecium aurelia"]
    time = np.array([float(row["Time"]) for row in prey])
    np.testing.assert_array_equal(time, [float(row["Time"]) for row in predator])
    observed = np.array([[float(row["Individuals"]) for row in species] for species in [prey, predator]])
    training = time <= 11
    scales = observed[:, training].max(axis=1)
    normalized = observed / scales[:, None]

    def model(params):
        a, b, c, d = params

        def rhs(_time, state):
            x, y = state
            return [a * x - b * x * y, -c * y + d * x * y]

        return rhs

    def residuals(params):
        fitted = trajectory(model(params), time[training], normalized[:, 0])
        return (fitted[:, 1:] - normalized[:, training][:, 1:]).ravel()

    trials = [
        least_squares(residuals, start, bounds=(0.015, 7))
        for start in [(0.60, 3.0, 0.60, 3.0), (0.80, 2.0, 0.80, 2.0), (0.45, 4.0, 0.45, 4.0)]
    ]
    best = min((trial for trial in trials if trial.success), key=lambda trial: trial.cost)
    dense_time = np.linspace(0, 19, 501)
    lv = trajectory(model(best.x), dense_time, normalized[:, 0]) * scales[:, None]
    estimated = trajectory(model(best.x), time, normalized[:, 0])
    for label, subset in [("train (day 0 excluded)", training & (time != 0)), ("holdout", ~training)]:
        error = (estimated[:, subset] - normalized[:, subset]) * scales[:, None]
        print(f"Gause digitized {label} RMSE: prey/predator {np.sqrt(np.mean(error**2, axis=1))}")

    # Separate simulated two-species competition scenario (dimensionless densities).
    def competition(_t, state, overlap):
        x, y = state
        return [0.5 * x * (1 - x - overlap * y), 0.6 * y * (1 - y - 0.65 * x)]

    competition_time = np.linspace(0, 100, 501)
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for i, label in enumerate(["Yeast / 0.1 mm³", "Paramecium / 15 cm³"]):
        panel = ax[0, i]
        fit_period = dense_time <= 11
        panel.plot(dense_time[fit_period], lv[i, fit_period], color="C0", label="Fitted trajectory")
        panel.plot(dense_time[~fit_period], lv[i, ~fit_period], "--", color="C0", label="Extrapolation")
        panel.scatter(time[training], observed[i, training], s=18, label="Digitized, training")
        panel.scatter(time[~training], observed[i, ~training], s=22, marker="x", label="Digitized, holdout")
        panel.axvline(11, color="0.6", linewidth=0.8)
        panel.set(title="Gause (1935), Figure 3", xlabel="Day", ylabel=label)
        panel.legend(fontsize=8, loc="upper center")
    ax[1, 0].plot(*lv, label="Predicted trajectory")
    ax[1, 0].scatter(*observed, s=18, alpha=0.8, label="Digitized observations")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set(title="Predator-prey phase plane", xlabel="Yeast / 0.1 mm³", ylabel="Paramecium / 15 cm³")
    for overlap in [0.45, 1.3]:
        series = trajectory(lambda t, z: competition(t, z, overlap), competition_time, [0.10, 0.13])
        ax[1, 1].plot(competition_time, series[0], label=f"A, alpha={overlap:g}")
        ax[1, 1].plot(competition_time, series[1], "--", label=f"B, alpha={overlap:g}")
        print(f"Competition simulated alpha={overlap:g}: terminal A/B={series[:, -1]}")
    ax[1, 1].set(title="Simulated competition", xlabel="Time", ylabel="Relative density")
    ax[1, 1].legend(fontsize=8)
    save(fig, out, "dynamics-community")


def main():
    labs = {"enzyme": enzyme_lab, "reporter": reporter_lab, "growth": growth_lab,
            "community": community_lab, "solvers": solver_lab}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", default="all", choices=["all", *labs])
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    for name, lab in labs.items():
        if args.lab in ("all", name):
            lab(args.out)


if __name__ == "__main__":
    main()
