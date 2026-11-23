"""Rebuild the four computational examples in Lesson 13."""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit, least_squares
from scipy.special import expit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]


def integrate(rhs, span, initial, sample_times, **options):
    """Fail visibly if the time integration fails."""
    solution = solve_ivp(
        rhs, span, initial, t_eval=sample_times, rtol=1e-8, atol=1e-10, **options
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def write_csv(path, headers, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(zip(*columns))


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=150)
    plt.close(figure)


def logistic(time, capacity, midpoint, scale):
    """R's SSlogis parameterization, evaluated with a stable sigmoid."""
    return capacity * expit((time - midpoint) / scale)


def yeast_growth(output):
    # Fudan 2025 Measurement, Table 4, BY4741 baseline.
    capacity, midpoint, scale = 3.67, 5.76, 1.3649
    growth_rate = 1 / scale
    start = logistic(0, capacity, midpoint, scale)
    times = np.linspace(0, 20, 201)
    curve = logistic(times, capacity, midpoint, scale)

    def rhs(_time, biomass):
        return growth_rate * biomass * (1 - biomass / capacity)

    numerical = integrate(rhs, (0, 20), [start], times).y[0]
    np.testing.assert_allclose(curve, numerical, atol=3e-7, rtol=2e-7)

    # The reported fitted curve is used as a source for synthetic observations.
    rng = np.random.default_rng(2713)
    observed_t = np.arange(17)
    synthetic = logistic(observed_t, capacity, midpoint, scale) + rng.normal(0, 0.015, 17)
    fit, _ = curve_fit(
        logistic, observed_t, synthetic,
        p0=(3.5, 6.0, 1.5), bounds=((0.1, 0, 0.05), (10, 25, 20)),
    )
    early = observed_t <= 4
    fitted_exponent, log_start = np.polyfit(observed_t[early], np.log(synthetic[early]), 1)
    early_exponential = np.exp(log_start + fitted_exponent * times)
    held_out = ~early
    log_fit_at_obs = logistic(observed_t, *fit)
    early_fit_at_obs = np.exp(log_start + fitted_exponent * observed_t)
    exponential_rmse = np.sqrt(np.mean((synthetic[held_out] - early_fit_at_obs[held_out]) ** 2))

    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    axis.plot(times, curve, label="Fudan Table 4 reconstruction")
    axis.plot(times, logistic(times, *fit), "--", label="Logistic fit to synthetic data")
    axis.scatter(observed_t, synthetic, s=12, color="black", label="Synthetic observations")
    axis.plot(times, early_exponential, ":", label="Exponential fit using hours 0-4")
    axis.axvline(4, color="0.75", linewidth=1)
    axis.set(xlabel="Time (h)", ylabel="Reported density proxy (1e8 / mL)", ylim=(0, 5))
    axis.legend()
    save_figure(figure, output, "fudan-growth")
    write_csv(
        output / "fudan-growth.csv",
        ["time_h", "table4_curve", "ode_solution", "synthetic_fit", "early_exponential"],
        [times, curve, numerical, logistic(times, *fit), early_exponential],
    )
    write_csv(
        output / "fudan-synthetic-observations.csv",
        ["time_h", "synthetic_observation", "used_in_exponential_fit", "logistic_fit"],
        [observed_t, synthetic, early, log_fit_at_obs],
    )
    print(f"Fudan: r={growth_rate:.4f}/h, N(0)={start:.4f}, N(t_mid)={capacity / 2:.3f}")
    print(f"  Synthetic logistic fit (K, t_mid, scale)={fit.round(3)}")
    print(f"  Early-exponential late RMSE={exponential_rmse:.2f}")


def fed_batch(output):
    # Illustrative inputs for the Cornell equations, with a benign generic culture.
    mu_max, half_saturation = 0.44, 0.45
    biomass_yield, product_yield, feed_substrate = 0.50, 0.15, 15.0
    start = np.array([0.20, 0, 4.0, 1.0])  # X, P, S (g/L), V (L)
    stages = [(0, 8, 0.0), (8, 14, 0.45), (14, 20, 0.0)]
    initial = start
    time_blocks, state_blocks, feed_blocks = [], [], []
    total_feed = 0.0

    for left, right, feed in stages:
        def rhs(_time, state):
            biomass, product, substrate, volume = state
            growth = mu_max * substrate / (half_saturation + substrate) * biomass
            dilution = feed / volume
            return [
                growth - dilution * biomass,
                product_yield * growth - dilution * product,
                dilution * (feed_substrate - substrate) - growth / biomass_yield,
                feed,
            ]

        times = np.linspace(left, right, round((right - left) * 20) + 1)
        solution = integrate(rhs, (left, right), initial, times)
        # Keep one sample at each boundary. The integrated state is continuous there.
        skip_boundary = 0 if left == 0 else 1
        time_blocks.append(times[skip_boundary:])
        state_blocks.append(solution.y[:, skip_boundary:])
        feed_blocks.append(total_feed + feed * (times[skip_boundary:] - left))
        total_feed += feed * (right - left)
        initial = solution.y[:, -1]

    time = np.concatenate(time_blocks)
    biomass, product, substrate, volume = np.concatenate(state_blocks, axis=1)
    feed_volume = np.concatenate(feed_blocks)
    biomass_mass, product_mass, substrate_mass = biomass * volume, product * volume, substrate * volume
    substrate_added = feed_volume * feed_substrate
    biomass_gained = biomass_mass - start[0] * start[3]
    substrate_used = start[2] * start[3] + substrate_added - substrate_mass
    np.testing.assert_allclose(biomass_gained, biomass_yield * substrate_used, atol=1e-6)
    np.testing.assert_allclose(product_mass, product_yield * biomass_gained, atol=1e-7)
    np.testing.assert_allclose(volume, start[3] + feed_volume, atol=1e-8)

    figure, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True, layout="constrained")
    axes[0].plot(time, biomass, label="Biomass concentration")
    axes[0].plot(time, product, label="Product concentration")
    axes[0].set_ylabel("Concentration (g/L)")
    axes[1].plot(time, biomass_mass, label="Total biomass")
    axes[1].plot(time, product_mass, label="Total product")
    axes[1].set_ylabel("Mass (g)")
    axes[2].plot(time, substrate, label="Substrate (g/L)")
    axes[2].plot(time, volume, label="Volume (L)")
    axes[2].set(xlabel="Time (h)", ylabel="Value (see legend)")
    for axis in axes:
        for left, right, feed in stages:
            if feed > 0:
                axis.axvspan(left, right, color="0.92", zorder=0)
        axis.legend(loc="best")
    save_figure(figure, output, "cornell-fed-batch")
    write_csv(
        output / "cornell-fed-batch.csv",
        [
            "time_h", "biomass_g_L", "product_g_L", "substrate_g_L", "volume_L",
            "biomass_g", "product_g", "substrate_g", "feed_added_L",
        ],
        [time, biomass, product, substrate, volume, biomass_mass, product_mass, substrate_mass, feed_volume],
    )
    print(f"Cornell: final volume={volume[-1]:.2f} L, final biomass={biomass_mass[-1]:.2f} g, product={product_mass[-1]:.2f} g")


def competition(output):
    source = REPO / "data" / "gause-competition-mixture.csv"
    with source.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    observed_t = np.array([int(row["day"]) for row in rows])
    abundance = np.array([[float(row["caudatum"]), float(row["aurelia"])] for row in rows]).T
    fit_through_day = 17
    training = observed_t <= fit_through_day
    times = np.linspace(observed_t[0], observed_t[-1], 221)
    initial = abundance[:, 0]

    def simulate(sample_times, start_t, start_abundance, growth, interaction):
        def rhs(_time, current):
            return current * (growth + interaction @ current)

        return integrate(
            rhs, (start_t, sample_times[-1]), start_abundance, sample_times
        ).y

    def constrained_fit(coupled):
        def unpack(parameters):
            growth = parameters[:2]
            self_limits = -parameters[2:4] / 100
            other_limits = -parameters[4:6] / 100 if coupled else np.zeros(2)
            interaction = np.array([
                [self_limits[0], other_limits[0]],
                [other_limits[1], self_limits[1]],
            ])
            return growth, interaction

        def residual(parameters):
            predictions = simulate(
                observed_t[training], observed_t[0], initial, *unpack(parameters)
            )
            return (predictions - abundance[:, training]).ravel()

        guess = [0.7, 0.7, 0.5, 0.5, 0.5, 0.5] if coupled else [0.7, 0.7, 0.5, 0.5]
        result = least_squares(residual, guess, bounds=(0, 4))
        if not result.success:
            raise RuntimeError(result.message)
        return unpack(result.x)

    # Rounded published estimates in Mühlbauer et al. 2020, Table 2(b).
    authors_growth = np.array([1.26, 1.03])
    authors_interaction = np.array([[-0.005, -0.008], [-0.002, -0.007]])
    paper_curve = simulate(
        times, 0.0, [1.60, 1.63], authors_growth, authors_interaction
    )

    fitted_growth, fitted_interaction = constrained_fit(coupled=True)
    coupled_curve = simulate(times, observed_t[0], initial, fitted_growth, fitted_interaction)
    independent_growth, independent_interaction = constrained_fit(coupled=False)
    independent_curve = simulate(times, observed_t[0], initial, independent_growth, independent_interaction)

    prediction = simulate(observed_t, observed_t[0], initial, fitted_growth, fitted_interaction)
    baseline = simulate(observed_t, observed_t[0], initial, independent_growth, independent_interaction)
    coupled_rmse = np.sqrt(np.mean((prediction[:, ~training] - abundance[:, ~training]) ** 2, axis=1))
    independent_rmse = np.sqrt(np.mean((baseline[:, ~training] - abundance[:, ~training]) ** 2, axis=1))

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
    for index, (axis, species) in enumerate(zip(axes, ("P. caudatum", "P. aurelia"))):
        axis.plot(times, coupled_curve[index], label="Coupled, fitted through day 17")
        axis.plot(times, independent_curve[index], ":", label="Independent, fitted through day 17")
        axis.plot(times, paper_curve[index], "--", color="0.5", label="Paper, rounded coefficients")
        axis.scatter(
            observed_t[training], abundance[index, training],
            s=13, color="black", label="Training data",
        )
        axis.scatter(
            observed_t[~training], abundance[index, ~training],
            s=20, edgecolors="black", facecolors="none", label="Holdout data",
        )
        axis.axvline(fit_through_day, color="0.75", linewidth=1)
        axis.set(xlabel="Day", title=species)
    axes[0].set_ylabel("Size-standardized volume (original scale)")
    axes[1].legend(fontsize=8)
    save_figure(figure, output, "gause-competition")
    write_csv(
        output / "gause-competition.csv",
        [
            "day", "caudatum", "aurelia", "fit_caudatum", "fit_aurelia",
            "independent_caudatum", "independent_aurelia", "training_day",
        ],
        [observed_t, *abundance, *prediction, *baseline, training],
    )
    print(
        f"Gause: holdout RMSE [caudatum, aurelia] coupled={coupled_rmse.round(2)}, "
        f"independent={independent_rmse.round(2)}"
    )
    print(f"  Teaching interaction matrix rows [caudatum, aurelia]: {fitted_interaction.tolist()}")


def sinking(output):
    water_density, viscosity, gravity = 1020.0, 1.07e-3, 9.81
    # Instructor-selected representative radius and buoyant mass.
    radius = 5e-6  # m
    buoyant_mass = 40e-15  # kg, 1 pg = 1e-15 kg
    volume = (4 / 3) * np.pi * radius**3
    density = water_density + buoyant_mass / volume
    drag = 6 * np.pi * viscosity * radius
    terminal_speed = buoyant_mass * gravity / drag
    relaxation = density * volume / drag
    reynolds = water_density * terminal_speed * 2 * radius / viscosity

    times = np.linspace(0, 8, 161)
    analytical = -np.expm1(-times)

    def scaled_rhs(_time, state):
        # time in units of the relaxation time, speed in units of terminal speed
        return 1 - state

    def reach_95_percent(_time, state):
        return state[0] - 0.95

    reach_95_percent.direction = 1
    numerical = integrate(
        scaled_rhs, (0, 8), [0], times, events=reach_95_percent
    )
    np.testing.assert_allclose(numerical.y[0], analytical, atol=1e-8, rtol=5e-8)
    np.testing.assert_allclose(
        terminal_speed, 2 * (density - water_density) * gravity * radius**2 / (9 * viscosity)
    )
    reach_time = numerical.t_events[0][0] * relaxation
    walk_depth = 1e-3  # m
    # At this timescale the initial acceleration lag is negligible.
    depth_time_at_terminal = walk_depth / terminal_speed

    # Keep the density contrast fixed to reproduce the theoretical scaling.
    volume_relative = np.geomspace(0.05, 20, 80)
    speed_relative = volume_relative ** (2 / 3)
    slope, _intercept = np.polyfit(np.log(volume_relative), np.log(speed_relative), 1)
    np.testing.assert_allclose(slope, 2 / 3, atol=1e-12)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    axes[0].plot(times, analytical, label="Linear Stokes drag")
    axes[0].plot(times, np.tanh(times), "--", label="Quadratic drag, normalized comparison")
    axes[0].axhline(0.95, color="0.65", linewidth=1)
    axes[0].set(xlabel="Dimensionless time", ylabel="Velocity / terminal velocity")
    axes[0].legend(fontsize=8)
    axes[1].loglog(volume_relative, speed_relative)
    axes[1].set(xlabel="Relative cell volume", ylabel="Relative sinking velocity", title="Fixed density contrast")
    save_figure(figure, output, "algae-sinking")
    write_csv(
        output / "algae-sinking.csv",
        ["time_per_relaxation", "linear_drag_speed_relative", "quadratic_drag_speed_relative"],
        [times, analytical, np.tanh(times)],
    )
    print(f"Algae example: terminal speed={terminal_speed * 1e6:.3f} um/s, Re={reynolds:.2e}")
    print(f"  95% speed at {reach_time * 1e6:.2f} us, 1 mm at terminal speed in {depth_time_at_terminal:.1f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("build") / "13-differential-equations")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fed_batch(args.outdir)
    yeast_growth(args.outdir)
    competition(args.outdir)
    sinking(args.outdir)
    print(f"Outputs: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
